#pragma once

#include "ds4_images.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::ds4 {

// An exact-prefix index over a physical token sequence. It answers "how much of
// this request is the same history" for live sequences and the cold prefix tier;
// it never aligns edited histories (whole live continuations do not need it).

namespace context_history_detail {

// A modality is an indivisible atom with a content identity and a model width.
// Text uses one-token atoms. Matching never relies on image placeholder IDs.
struct Atom {
    size_t start;
    size_t width;
    uint64_t identity;
    bool image;

    bool operator==(const Atom& other) const {
        return image == other.image && width == other.width && identity == other.identity;
    }
};

}  // namespace context_history_detail

// This index belongs to the physical token sequence, not a scheduler candidate.
// Image placeholders are validated once on ingestion and represented by a single
// atom carrying both their width and pixel identity. No source tokens are owned.
class ContextHistoryIndex {
public:
    struct Fingerprint {
        uint64_t first = 0;
        uint64_t second = 0;
        bool operator==(const Fingerprint& other) const {
            return first == other.first && second == other.second;
        }
    };

    ContextHistoryIndex() { clear(); }

    void clear() {
        atoms_.clear();
        hashes_.assign(1, {});
        powers_.assign(1, {1, 1});
        text_counts_.assign(1, 0);
        size_ = 0;
        valid_ = false;
    }

    void assign(const std::vector<int32_t>& tokens, const std::vector<ImageSpan>& images,
                int image_token, bool index_anchors = true) {
        clear();
        index_anchors_ = index_anchors;
        extend(tokens, images, image_token);
    }

    // The caller guarantees that the existing common token/metadata prefix has
    // not changed. Only new tokens and images are checked; a shorter sequence
    // removes its suffix.
    void extend(const std::vector<int32_t>& tokens, const std::vector<ImageSpan>& images,
                int image_token) {
        if (tokens.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            clear();
            return;
        }
        if (valid_ && image_token_ != image_token) {
            assign(tokens, images, image_token, index_anchors_);
            return;
        }
        if (valid_ && tokens.size() == size_) return;
        if (!valid_) clear();
        image_token_ = image_token;
        size_t keep = atoms_.size();
        const size_t begin = std::min(size_, tokens.size());
        if (begin < size_) {
            keep = boundary(begin);
            if (keep == npos) {
                clear();
                return;
            }
        }
        atoms_.resize(keep);
        hashes_.resize(keep + 1);
        powers_.resize(keep + 1);
        text_counts_.resize(keep + 1);

        const size_t known_images = keep - text_counts_.back();
        if (known_images > images.size()) {
            clear();
            return;
        }
        auto image = images.begin() + known_images;
        for (size_t pos = begin; pos < tokens.size();) {
            context_history_detail::Atom atom{pos, 1, static_cast<uint64_t>(tokens[pos]), false};
            if (image != images.end() && image->start < static_cast<int64_t>(tokens.size())) {
                if (image->start < static_cast<int64_t>(pos) || image->len <= 0 ||
                    image->len > static_cast<int64_t>(tokens.size()) - image->start) {
                    clear();
                    return;
                }
                if (image->start == static_cast<int64_t>(pos)) {
                    atom = {pos, static_cast<size_t>(image->len), image->hash, true};
                    for (size_t j = 0; j < atom.width; ++j) {
                        if (tokens[pos + j] != image_token) {
                            clear();
                            return;
                        }
                    }
                    ++image;
                }
            }
            if (!atom.image && tokens[pos] == image_token) {
                clear();
                return;
            }
            const auto value = fingerprint(atom);
            hashes_.push_back({hashes_.back().first * base_first + value.first,
                               hashes_.back().second * base_second + value.second});
            powers_.push_back({powers_.back().first * base_first, powers_.back().second * base_second});
            text_counts_.push_back(text_counts_.back() + (atom.image ? 0 : 1));
            atoms_.push_back(atom);
            pos += atom.width;
        }
        // Live model metadata can already contain pictures in the unevaluated
        // prompt suffix. Request (hash-only) indices instead require all spans.
        if (image != images.end() &&
            (!index_anchors_ || image->start < static_cast<int64_t>(tokens.size()))) {
            clear();
            return;
        }
        size_ = tokens.size();
        valid_ = true;
    }

    bool valid() const { return valid_; }
    bool valid_for(int image_token) const { return valid_ && image_token_ == image_token; }
    size_t size() const { return size_; }
    size_t atom_count() const { return atoms_.size(); }
    const context_history_detail::Atom& atom_at(size_t pos) const { return atoms_[pos]; }
    Fingerprint prefix_fingerprint(size_t atom_count) const { return hashes_[atom_count]; }
    size_t token_position(size_t pos) const { return pos == atoms_.size() ? size_ : atoms_[pos].start; }
    static constexpr size_t npos = std::numeric_limits<size_t>::max();

    // Token checkpoints may fall between atoms, but never inside an image.
    size_t boundary(size_t tokens) const {
        if (tokens == size_) return atoms_.size();
        if (tokens > size_) return npos;
        const auto found = std::lower_bound(atoms_.begin(), atoms_.end(), tokens,
            [](const context_history_detail::Atom& atom, size_t pos) { return atom.start < pos; });
        return found != atoms_.end() && found->start == tokens
            ? static_cast<size_t>(found - atoms_.begin()) : npos;
    }

    Fingerprint range(size_t begin, size_t end) const {
        const auto& power = powers_[end - begin];
        return {hashes_[end].first - hashes_[begin].first * power.first,
                hashes_[end].second - hashes_[begin].second * power.second};
    }

    // Binary-search equal runs, rather than comparing every token of every live
    // candidate. Callers verify a hit against the actual tokens before reuse.
    size_t common_atoms(size_t begin, size_t end, const ContextHistoryIndex& other,
                        size_t incoming) const {
        size_t low = 0, high = std::min(end - begin, other.atom_count() - incoming);
        while (low < high) {
            const size_t middle = low + (high - low + 1) / 2;
            if (range(begin, begin + middle) == other.range(incoming, incoming + middle)) low = middle;
            else high = middle - 1;
        }
        return low;
    }

    size_t common_prefix(const ContextHistoryIndex& other) const {
        if (!valid_ || !other.valid_) return 0;
        return token_position(common_atoms(0, atom_count(), other, 0));
    }

private:
    static constexpr uint64_t base_first = 0x9e3779b185ebca87ULL;
    static constexpr uint64_t base_second = 0xc2b2ae3d27d4eb4fULL;

    static uint64_t mix(uint64_t value) {
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
    static Fingerprint fingerprint(const context_history_detail::Atom& atom) {
        const uint64_t kind = atom.image ? 0xd6e8feb86659fd93ULL : 0xa0761d6478bd642fULL;
        return {mix(atom.identity ^ kind) + mix(atom.width ^ base_first),
                mix(atom.identity + kind) + mix(atom.width ^ base_second)};
    }

    std::vector<context_history_detail::Atom> atoms_;
    std::vector<Fingerprint> hashes_;
    std::vector<Fingerprint> powers_;
    std::vector<size_t> text_counts_;
    size_t size_ = 0;
    int image_token_ = 0;
    bool valid_ = false;
    bool index_anchors_ = true;
};

}  // namespace strata::ds4
