#pragma once

#include "ds4_history_projection.hpp"
#include "ds4_images.hpp"

#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

namespace strata::ds4 {

// Historical presentation folds preserve the original causal model positions.
// Global cache reuse is controlled by the caller, not a modality-specific policy.
inline constexpr size_t context_history_anchor_tokens = 8;
inline constexpr size_t context_history_annotation_tokens = 256;

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

template <typename Token>
inline bool atoms(const std::vector<Token>& tokens, const std::vector<ImageSpan>& images,
                  int image_token, size_t limit, std::vector<Atom>& out) {
    if (limit > tokens.size() || tokens.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !image_spans_valid(images, static_cast<int64_t>(tokens.size()))) return false;
    size_t image = 0;
    for (size_t pos = 0; pos < tokens.size();) {
        if (image < images.size() && static_cast<size_t>(images[image].start) == pos) {
            const ImageSpan& s = images[image++];
            const size_t width = static_cast<size_t>(s.len);
            for (size_t i = 0; i < width; ++i)
                if (tokens[pos + i] != image_token) return false;
            if (pos < limit) {
                if (width > limit - pos) return false;  // No checkpoint inside a modality.
                out.push_back({pos, width, s.hash, true});
            }
            pos += width;
        } else {
            if (tokens[pos] == image_token) return false;
            if (pos < limit) out.push_back({pos, 1, static_cast<uint64_t>(tokens[pos]), false});
            ++pos;
        }
    }
    return true;
}

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
        keys_.clear();
        positions_.clear();
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
    // removes its suffix, including obsolete anchor positions. At most seven
    // surviving tail windows need rebuilding on either append or truncation.
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
        const size_t rebuild = keep > context_history_anchor_tokens - 1
            ? keep - (context_history_anchor_tokens - 1) : 0;
        if (index_anchors_) {
            for (size_t i = keys_.size(); i > rebuild; --i) {
                const auto& key = keys_[i - 1];
                erase_last(key);
                if (key.count > 1 && atoms_[i + key.count - 2].image)
                    erase_last({range(i - 1, i + key.count - 2), key.count - 1});
            }
            keys_.resize(rebuild);
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
        if (index_anchors_) {
            for (size_t i = rebuild; i < atoms_.size(); ++i) {
                const auto key = anchor(i, atoms_.size());
                keys_.push_back(key);
                positions_[key].push_back(i);
                // A short retained run between independently omitted pictures
                // must not require the following picture to remain visible.
                // Index that text-only boundary as a weaker anchor as well.
                if (key.count > 1 && atoms_[i + key.count - 1].image)
                    positions_[{range(i, i + key.count - 1), key.count - 1}].push_back(i);
            }
        }
    }

    bool valid() const { return valid_; }
    bool valid_for(int image_token) const { return valid_ && image_token_ == image_token; }
    size_t size() const { return size_; }
    size_t atom_count() const { return atoms_.size(); }
    const context_history_detail::Atom& atom(size_t pos) const { return atoms_[pos]; }
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

    size_t floor_boundary(size_t tokens) const {
        if (tokens >= size_) return atoms_.size();
        const auto found = std::upper_bound(atoms_.begin(), atoms_.end(), tokens,
            [](size_t pos, const context_history_detail::Atom& atom) { return pos < atom.start; });
        return found == atoms_.begin() ? 0 : static_cast<size_t>(found - atoms_.begin() - 1);
    }

    Fingerprint range(size_t begin, size_t end) const {
        const auto& power = powers_[end - begin];
        return {hashes_[end].first - hashes_[begin].first * power.first,
                hashes_[end].second - hashes_[begin].second * power.second};
    }

    // Binary-search equal runs, rather than comparing every token of every live
    // candidate (or every checkpoint). Hash hits are checked exactly on apply.
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

    std::pair<size_t, size_t> find_anchor(size_t begin, size_t end, size_t limit,
                                       const ContextHistoryIndex& request, size_t incoming) const {
        std::pair<size_t, size_t> best{npos, npos};
        size_t strength = 0;
        // Only the bounded annotation is walked. Each key performs one position
        // lower_bound, never a scan through repeated matches in a long history.
        for (size_t at = incoming; at < request.atom_count(); ++at) {
            const auto longest = request.anchor(at, request.atom_count());
            for (size_t count = longest.count; count != 0; --count) {
                const Key key{request.range(at, at + count), count};
                size_t candidate = npos;
                const auto found = positions_.find(key);
                if (found != positions_.end()) {
                    const auto pos = std::lower_bound(found->second.begin(), found->second.end(), begin);
                    if (pos != found->second.end() && *pos <= end && count <= limit - *pos)
                        candidate = *pos;
                }
                // A checkpoint has a virtual short tail not present in the
                // full-sequence anchor table. There is only one possible tail
                // for each width, so checking it never scans cached candidates.
                if (count < context_history_anchor_tokens && count <= limit) {
                    const size_t tail = limit - count;
                    if (tail >= begin && tail <= end && tail < candidate &&
                        anchor(tail, limit) == key) candidate = tail;
                }
                if (candidate == npos) continue;
                const size_t score = request.atom(at + count - 1).image
                    ? context_history_anchor_tokens : count;
                if (score > strength || (score == strength && candidate < best.first)) {
                    best = {candidate, at};
                    strength = score;
                }
                if (best.first == begin && strength == context_history_anchor_tokens) return best;
            }
            if (at - incoming == context_history_annotation_tokens || request.atom(at).image) break;
        }
        return best;
    }

private:
    struct Key {
        Fingerprint hash;
        size_t count;
        bool operator==(const Key& other) const { return count == other.count && hash == other.hash; }
    };
    struct KeyHash {
        size_t operator()(const Key& key) const {
            return static_cast<size_t>(key.hash.first ^ (key.hash.second << 1) ^ key.count);
        }
    };
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
    Key anchor(size_t begin, size_t end) const {
        size_t next = begin;
        while (next < end && next - begin < context_history_anchor_tokens) {
            if (atoms_[next++].image) break;
        }
        return {range(begin, next), next - begin};
    }
    void erase_last(const Key& key) {
        auto found = positions_.find(key);
        // Tail maintenance visits source positions in descending order.
        found->second.pop_back();
        if (found->second.empty()) positions_.erase(found);
    }

    std::vector<context_history_detail::Atom> atoms_;
    std::vector<Fingerprint> hashes_;
    std::vector<Fingerprint> powers_;
    std::vector<size_t> text_counts_;
    std::vector<Key> keys_;
    std::unordered_map<Key, std::vector<size_t>, KeyHash> positions_;
    size_t size_ = 0;
    int image_token_ = 0;
    bool valid_ = false;
    bool index_anchors_ = true;
};

// Buffer addresses survive Model::seq_bind's vector swaps. Vector-object
// references do not. Views are valid until the chosen source is mutated; apply
// must therefore run before restore/prefill can reallocate its storage.
template <typename T>
class ContextHistorySourceView {
public:
    ContextHistorySourceView() = default;
    explicit ContextHistorySourceView(const std::vector<T>& source)
        : data_(source.empty() ? empty_data() : source.data()), size_(source.size()) {}
    const T* data() const { return data_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const T& operator[](size_t pos) const { return data_[pos]; }

private:
    static const T* empty_data() {
        static const T empty{};
        return &empty;
    }
    const T* data_ = empty_data();
    size_t size_ = 0;
};

class ContextHistoryPlan {
public:
    ContextHistoryPlan() = default;
    ContextHistoryPlan(HistoryProjection projection, const std::vector<int32_t>& prefix,
                       const std::vector<ImageSpan>& images, const ContextHistoryIndex& index)
        : projection_(std::move(projection)), prefix_(prefix), images_(images) {
        const size_t end = index.boundary(projection_.model_tokens());
        if (!projection_.valid() || !index.valid() || index.size() != prefix.size() ||
            end == ContextHistoryIndex::npos) projection_ = {};
        else fingerprint_ = index.range(0, end);
    }

    bool valid() const { return projection_.valid(); }
    size_t cached_tokens() const { return projection_.model_tokens(); }
    size_t retained_tokens() const { return projection_.retained_tokens(); }
    int64_t to_model(int64_t pos) const { return projection_.to_model(pos); }
    size_t to_request(size_t pos) const { return projection_.to_request(pos); }
    const HistoryProjection& projection() const { return projection_; }
    // Sources stay immutable through scheduler selection and materialization.
    // They can extend beyond the chosen checkpoint; cached_tokens() is its end.
    const ContextHistorySourceView<int32_t>& prefix() const { return prefix_; }
    const ContextHistorySourceView<ImageSpan>& images() const { return images_; }

    // The full causal prefix is fingerprinted, including hidden image identities
    // and widths, not merely its visible projection. Exact validation is deferred
    // until apply, so candidate selection never copies or scans source histories.
    bool same_history(const ContextHistoryPlan& other) const {
        return valid() && other.valid() && retained_tokens() == other.retained_tokens() &&
            cached_tokens() == other.cached_tokens() && fingerprint_ == other.fingerprint_;
    }

    // Called only after exact selected-source validation. Source views remain
    // unchanged; the visible/model boundary and its fingerprint shrink together.
    bool shrink_to(HistoryProjection projection, const ContextHistoryIndex& index) {
        if (!projection.valid() || projection.model_tokens() > cached_tokens() ||
            projection.retained_tokens() > retained_tokens() ||
            projection.request_tokens() != projection_.request_tokens()) return false;
        const size_t end = index.boundary(projection.model_tokens());
        if (end == ContextHistoryIndex::npos) return false;
        fingerprint_ = index.prefix_fingerprint(end);
        projection_ = std::move(projection);
        return true;
    }

private:
    HistoryProjection projection_;
    ContextHistorySourceView<int32_t> prefix_;
    ContextHistorySourceView<ImageSpan> images_;
    ContextHistoryIndex::Fingerprint fingerprint_;
};

namespace context_history_detail {

inline void append_edit(std::vector<HistoryEdit>& edits, size_t begin, size_t end,
                        size_t incoming, size_t incoming_end) {
    if (!edits.empty() && edits.back().model_begin + edits.back().model_length == begin &&
        edits.back().request_begin + edits.back().request_length == incoming) {
        edits.back().model_length += end - begin;
        edits.back().request_length += incoming_end - incoming;
    } else {
        edits.push_back({begin, end - begin, incoming, incoming_end - incoming});
    }
}

}  // namespace context_history_detail

// Project a client-visible history onto a complete restorable causal prefix.
// Short replacement/annotation text between retained chunks is a presentation
// edit, not a new instruction to evaluate in the past. Newly appended content
// is always evaluated normally. No KV is spliced or rebased.
inline ContextHistoryPlan reconcile_context_history(const std::vector<int32_t>& cached,
                                                    const std::vector<ImageSpan>& cached_images,
                                                    const ContextHistoryIndex& cached_index,
                                                    const std::vector<int>& request,
                                                    const std::vector<ImageSpan>& request_images,
                                                    const ContextHistoryIndex& request_index,
                                                    int image_token, int64_t max_tokens,
                                                    size_t cached_limit = std::numeric_limits<size_t>::max()) {
    if (cached_limit == std::numeric_limits<size_t>::max()) cached_limit = cached.size();
    if (max_tokens <= 0 ||
        cached_limit > cached.size() || cached_limit > static_cast<uint64_t>(max_tokens) ||
        request.size() > static_cast<uint64_t>(max_tokens) ||
        !cached_index.valid_for(image_token) || !request_index.valid_for(image_token) ||
        cached_index.size() != cached.size() || request_index.size() != request.size()) return {};
    // Metadata and placeholders were checked while building/extending the two
    // indices. The supplied sources must still describe those physical buffers.
    (void)request_images;
    const size_t limit = cached_index.floor_boundary(cached_limit);
    cached_limit = cached_index.token_position(limit);
    using context_history_detail::append_edit;
    std::vector<HistoryEdit> edits;
    size_t old = 0, incoming = 0;
    while (old < limit) {
        if (incoming >= request_index.atom_count()) return {};
        const size_t equal = cached_index.common_atoms(old, limit, request_index, incoming);
        if (equal != 0) {
            old += equal;
            incoming += equal;
            continue;
        }
        const auto& previous = cached_index.atom(old);
        const auto& next = request_index.atom(incoming);
        const auto resume = cached_index.find_anchor(old, limit - 1, limit, request_index, incoming);
        if (resume.first == ContextHistoryIndex::npos) {
            // With no later retained anchor, the old tail is hidden uniformly:
            // text, tool results and images are all historical model fragments.
            // Require some retained evidence rather than adopting an unrelated
            // source whose entire visible history has disappeared.
            if (incoming == 0) return {};
            append_edit(edits, previous.start, cached_limit, next.start, next.start);
            old = limit;
            continue;
        }
        if (resume.first == old && resume.second == incoming) return {};
        append_edit(edits, previous.start, cached_index.token_position(resume.first),
                    next.start, request_index.token_position(resume.second));
        old = resume.first;
        incoming = resume.second;
    }
    if (edits.empty() || incoming >= request_index.atom_count()) return {};
    const size_t retained = request_index.token_position(incoming);
    if (request.size() - retained > static_cast<uint64_t>(max_tokens) - cached_limit) return {};
    auto projection = HistoryProjection::make(std::move(edits), cached_limit, request.size(), retained);
    if (!projection.valid()) return {};
    return {std::move(projection), cached, cached_images, cached_index};
}

// Materialize only after choosing a cache source. Hidden text and image metadata
// come from that exact evaluated prefix; omitted pixels are never synthesized or
// evaluated. Callers must restore and verify its KV before committing this prompt.
using ContextHistoryRestorable = std::function<size_t(size_t)>;

inline bool apply_context_history(ContextHistoryPlan& plan, const std::vector<int>& request,
                                  const std::vector<ImageSpan>& request_images, int image_token,
                                  std::vector<int>& expanded, std::vector<ImageSpan>& expanded_images,
                                  size_t* verified_prefix = nullptr,
                                  const ContextHistoryRestorable& restorable = ContextHistoryRestorable()) {
    if (verified_prefix) *verified_prefix = 0;
    if (!plan.valid() || plan.projection().request_tokens() != request.size() ||
        plan.cached_tokens() > plan.prefix().size()) return false;
    std::vector<context_history_detail::Atom> checked;
    if (!context_history_detail::atoms(request, request_images, image_token, request.size(), checked)) return false;

    // Only the selected source is copied and normalized exactly. Candidate
    // matching never performs these token/image scans or allocations.
    std::vector<int> tokens(plan.prefix().begin(), plan.prefix().begin() + plan.cached_tokens());
    const auto image_end = std::lower_bound(plan.images().begin(), plan.images().end(), plan.cached_tokens(),
        [](const ImageSpan& span, size_t pos) { return span.start < static_cast<int64_t>(pos); });
    std::vector<ImageSpan> images(plan.images().begin(), image_end);
    ContextHistoryIndex source;
    source.assign(tokens, images, image_token, false);
    if (!source.valid()) return false;

    const auto request_boundary = [&](size_t pos) {
        if (pos == request.size()) return checked.size();
        const auto found = std::lower_bound(checked.begin(), checked.end(), pos,
            [](const context_history_detail::Atom& atom, size_t at) { return atom.start < at; });
        return found != checked.end() && found->start == pos
            ? static_cast<size_t>(found - checked.begin()) : ContextHistoryIndex::npos;
    };
    size_t model = 0, incoming = 0, old_atom = 0, new_atom = 0;
    std::vector<HistoryEdit> verified_edits;
    // Token identity, image identity and complete image width are checked here,
    // not just hashes or equal image-placeholder tokens. Never step through a
    // partial image when a presentation/checkpoint boundary has moved.
    const auto equal_until = [&](size_t model_end, size_t request_end) {
        if (model_end < model || request_end < incoming) return false;
        while (model < model_end && incoming < request_end) {
            if (old_atom >= source.atom_count() || new_atom >= checked.size()) return false;
            const auto& previous = source.atom_at(old_atom);
            const auto& next = checked[new_atom];
            if (previous.start != model || next.start != incoming ||
                previous.width > model_end - model || next.width > request_end - incoming ||
                !(previous == next)) return false;
            model += previous.width;
            incoming += next.width;
            ++old_atom;
            ++new_atom;
            if (verified_prefix && verified_edits.empty()) *verified_prefix = incoming;
        }
        return model == model_end && incoming == request_end;
    };
    bool exact = true;
    for (const auto& edit : plan.projection().edits()) {
        if (!equal_until(edit.model_begin, edit.request_begin)) {
            exact = false;
            break;
        }
        const size_t model_end = edit.model_begin + edit.model_length;
        const size_t request_end = edit.request_begin + edit.request_length;
        const size_t old_end = source.boundary(model_end);
        const size_t new_end = request_boundary(request_end);
        if (old_end == ContextHistoryIndex::npos || new_end == ContextHistoryIndex::npos) {
            exact = false;
            break;
        }
        // Incoming images are never annotations, even if their placeholders
        // happen to equal an omitted source image's placeholders.
        for (size_t at = new_atom; at < new_end; ++at) {
            if (checked[at].image) {
                exact = false;
                break;
            }
        }
        if (!exact) break;
        verified_edits.push_back(edit);
        old_atom = old_end;
        new_atom = new_end;
        model = model_end;
        incoming = request_end;
    }
    if (exact) exact = equal_until(plan.cached_tokens(), plan.retained_tokens());
    // Keep every proven atom, then align with a physical checkpoint even when
    // content is exact but its presentation boundary has moved. Rounding through
    // an image can move off a checkpoint, so ask again until both agree.
    size_t limit = model;
    if (restorable) {
        while (limit != 0) {
            const size_t candidate = std::min(limit, restorable(limit));
            const size_t rounded = source.token_position(source.floor_boundary(candidate));
            if (rounded == limit) break;
            limit = rounded;
        }
    }
    if (!exact || limit < plan.cached_tokens()) {
        if (limit == 0) return false;
        std::vector<HistoryEdit> edits;
        size_t model_cursor = 0, request_cursor = 0;
        for (const auto& edit : verified_edits) {
            if (edit.model_begin > limit) break;
            const size_t end = edit.model_begin + edit.model_length;
            if (end <= limit) {
                edits.push_back(edit);
                model_cursor = end;
                request_cursor = edit.request_begin + edit.request_length;
            } else {
                if (edit.model_begin < limit) {
                    edits.push_back({edit.model_begin, limit - edit.model_begin, edit.request_begin, 0});
                    model_cursor = limit;
                    request_cursor = edit.request_begin;
                }
                break;
            }
        }
        // Before the first fold there is no historical projection to retain.
        // Report its exact ordinary prefix instead of discarding all reuse.
        if (edits.empty()) return false;
        const size_t retained = request_cursor + limit - model_cursor;
        auto projection = HistoryProjection::make(std::move(edits), limit, request.size(), retained);
        if (!plan.shrink_to(std::move(projection), source)) return false;
        tokens.resize(limit);
        images.erase(std::lower_bound(images.begin(), images.end(), limit,
            [](const ImageSpan& span, size_t pos) { return span.start < static_cast<int64_t>(pos); }), images.end());
    }

    if (plan.projection().expanded_tokens() > tokens.max_size()) return false;
    tokens.reserve(plan.projection().expanded_tokens());
    tokens.insert(tokens.end(), request.begin() + plan.retained_tokens(), request.end());
    for (const auto& image : request_images) {
        if (static_cast<size_t>(image.start) >= plan.retained_tokens())
            images.push_back({plan.to_model(image.start), image.len, image.hash});
    }
    if (!image_spans_valid(images, static_cast<int64_t>(tokens.size()))) return false;
    expanded.swap(tokens);
    expanded_images.swap(images);
    return true;
}

}  // namespace strata::ds4
