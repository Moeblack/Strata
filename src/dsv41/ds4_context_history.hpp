#pragma once

#include "ds4_history_projection.hpp"
#include "ds4_images.hpp"

#include <array>
#include <limits>
#include <unordered_map>
#include <utility>

namespace strata::ds4 {

// Retention is a session policy, not an assertion that deleting causal inputs
// leaves their successors' KV unchanged. Strict callers must select disabled.
enum class ContextHistoryPolicy { disabled, images, fragments };
inline constexpr size_t context_history_anchor_tokens = 8;
inline constexpr size_t context_history_annotation_tokens = 256;

class ContextHistoryPlan {
public:
    ContextHistoryPlan() = default;
    ContextHistoryPlan(HistoryProjection projection, std::vector<int32_t> prefix,
                       std::vector<ImageSpan> images)
        : projection_(std::move(projection)), prefix_(std::move(prefix)), images_(std::move(images)) {
        if (!projection_.valid() || prefix_.size() != projection_.model_tokens() ||
            !image_spans_valid(images_, static_cast<int64_t>(prefix_.size()))) projection_ = {};
    }

    bool valid() const { return projection_.valid(); }
    size_t cached_tokens() const { return projection_.model_tokens(); }
    size_t retained_tokens() const { return projection_.retained_tokens(); }
    int64_t to_model(int64_t pos) const { return projection_.to_model(pos); }
    size_t to_request(size_t pos) const { return projection_.to_request(pos); }
    const HistoryProjection& projection() const { return projection_; }
    const std::vector<int32_t>& prefix() const { return prefix_; }
    const std::vector<ImageSpan>& images() const { return images_; }

    // Compare the actual causal history, not just omitted image identities or a
    // hash. Equal visible projections of different histories are not a session ID.
    bool same_history(const ContextHistoryPlan& other) const {
        if (!valid() || !other.valid() || retained_tokens() != other.retained_tokens() ||
            prefix_ != other.prefix_ || images_.size() != other.images_.size()) return false;
        for (size_t i = 0; i < images_.size(); ++i) {
            const auto& a = images_[i];
            const auto& b = other.images_[i];
            if (a.start != b.start || a.len != b.len || a.hash != b.hash) return false;
        }
        return true;
    }

private:
    HistoryProjection projection_;
    std::vector<int32_t> prefix_;
    std::vector<ImageSpan> images_;
};

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

// Index short exact chunks, with image identities as independent anchors. Short
// complete suffixes also qualify: there is no fixed trailing-text requirement.
// Hashes locate candidates only; every candidate is compared atom by atom.
class AnchorIndex {
public:
    explicit AnchorIndex(const std::vector<Atom>& cached) : cached_(cached) {}

    std::pair<size_t, size_t> find(size_t begin, size_t end, const std::vector<Atom>& request,
                                 size_t incoming) {
        if (!built_) {
            for (size_t i = 0; i < cached_.size(); ++i) {
                const auto keys = anchors(cached_, i);
                positions_[keys.values[keys.count - 1]].push_back(i);
            }
            built_ = true;
        }
        // Prefer strong chunks and the earliest reusable model position, not a
        // coincidental short suffix that would turn a long continuation into a
        // fresh read. Only short text annotations may be bypassed, never images.
        std::pair<size_t, size_t> best{cached_.size(), request.size()};
        size_t strength = 0;
        for (size_t at = incoming; at < request.size(); ++at) {
            const auto keys = anchors(request, at);
            for (size_t count = keys.count; count != 0; --count) {
                const auto found = positions_.find(keys.values[count - 1]);
                if (found == positions_.end()) continue;
                const auto& positions = found->second;
                auto pos = std::lower_bound(positions.begin(), positions.end(), begin);
                for (; pos != positions.end() && *pos <= end; ++pos) {
                    if (anchors(cached_, *pos).count != count) continue;
                    if (!std::equal(request.begin() + at, request.begin() + at + count,
                                    cached_.begin() + *pos)) continue;
                    const size_t score = cached_[*pos + count - 1].image ? context_history_anchor_tokens : count;
                    if (score > strength || (score == strength && *pos < best.first)) {
                        best = {*pos, at};
                        strength = score;
                    }
                    if (best.first == begin && strength == context_history_anchor_tokens) return best;
                    break;
                }
            }
            if (at - incoming == context_history_annotation_tokens || request[at].image) break;
        }
        return best;
    }

private:
    struct Keys {
        std::array<uint64_t, context_history_anchor_tokens> values{};
        size_t count = 0;
    };
    static Keys anchors(const std::vector<Atom>& atoms, size_t begin) {
        Keys keys;
        uint64_t key = 14695981039346656037ULL;
        for (size_t i = begin; i < atoms.size() && i - begin < context_history_anchor_tokens; ++i) {
            const auto& atom = atoms[i];
            key = (key ^ static_cast<uint64_t>(atom.image)) * 1099511628211ULL;
            key = (key ^ atom.identity) * 1099511628211ULL;
            key = (key ^ atom.width) * 1099511628211ULL;
            keys.values[keys.count++] = key;
            if (atom.image) break;
        }
        return keys;
    }

    const std::vector<Atom>& cached_;
    std::unordered_map<uint64_t, std::vector<size_t>> positions_;
    bool built_ = false;
};

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
// is always evaluated normally. Strict matching remains available for edits
// that must change historical conditioning. No KV is spliced or rebased.
inline ContextHistoryPlan reconcile_context_history(const std::vector<int32_t>& cached,
                                                    const std::vector<ImageSpan>& cached_images,
                                                    const std::vector<int>& request,
                                                    const std::vector<ImageSpan>& request_images,
                                                    int image_token, int64_t max_tokens,
                                                    ContextHistoryPolicy policy,
                                                    size_t cached_limit = std::numeric_limits<size_t>::max()) {
    if (cached_limit == std::numeric_limits<size_t>::max()) cached_limit = cached.size();
    if (policy == ContextHistoryPolicy::disabled || max_tokens <= 0 ||
        cached_limit > cached.size() || cached_limit > static_cast<uint64_t>(max_tokens) ||
        request.size() > static_cast<uint64_t>(max_tokens)) return {};
    using namespace context_history_detail;
    std::vector<Atom> old_atoms, new_atoms;
    if (!atoms(cached, cached_images, image_token, cached_limit, old_atoms) ||
        !atoms(request, request_images, image_token, request.size(), new_atoms)) return {};

    std::vector<HistoryEdit> edits;
    AnchorIndex anchors(old_atoms);
    size_t old = 0, incoming = 0;
    while (old < old_atoms.size()) {
        if (incoming >= new_atoms.size()) return {};
        const Atom& previous = old_atoms[old];
        const Atom& next = new_atoms[incoming];
        if (previous == next) {
            ++old;
            ++incoming;
            continue;
        }
        size_t end = old_atoms.size() - 1;
        if (policy == ContextHistoryPolicy::images) {
            // Image retention also tolerates short changed captions/markers.
            // Larger text folds are an explicit fragment-retention policy.
            size_t text = 0;
            end = old;
            while (end < old_atoms.size() - 1 && text < context_history_annotation_tokens) {
                if (!old_atoms[end].image) ++text;
                ++end;
            }
        }
        const auto resume = anchors.find(old, end, new_atoms, incoming);
        if (resume.first == old_atoms.size()) {
            if (!previous.image) return {};
            // A trailing omitted picture needs no trailing text anchor. Try
            // resynchronization first so annotations before retained pictures
            // cannot accidentally discard those pictures.
            append_edit(edits, previous.start, previous.start + previous.width, next.start, next.start);
            ++old;
            continue;
        }
        append_edit(edits, previous.start, old_atoms[resume.first].start,
                    next.start, new_atoms[resume.second].start);
        old = resume.first;
        incoming = resume.second;
    }
    if (edits.empty() || incoming >= new_atoms.size()) return {};
    const size_t retained = new_atoms[incoming].start;
    if (request.size() - retained > static_cast<uint64_t>(max_tokens) - cached_limit) return {};
    auto projection = HistoryProjection::make(std::move(edits), cached_limit, request.size(), retained);
    if (!projection.valid()) return {};
    std::vector<ImageSpan> images;
    for (const auto& image : cached_images) {
        if (static_cast<uint64_t>(image.start) >= cached_limit) break;
        images.push_back(image);
    }
    return {std::move(projection), std::vector<int32_t>(cached.begin(), cached.begin() + cached_limit),
            std::move(images)};
}

// Materialize only after choosing a cache source. Hidden text and image metadata
// come from that exact evaluated prefix; omitted pixels are never synthesized or
// evaluated. Callers must restore and verify its KV before committing this prompt.
inline bool apply_context_history(const ContextHistoryPlan& plan, const std::vector<int>& request,
                                  const std::vector<ImageSpan>& request_images, int image_token,
                                  std::vector<int>& expanded, std::vector<ImageSpan>& expanded_images) {
    if (!plan.valid() || plan.projection().request_tokens() != request.size()) return false;
    std::vector<context_history_detail::Atom> checked;
    if (!context_history_detail::atoms(request, request_images, image_token, request.size(), checked)) return false;

    // Validate retained intervals against the immutable source. The request-side
    // text inside edits is deliberately treated as a historical annotation.
    size_t model = 0, incoming = 0;
    for (const auto& edit : plan.projection().edits()) {
        if (!std::equal(plan.prefix().begin() + model, plan.prefix().begin() + edit.model_begin,
                        request.begin() + incoming)) return false;
        incoming = edit.request_begin + edit.request_length;
        model = edit.model_begin + edit.model_length;
    }
    if (!std::equal(plan.prefix().begin() + model, plan.prefix().end(), request.begin() + incoming)) return false;
    size_t image = 0, edit_index = 0;
    for (const auto& s : request_images) {
        if (static_cast<size_t>(s.start) >= plan.retained_tokens()) break;
        if (static_cast<size_t>(s.start + s.len) > plan.retained_tokens()) return false;
        const auto& edits = plan.projection().edits();
        while (edit_index < edits.size() &&
               edits[edit_index].request_begin + edits[edit_index].request_length <= static_cast<size_t>(s.start))
            ++edit_index;
        if (edit_index < edits.size() && edits[edit_index].request_begin < static_cast<size_t>(s.start + s.len))
            return false;  // An incoming image must never be discarded as annotation text.
        const int64_t at = plan.to_model(s.start);
        while (image < plan.images().size() && plan.images()[image].start < at) ++image;
        if (image == plan.images().size()) return false;
        const auto& previous = plan.images()[image];
        if (previous.start != at || previous.len != s.len || previous.hash != s.hash) return false;
        // An edit cannot split a retained image even if it contains equal placeholders.
        if (plan.to_model(s.start + s.len - 1) != at + s.len - 1) return false;
    }

    std::vector<int> tokens;
    std::vector<ImageSpan> images = plan.images();
    if (plan.projection().expanded_tokens() > tokens.max_size()) return false;
    tokens.reserve(plan.projection().expanded_tokens());
    tokens.insert(tokens.end(), plan.prefix().begin(), plan.prefix().end());
    tokens.insert(tokens.end(), request.begin() + plan.retained_tokens(), request.end());
    for (const auto& s : request_images) {
        if (static_cast<size_t>(s.start) >= plan.retained_tokens())
            images.push_back({plan.to_model(s.start), s.len, s.hash});
    }
    if (!image_spans_valid(images, static_cast<int64_t>(tokens.size()))) return false;
    expanded.swap(tokens);
    expanded_images.swap(images);
    return true;
}

}  // namespace strata::ds4
