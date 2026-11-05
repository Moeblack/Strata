#pragma once

#include "ds4_images.hpp"

#include <limits>

namespace strata::ds4 {

// Text after the final omission must independently anchor the reused history.
inline constexpr size_t image_history_min_text_tokens = 32;

struct ImageHistoryPlan {
    std::vector<ImageSpan> omitted;
    size_t cached_tokens = 0;    // Complete reusable model prefix (possibly a checkpoint).
    size_t request_tokens = 0;   // Full incoming prompt length.
    size_t retained_tokens = 0;  // Matched incoming prefix, excluding omitted pictures.

    bool valid() const {
        const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        if (omitted.empty() || cached_tokens > limit || request_tokens > limit ||
            retained_tokens >= request_tokens || retained_tokens > cached_tokens ||
            !image_spans_valid(omitted, static_cast<int64_t>(cached_tokens))) return false;
        size_t removed = 0;
        for (const ImageSpan& s : omitted) {
            const size_t len = static_cast<size_t>(s.len);
            if (len > cached_tokens - removed) return false;
            removed += len;
        }
        return cached_tokens - removed == retained_tokens && removed <= limit - request_tokens &&
            removed <= std::numeric_limits<size_t>::max() - request_tokens;
    }

    // Boundaries are right-biased: omitted pictures precede the token at a boundary.
    int64_t to_model(int64_t request_pos) const {
        if (!valid() || request_pos < 0 || static_cast<uint64_t>(request_pos) > request_tokens) return -1;
        int64_t model_pos = request_pos;
        for (const ImageSpan& s : omitted) {
            if (s.start > model_pos) break;
            model_pos += s.len;
        }
        return model_pos;
    }

    // Positions inside omitted pictures clamp to their request boundary.
    size_t to_request(size_t model_pos) const {
        if (!valid() || model_pos > request_tokens + (cached_tokens - retained_tokens))
            return std::numeric_limits<size_t>::max();
        size_t removed = 0;
        for (const ImageSpan& s : omitted) {
            const size_t start = static_cast<size_t>(s.start);
            if (model_pos <= start) break;
            removed += std::min(model_pos - start, static_cast<size_t>(s.len));
            if (model_pos < start + static_cast<size_t>(s.len)) break;
        }
        return model_pos - removed;
    }
};

namespace image_history_detail {

// A placeholder without an identity (or an identity covering ordinary text) is
// not safe to reuse, even if the integer token prefix happens to match.
template <typename Token>
inline bool tokens_valid(const std::vector<Token>& tokens, const std::vector<ImageSpan>& images,
                         int image_token) {
    if (tokens.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !image_spans_valid(images, static_cast<int64_t>(tokens.size()))) return false;
    size_t image = 0;
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        while (image < images.size() &&
               pos >= static_cast<size_t>(images[image].start + images[image].len)) ++image;
        const bool inside = image < images.size() && pos >= static_cast<size_t>(images[image].start);
        if ((tokens[pos] == image_token) != inside) return false;
    }
    return true;
}

}  // namespace image_history_detail

inline ImageHistoryPlan reconcile_image_history(const std::vector<int32_t>& cached,
                                                const std::vector<ImageSpan>& cached_images,
                                                const std::vector<int>& request,
                                                const std::vector<ImageSpan>& request_images,
                                                int image_token, int64_t max_tokens,
                                                size_t cached_limit = std::numeric_limits<size_t>::max()) {
    if (cached_limit == std::numeric_limits<size_t>::max()) cached_limit = cached.size();
    if (cached_limit > cached.size() || max_tokens <= 0 ||
        request.size() > static_cast<uint64_t>(max_tokens) ||
        cached_limit > static_cast<uint64_t>(max_tokens) ||
        !image_history_detail::tokens_valid(cached, cached_images, image_token) ||
        !image_history_detail::tokens_valid(request, request_images, image_token)) return {};

    // Later cached pictures belong to an unused continuation, but a checkpoint
    // may never cut an image in half. Validate the original metadata above.
    for (const ImageSpan& s : cached_images) {
        if (static_cast<uint64_t>(s.start) >= cached_limit) break;
        if (static_cast<uint64_t>(s.len) > cached_limit - static_cast<size_t>(s.start)) return {};
    }
    ImageHistoryPlan plan;
    size_t model = 0, incoming = 0, old_image = 0, new_image = 0;
    size_t removed = 0, text_after_omission = 0;
    bool retained_image = false;
    while (model < cached_limit) {
        const bool old_here = old_image < cached_images.size() &&
            static_cast<size_t>(cached_images[old_image].start) == model;
        const bool new_here = new_image < request_images.size() &&
            static_cast<size_t>(request_images[new_image].start) == incoming;
        if (old_here) {
            const ImageSpan& old = cached_images[old_image];
            if (new_here) {
                // A shorter incoming run can only retain the exact suffix of
                // this old run. Counting first also disambiguates repeated
                // identities, such as [A, A, A] becoming [A].
                size_t old_end = old_image + 1, new_end = new_image + 1;
                while (old_end < cached_images.size() &&
                       static_cast<size_t>(cached_images[old_end].start) < cached_limit &&
                       cached_images[old_end].start ==
                           cached_images[old_end - 1].start + cached_images[old_end - 1].len) ++old_end;
                while (new_end < request_images.size() &&
                       request_images[new_end].start ==
                           request_images[new_end - 1].start + request_images[new_end - 1].len) ++new_end;
                const size_t old_count = old_end - old_image;
                const size_t new_count = new_end - new_image;
                if (new_count > old_count) return {};
                const size_t skip = old_count - new_count;
                if (skip != 0 && retained_image) return {};
                for (size_t i = 0; i < new_count; ++i) {
                    const ImageSpan& previous = cached_images[old_image + skip + i];
                    const ImageSpan& next = request_images[new_image + i];
                    if (previous.len != next.len || previous.hash != next.hash) return {};
                }
                for (size_t i = 0; i < skip; ++i) {
                    const ImageSpan& missing = cached_images[old_image + i];
                    plan.omitted.push_back(missing);
                    removed += static_cast<size_t>(missing.len);
                }
                if (skip != 0) text_after_omission = 0;
                model = static_cast<size_t>(cached_images[old_end - 1].start + cached_images[old_end - 1].len);
                incoming = static_cast<size_t>(request_images[new_end - 1].start + request_images[new_end - 1].len);
                old_image = old_end;
                new_image = new_end;
                retained_image = true;
            } else {
                if (retained_image) return {};
                plan.omitted.push_back(old);
                const size_t len = static_cast<size_t>(old.len);
                model += len;
                removed += len;
                ++old_image;
                text_after_omission = 0;
            }
        } else {
            if (new_here || incoming >= request.size() || cached[model] != request[incoming]) return {};
            ++model;
            ++incoming;
            if (!plan.omitted.empty()) ++text_after_omission;
        }
    }
    if (plan.omitted.empty() || incoming >= request.size() ||
        text_after_omission < image_history_min_text_tokens ||
        removed > static_cast<uint64_t>(max_tokens) - request.size()) return {};
    plan.cached_tokens = cached_limit;
    plan.request_tokens = request.size();
    plan.retained_tokens = incoming;
    if (!plan.valid()) return {};
    return plan;
}

inline bool apply_image_history(const ImageHistoryPlan& plan, const std::vector<int>& request,
                                const std::vector<ImageSpan>& request_images, int image_token,
                                std::vector<int>& expanded, std::vector<ImageSpan>& expanded_images) {
    if (!plan.valid() || plan.request_tokens != request.size() ||
        !image_history_detail::tokens_valid(request, request_images, image_token)) return false;

    // Validate all insertion boundaries before allocating or modifying outputs.
    size_t removed = 0, image = 0;
    for (const ImageSpan& s : plan.omitted) {
        const size_t boundary = static_cast<size_t>(s.start) - removed;
        if (boundary > plan.retained_tokens) return false;
        while (image < request_images.size() &&
               static_cast<size_t>(request_images[image].start + request_images[image].len) <= boundary) ++image;
        // Omissions must precede every retained picture. They may meet the
        // first picture's start, but cannot split or follow a retained picture.
        if (image != 0 || (image < request_images.size() &&
            static_cast<size_t>(request_images[image].start) < boundary)) return false;
        removed += static_cast<size_t>(s.len);
    }
    for (const ImageSpan& s : request_images) {
        if (static_cast<size_t>(s.start) < plan.retained_tokens &&
            static_cast<size_t>(s.start + s.len) > plan.retained_tokens) return false;
    }

    const size_t total = request.size() + removed;
    std::vector<int> tokens;
    std::vector<ImageSpan> images;
    if (total > tokens.max_size() || plan.omitted.size() > images.max_size() - request_images.size()) return false;
    tokens.reserve(total);
    images.reserve(plan.omitted.size() + request_images.size());
    size_t source = 0;
    removed = 0;
    for (const ImageSpan& s : plan.omitted) {
        const size_t boundary = static_cast<size_t>(s.start) - removed;
        tokens.insert(tokens.end(), request.begin() + source, request.begin() + boundary);
        tokens.insert(tokens.end(), static_cast<size_t>(s.len), image_token);
        images.push_back(s);
        source = boundary;
        removed += static_cast<size_t>(s.len);
    }
    tokens.insert(tokens.end(), request.begin() + source, request.end());
    for (const ImageSpan& s : request_images) {
        // All insertions precede all retained pictures, so their shift is uniform.
        images.push_back({s.start + static_cast<int64_t>(removed), s.len, s.hash});
    }
    expanded.swap(tokens);
    expanded_images.swap(images);
    return true;
}

}  // namespace strata::ds4
