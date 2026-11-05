#pragma once

#include "ds4_turns.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::ds4 {

enum class PlanKind { Whole, Rewind, Fresh, Overflow };

struct CachePlan {
    PlanKind kind = PlanKind::Overflow;
    std::size_t keep = 0;          // q: retained model positions
    std::size_t request_from = 0;  // q': request boundary corresponding to q
    std::size_t request_size = 0;

    bool fits() const { return kind != PlanKind::Overflow; }
    bool reuses() const { return kind == PlanKind::Whole || kind == PlanKind::Rewind; }
    // Prefill work, used unchanged by the concurrent sequence selector. Equal
    // costs prefer Whole, then the checkpoint that retains fewer positions.
    std::size_t cost() const {
        return fits() ? request_size - request_from : std::numeric_limits<std::size_t>::max();
    }
    int64_t to_model(int64_t position) const {
        if (position < 0) return -1;
        if (!reuses()) return position;
        if (static_cast<uint64_t>(position) < request_from) return -1;
        return position - static_cast<int64_t>(request_from) + static_cast<int64_t>(keep);
    }
    std::size_t to_request(std::size_t position) const {
        if (!reuses()) return position;
        return position < keep ? std::min(position, request_from) : position - keep + request_from;
    }
};

// Subtractions prove q + (|R| - q') + max_new <= min(C_mem, B_pos)
// without overflowing even for malformed wire budgets. B_pos is independent of
// the allocated KV capacity; a large memory reservation cannot extend RoPE.
inline bool cache_fits(std::size_t q, std::size_t request_size, std::size_t request_from,
                       int64_t max_new, int64_t memory_capacity, int64_t position_budget) {
    const int64_t limit = std::min(memory_capacity, position_budget);
    if (limit < 0 || max_new < 0 || max_new > limit || request_from > request_size) return false;
    const auto room = static_cast<uint64_t>(limit - max_new);
    return q <= room && request_size - request_from <= room - q;
}

inline bool image_boundary(std::size_t position, const std::vector<ImageSpan>& images) {
    const auto it = std::upper_bound(images.begin(), images.end(), position,
        [](std::size_t p, const ImageSpan& image) { return p < static_cast<std::size_t>(image.start); });
    if (it == images.begin()) return true;
    const auto& image = *std::prev(it);
    return position == static_cast<std::size_t>(image.start) ||
           position - static_cast<std::size_t>(image.start) >= static_cast<std::size_t>(image.len);
}

inline CachePlan solve_plan(const TurnAlignment& alignment, std::size_t state_size,
                             std::size_t request_size, const std::vector<int64_t>& checkpoints,
                             int64_t max_new, int64_t memory_capacity, int64_t position_budget,
                             const std::vector<ImageSpan>& state_images = {},
                             const std::vector<ImageSpan>& request_images = {}) {
    CachePlan best{PlanKind::Overflow, 0, 0, request_size};
    if (cache_fits(0, request_size, 0, max_new, memory_capacity, position_budget))
        best.kind = PlanKind::Fresh;
    const auto consider = [&](PlanKind kind, std::size_t q) {
        const auto rp = alignment.to_request(q);
        // At least one request token is read to produce this prompt's logits.
        if (q == 0 || rp == TurnAlignment::npos || rp >= request_size ||
            !image_boundary(q, state_images) || !image_boundary(rp, request_images) ||
            !cache_fits(q, request_size, rp, max_new, memory_capacity, position_budget)) return;
        CachePlan candidate{kind, q, rp, request_size};
        if (!best.fits() || candidate.cost() < best.cost() ||
            (candidate.cost() == best.cost() &&
             (kind == PlanKind::Whole || (best.kind != PlanKind::Whole && q < best.keep)))) best = candidate;
    };
    if (alignment.state_end == state_size) consider(PlanKind::Whole, state_size);
    // Enumerate every restorable checkpoint. The largest q need not fit: old
    // elided images/reasoning still occupy real positions in that checkpoint.
    for (int64_t q : checkpoints)
        if (q > 0 && static_cast<uint64_t>(q) <= state_size)
            consider(PlanKind::Rewind, static_cast<std::size_t>(q));
    return best;
}

template <typename StateToken, typename RequestToken>
inline CachePlan plan_cache(const std::vector<StateToken>& state, const std::vector<RequestToken>& request,
                             const std::vector<int64_t>& checkpoints, int64_t max_new,
                             int64_t memory_capacity, int64_t position_budget,
                             const std::vector<ImageSpan>& state_images = {},
                             const std::vector<ImageSpan>& request_images = {},
                             const TurnTokens& ids = {}) {
    const auto alignment = align_turns(state, request, state_images, request_images, ids);
    return solve_plan(alignment, state.size(), request.size(), checkpoints, max_new,
                      memory_capacity, position_budget, state_images, request_images);
}

}  // namespace strata::ds4
