#pragma once

#include "ds4_images.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace strata::ds4 {

// DeepSeek-V4.1's reserved role tokens, never ordinary message text. Keeping the
// vocabulary at the boundary makes the topology independent of the GPU engine.
struct TurnTokens {
    int user = 128803;
    int assistant = 128804;
    int system = 128799;
    int think = 128821;
    int think_end = 128822;
    int image = -1;
};

enum class TurnRole { Preamble, User, Assistant, System };
enum class TurnState { Same, Elided, Novel };

struct Turn {
    std::size_t begin = 0, end = 0;
    TurnRole role = TurnRole::Preamble;
};

inline TurnRole turn_role(int64_t token, const TurnTokens& ids) {
    if (token == ids.user) return TurnRole::User;
    if (token == ids.assistant) return TurnRole::Assistant;
    if (token == ids.system) return TurnRole::System;
    return TurnRole::Preamble;
}

template <typename Token>
inline std::vector<Turn> split_turns(const std::vector<Token>& tokens, const TurnTokens& ids = {}) {
    std::vector<Turn> turns;
    std::size_t begin = 0;
    TurnRole role = TurnRole::Preamble;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto next = turn_role(static_cast<int64_t>(tokens[i]), ids);
        if (next == TurnRole::Preamble) continue;
        if (i > begin) turns.push_back({begin, i, role});
        begin = i;
        role = next;
    }
    if (begin < tokens.size()) turns.push_back({begin, tokens.size(), role});
    return turns;
}

struct TurnMatch {
    Turn state;
    std::size_t request_begin = 0, request_end = 0;
    TurnState kind = TurnState::Novel;
};

// Runs are monotone boundary maps, not suffix anchors. Same has a unit slope;
// Elided keeps cached-only tokens the request dropped or shortened in place
// (old pictures, reasoning, placeholders for pruned tool output): inside it the
// request position stays put, and its end maps to request_begin + request_len.
struct TurnRun {
    std::size_t state_begin = 0, state_end = 0, request_begin = 0, request_len = 0;
    TurnState kind = TurnState::Same;

    std::size_t request_end() const {
        return request_begin + (kind == TurnState::Same ? state_end - state_begin : request_len);
    }
};

struct TurnAlignment {
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    std::vector<TurnRun> runs;
    std::vector<TurnMatch> turns;
    std::size_t state_end = 0, request_end = 0;

    std::size_t to_request(std::size_t position) const {
        if (position > state_end) return npos;
        const auto it = std::upper_bound(runs.begin(), runs.end(), position,
            [](std::size_t p, const TurnRun& r) { return p < r.state_begin; });
        if (it == runs.begin()) return position == 0 ? 0 : npos;
        const auto& r = *std::prev(it);
        if (position > r.state_end) return npos;
        if (r.kind == TurnState::Same) return r.request_begin + (position - r.state_begin);
        return position == r.state_end ? r.request_end() : r.request_begin;
    }
};

// Alignment never skips a role. Inside a history turn (User/Assistant, not the
// state's last turn) a client may drop or shorten old content in place: omit a
// picture or a <think> block, swap a picture for a text placeholder, cut a tool
// output down to a stub. The cached original stays in the KV state and the
// request's shorter replacement is not read. The turn must keep its exact head
// and tail up to the next role token, and the replacement may carry no picture
// and no more tokens than what it replaces. Any other edit makes that turn and
// every later turn Novel; prefix checkpoints before it remain usable.
template <typename StateToken, typename RequestToken>
inline TurnAlignment align_turns(const std::vector<StateToken>& state,
                                  const std::vector<RequestToken>& request,
                                  const std::vector<ImageSpan>& state_images,
                                  const std::vector<ImageSpan>& request_images,
                                  const TurnTokens& ids = {}) {
    TurnAlignment out;
    if (!image_spans_valid(state_images, static_cast<int64_t>(state.size())) ||
        !image_spans_valid(request_images, static_cast<int64_t>(request.size()))) return out;
    const auto topology = split_turns(state, ids);
    std::size_t s = 0, r = 0, si = 0, ri = 0;
    TurnRole role = TurnRole::Preamble;
    const auto append = [&](std::size_t end, TurnState kind, std::size_t request_len = 0) {
        if (!out.runs.empty()) {
            auto& prev = out.runs.back();
            if (prev.kind == kind && prev.state_end == s && prev.request_end() == r) {
                prev.state_end = end;
                prev.request_len += request_len;
                return;
            }
        }
        out.runs.push_back({s, end, r, request_len, kind});
    };
    const auto eq = [&](std::size_t i, std::size_t j) {
        return static_cast<int64_t>(state[i]) == static_cast<int64_t>(request[j]);
    };
    const auto next_role = [&](const auto& v, std::size_t i) {
        while (i < v.size() && turn_role(static_cast<int64_t>(v[i]), ids) == TurnRole::Preamble) ++i;
        return i;
    };
    // At a mismatch inside turn `role`: keep the cached rest of the turn's middle,
    // skip the request's shorter replacement, resume at the common turn tail.
    const auto shrink = [&]() -> bool {
        if (role != TurnRole::User && role != TurnRole::Assistant) return false;
        const auto se = next_role(state, s);
        const auto re = next_role(request, r);
        if (se >= state.size() || re >= request.size() || !eq(se, re)) return false;
        auto sm = se, rm = re;
        while (sm > s && rm > r && eq(sm - 1, rm - 1)) --sm, --rm;
        // the tail must not start inside a cached picture: give the whole picture to the middle
        for (const auto& img : state_images) {
            const auto a = static_cast<std::size_t>(img.start), b = a + static_cast<std::size_t>(img.len);
            if (a < sm && sm < b) {
                rm += b - sm;
                sm = b;
            }
        }
        if (sm > se || rm > re || rm - r > sm - s || sm == s) return false;
        for (const auto& img : request_images) {
            const auto a = static_cast<std::size_t>(img.start), b = a + static_cast<std::size_t>(img.len);
            if (a < rm && b > r) return false;   // a picture in the replacement is new: read it
        }
        append(sm, TurnState::Elided, rm - r);
        s = sm;
        r = rm;
        return true;
    };
    while (s < state.size()) {
        while (si < state_images.size() && static_cast<std::size_t>(state_images[si].start) < s) ++si;
        while (ri < request_images.size() && static_cast<std::size_t>(request_images[ri].start) < r) ++ri;
        const auto* a = si < state_images.size() && static_cast<std::size_t>(state_images[si].start) == s
                            ? &state_images[si] : nullptr;
        const auto* b = ri < request_images.size() && static_cast<std::size_t>(request_images[ri].start) == r
                            ? &request_images[ri] : nullptr;
        if (a) {
            const auto n = static_cast<std::size_t>(a->len);
            if (b && a->len == b->len && a->hash == b->hash &&
                std::equal(state.begin() + s, state.begin() + s + n, request.begin() + r,
                    [](StateToken x, RequestToken y) { return static_cast<int64_t>(x) == static_cast<int64_t>(y); })) {
                append(s + n, TurnState::Same);
                s += n;
                r += n;
                continue;
            }
            // A different retained picture is an edit, not an omission. Only a
            // picture absent from the request can be left in the cached history.
            if (!b && (role == TurnRole::User || role == TurnRole::Assistant)) {
                append(s + n, TurnState::Elided);
                s += n;
                continue;
            }
            if (shrink()) continue;
            break;
        }
        if (b || (ids.image >= 0 && static_cast<int64_t>(state[s]) == ids.image)) {
            if (shrink()) continue;
            break;
        }
        if (role == TurnRole::Assistant && static_cast<int64_t>(state[s]) == ids.think &&
            r < request.size() && static_cast<int64_t>(request[r]) == ids.think_end) {
            std::size_t end = s + 1;
            while (end < state.size() && static_cast<int64_t>(state[end]) != ids.think_end &&
                   turn_role(static_cast<int64_t>(state[end]), ids) == TurnRole::Preamble) ++end;
            if (end == state.size() || static_cast<int64_t>(state[end]) != ids.think_end) break;
            append(end, TurnState::Elided); // retain the closing token in the exact run
            s = end;
            continue;
        }
        if (r == request.size() || !eq(s, r)) {
            if (shrink()) continue;
            break;
        }
        const auto next = turn_role(static_cast<int64_t>(state[s]), ids);
        if (next != TurnRole::Preamble) role = next;
        append(s + 1, TurnState::Same);
        ++s;
        ++r;
    }
    out.state_end = s;
    out.request_end = r;
    std::size_t first_run = 0;
    for (const auto& t : topology) {
        while (first_run < out.runs.size() && out.runs[first_run].state_end <= t.begin) ++first_run;
        TurnMatch match{t, 0, 0, TurnState::Novel};
        const auto begin = out.to_request(t.begin);
        const auto end = out.to_request(t.end);
        if (begin != TurnAlignment::npos) match.request_begin = begin;
        if (end != TurnAlignment::npos) match.request_end = end;
        if (end != TurnAlignment::npos) {
            match.kind = TurnState::Same;
            for (std::size_t i = first_run; i < out.runs.size(); ++i) {
                const auto& run = out.runs[i];
                if (run.state_begin >= t.end) break;
                if (run.kind == TurnState::Elided) {
                    match.kind = TurnState::Elided;
                    break;
                }
            }
        }
        out.turns.push_back(match);
    }
    return out;
}

}  // namespace strata::ds4
