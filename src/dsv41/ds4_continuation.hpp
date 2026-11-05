#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4 {

// Whole-history continuation. A live conversation is reused as one unit: clients
// may prune old pictures, drop reasoning or rewrite earlier turns, and none of that
// is reconciled. The live KV state is kept exactly as it is, and everything the
// request carries after the state's latest output is appended to it as new input.
class Continuation {
public:
    Continuation() = default;
    Continuation(std::size_t model_from, std::size_t request_from)
        : model_from_(model_from), request_from_(request_from) {}

    bool valid() const { return model_from_ > 0; }
    std::size_t model_from() const { return model_from_; }      // live tokens kept whole
    std::size_t request_from() const { return request_from_; }  // first request token appended

    // Request positions before the continuation point belong to history the
    // state already holds; they have no model position of their own (-1).
    std::int64_t to_model(std::int64_t request_pos) const {
        if (!valid()) return request_pos;
        if (request_pos < static_cast<std::int64_t>(request_from_)) return -1;
        return request_pos - static_cast<std::int64_t>(request_from_) + static_cast<std::int64_t>(model_from_);
    }

    std::size_t to_request(std::size_t model_pos) const {
        if (!valid()) return model_pos;
        return model_pos < model_from_ ? std::min(model_pos, request_from_)
                                       : model_pos - model_from_ + request_from_;
    }

private:
    std::size_t model_from_ = 0;
    std::size_t request_from_ = 0;
};

// The state's most recent output (the answer it produced last) identifies where the
// request's new input begins: its last occurrence in the request. A few trailing
// state tokens may never appear there (drafts accepted past a stop, an ending the
// client tokenizes differently), so the anchor may end slightly before the state's
// end. Picture placeholders carry no identity and never form an anchor. At least
// one request token always remains to be read.
template <typename StateToken, typename RequestToken>
inline Continuation find_continuation(const std::vector<StateToken>& state,
                                      const std::vector<RequestToken>& request, int image_token) {
    constexpr std::size_t anchor = 24;
    constexpr std::size_t settle = 32;
    if (state.size() < anchor || request.size() <= anchor) return {};
    const auto searchable = request.end() - 1;
    const auto same = [](const RequestToken& a, const StateToken& b) {
        return static_cast<std::int64_t>(a) == static_cast<std::int64_t>(b);
    };
    for (std::size_t skip = 0; skip <= settle && skip + anchor <= state.size(); ++skip) {
        const auto end = state.end() - static_cast<std::ptrdiff_t>(skip);
        const auto begin = end - static_cast<std::ptrdiff_t>(anchor);
        if (image_token >= 0 &&
            std::find(begin, end, static_cast<StateToken>(image_token)) != end) continue;
        const auto hit = std::find_end(request.begin(), searchable, begin, end, same);
        if (hit == searchable) continue;
        return {state.size(), static_cast<std::size_t>(hit - request.begin()) + anchor};
    }
    return {};
}

}  // namespace strata::ds4
