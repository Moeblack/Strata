#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace strata::ds4 {

struct HistoryEdit {
    std::size_t model_begin;
    std::size_t model_length;
    std::size_t request_begin;
    std::size_t request_length;
};

// Preserve the original causal history positions without moving KV entries.
// Short annotations replacing history are not new instructions and are not
// sent to the model; this projection is independent of their contents.
class HistoryProjection {
public:
    static HistoryProjection make(std::vector<HistoryEdit> edits,
                                  std::size_t model_tokens,
                                  std::size_t request_tokens,
                                  std::size_t retained_tokens) {
        if (edits.empty() || retained_tokens >= request_tokens) return {};

        std::size_t model_cursor = 0;
        std::size_t request_cursor = 0;
        for (const HistoryEdit& edit : edits) {
            if ((edit.model_length == 0 && edit.request_length == 0) ||
                edit.model_begin < model_cursor ||
                edit.request_begin < request_cursor ||
                edit.model_begin > model_tokens ||
                edit.model_length > model_tokens - edit.model_begin ||
                edit.request_begin > retained_tokens ||
                edit.request_length > retained_tokens - edit.request_begin ||
                edit.model_begin - model_cursor !=
                    edit.request_begin - request_cursor) return {};
            model_cursor = edit.model_begin + edit.model_length;
            request_cursor = edit.request_begin + edit.request_length;
        }
        if (model_tokens - model_cursor != retained_tokens - request_cursor) return {};

        const auto signed_limit = static_cast<std::uintmax_t>(
            std::numeric_limits<std::int64_t>::max());
        const std::size_t suffix = request_tokens - retained_tokens;
        if (model_tokens > signed_limit || request_tokens > signed_limit ||
            suffix > signed_limit - model_tokens ||
            suffix > std::numeric_limits<std::size_t>::max() - model_tokens) return {};

        HistoryProjection result;
        result.edits_ = std::move(edits);
        result.model_tokens_ = model_tokens;
        result.request_tokens_ = request_tokens;
        result.retained_tokens_ = retained_tokens;
        result.valid_ = true;
        return result;
    }

    bool valid() const { return valid_; }
    const std::vector<HistoryEdit>& edits() const { return edits_; }
    std::size_t model_tokens() const { return model_tokens_; }
    std::size_t request_tokens() const { return request_tokens_; }
    std::size_t retained_tokens() const { return retained_tokens_; }
    std::size_t expanded_tokens() const {
        return model_tokens_ + (request_tokens_ - retained_tokens_);
    }

    // End positions are accepted. Right bias selects the last edit at a
    // shared request boundary; annotation positions map to its model end.
    std::int64_t to_model(std::int64_t request_pos) const {
        if (!valid_ || request_pos < 0 ||
            static_cast<std::uintmax_t>(request_pos) > request_tokens_) return -1;
        const std::size_t pos = static_cast<std::size_t>(request_pos);
        auto next = std::upper_bound(edits_.begin(), edits_.end(), pos,
            [](std::size_t value, const HistoryEdit& edit) {
                return value < edit.request_begin;
            });
        if (next == edits_.begin()) return request_pos;
        const HistoryEdit& edit = *--next;
        const std::size_t model_end = edit.model_begin + edit.model_length;
        const std::size_t request_end = edit.request_begin + edit.request_length;
        if (pos < request_end) return static_cast<std::int64_t>(model_end);
        return static_cast<std::int64_t>(model_end + (pos - request_end));
    }

    // Model edit interiors clamp to their request start. At a model end,
    // resume mapping from both edit ends, skipping request-only annotations.
    std::size_t to_request(std::size_t model_pos) const {
        if (!valid_ || model_pos > expanded_tokens())
            return std::numeric_limits<std::size_t>::max();
        auto next = std::upper_bound(edits_.begin(), edits_.end(), model_pos,
            [](std::size_t value, const HistoryEdit& edit) {
                return value < edit.model_begin;
            });
        if (next == edits_.begin()) return model_pos;
        const HistoryEdit& edit = *--next;
        const std::size_t model_end = edit.model_begin + edit.model_length;
        if (model_pos < model_end) return edit.request_begin;
        const std::size_t request_end = edit.request_begin + edit.request_length;
        return request_end + (model_pos - model_end);
    }

private:
    std::vector<HistoryEdit> edits_;
    std::size_t model_tokens_ = 0;
    std::size_t request_tokens_ = 0;
    std::size_t retained_tokens_ = 0;
    bool valid_ = false;
};

}  // namespace strata::ds4
