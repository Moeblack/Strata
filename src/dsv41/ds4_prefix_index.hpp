#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4 {

// Compressed exact-prefix index; coordinates are TOKENS, keys are indivisible
// logical atoms. A multimodal key must include modality, token width and content
// identity, not merely its placeholder token. Fingerprint is any dual-hash value
// with operator==. Prefix hashes must cover whole atoms, including image identity.
//
// Sources are immutable while indexed. One stable source ID can own several
// endpoints (a full snapshot and its checkpoints). The index owns only endpoint
// and branch metadata: no tokens, atom arrays or per-source prefix-hash arrays.
// There are at most two nodes per distinct endpoint, plus the root.
//
// Hash equality is only a candidate filter. Before loading a result, the caller
// MUST verify that source's token/image prefix exactly. Failed candidates may be
// skipped without discarding other sources on the same matching radix path.
// This class is not synchronized; readers and updates must be serialized.
template <typename AtomKey, typename Fingerprint, typename Compare = std::less<AtomKey>>
class ExactPrefixRadixIndex {
public:
    using Id = uint64_t;
    struct Match {
        Id id = 0;
        size_t length = 0;
        bool found = false;
        explicit operator bool() const { return found; }
    };

    ExactPrefixRadixIndex() = default;
    ExactPrefixRadixIndex(const ExactPrefixRadixIndex&) = delete;
    ExactPrefixRadixIndex& operator=(const ExactPrefixRadixIndex&) = delete;
    ExactPrefixRadixIndex(ExactPrefixRadixIndex&&) = delete;
    ExactPrefixRadixIndex& operator=(ExactPrefixRadixIndex&&) = delete;

    bool empty() const { return endpoint_count_ == 0; }
    size_t size() const { return endpoint_count_; }
    size_t source_count() const { return sources_.size(); }

    void clear() {
        sources_.clear();
        root_.children.clear();
        endpoint_count_ = 0;
    }

    // Insert one whole-atom endpoint of source id. Callbacks are used only during
    // this update and are never retained:
    //   prefix_hash(token_length) -> Fingerprint, for the inserted source;
    //   atom_at(token_position) -> AtomKey, for the inserted source;
    //   common_prefix(existing_id, token_limit) -> token length of the EXACT
    //     common whole-atom prefix, capped at token_limit;
    //   representative_atom_at(existing_id, token_position) -> AtomKey.
    // Existing sources need only be normalized/read on this update path.
    // Both endpoint length and all returned LCPs must be atom boundaries.
    // Duplicate (id,length), zero length, or an inconsistent LCP returns false.
    template <typename PrefixHash, typename AtomAt, typename CommonPrefix, typename RepresentativeAtomAt>
    bool insert(Id id, size_t length, PrefixHash prefix_hash, AtomAt atom_at,
                CommonPrefix common_prefix, RepresentativeAtomAt representative_atom_at) {
        if (length == 0) return false;
        auto source = sources_.find(id);
        if (source != sources_.end() && source->second.count(length) != 0) return false;
        Node* parent = &root_;
        for (;;) {
            const AtomKey key = atom_at(parent->length);
            auto edge = parent->children.find(key);
            if (edge == parent->children.end()) {
                auto leaf = std::make_unique<Node>(length, prefix_hash(length), id, parent);
                Node* inserted = leaf.get();
                parent->children.emplace(key, std::move(leaf));
                add_endpoint(*inserted, id);
                return true;
            }
            Node* child = edge->second.get();
            const size_t limit = std::min(length, child->length);
            const size_t common = common_prefix(child->representative, limit);
            if (common <= parent->length || common > limit) return false;
            if (common == child->length) {
                if (common == length) {
                    add_endpoint(*child, id);
                    return true;
                }
                parent = child;
                continue;
            }

            // The old edge's suffix needs one key, read from its representative
            // on the write path. No representative is ever read during find().
            const AtomKey old_key = representative_atom_at(child->representative, common);
            auto branch = std::make_unique<Node>(common, prefix_hash(common),
                                                 child->representative, parent);
            Node* inserted = branch.get();
            if (common != length) {
                const AtomKey new_key = atom_at(common);
                const Compare less = branch->children.key_comp();
                if (!less(old_key, new_key) && !less(new_key, old_key)) return false;
                auto leaf = std::make_unique<Node>(length, prefix_hash(length), id, branch.get());
                inserted = leaf.get();
                branch->children.emplace(new_key, std::move(leaf));
            }
            child->parent = branch.get();
            branch->children.emplace(old_key, std::move(edge->second));
            edge->second = std::move(branch);
            add_endpoint(*inserted, id);
            return true;
        }
    }

    // Removes every checkpoint/full endpoint for id. Other IDs remain stable;
    // empty branches disappear and non-endpoint unary branches are compressed.
    // No source callbacks or token/image reads are needed.
    bool erase(Id id) {
        auto source = sources_.find(id);
        if (source == sources_.end()) return false;
        for (const auto& endpoint : source->second) {
            Node* node = endpoint.second;
            node->endpoints.erase(id);
            --endpoint_count_;
            prune(node);
        }
        sources_.erase(source);
        return true;
    }

    // Returns the longest indexed endpoint strictly above floor and strictly
    // below request_tokens, leaving the final request token for evaluation.
    // Callbacks access ONLY the request's cached index:
    //   prefix_hash(token_length) -> Fingerprint;
    //   atom_at(token_position) -> AtomKey.
    // Each visited compressed boundary gets one full-prefix hash comparison.
    // No token walk, source callback, endpoint-list scan or cold-slot scan occurs.
    // Cost is O(visited branches * (key lookup + request prefix-hash lookup));
    // request prefix_hash/atom_at may use O(log request_atoms) boundary lookup.
    template <typename PrefixHash, typename AtomAt>
    Match find(size_t request_tokens, PrefixHash prefix_hash, AtomAt atom_at, size_t floor = 0) const {
        Match best;
        if (request_tokens == 0 || request_tokens - 1 <= floor) return best;
        const size_t ceiling = request_tokens - 1;
        const Node* parent = &root_;
        while (parent->length < ceiling) {
            const auto edge = parent->children.find(atom_at(parent->length));
            if (edge == parent->children.end()) break;
            const Node& child = *edge->second;
            if (child.length > ceiling || !(prefix_hash(child.length) == child.fingerprint)) break;
            if (child.length > floor && !child.endpoints.empty())
                best = {*child.endpoints.begin(), child.length, true};
            parent = &child;
        }
        return best;
    }

    // Snapshot all candidate IDs on the request's matching path. Callers rank
    // equal-length sources using their own RAM/LRU metadata, then try shorter
    // endpoints if needed. No cold metadata or unrelated branch is visited.
    // The returned IDs remain safe to iterate while the caller updates the index.
    template <typename PrefixHash, typename AtomAt>
    std::vector<Match> find_candidates(size_t request_tokens, PrefixHash prefix_hash,
                                       AtomAt atom_at, size_t floor = 0) const {
        std::vector<Match> matches;
        if (request_tokens == 0 || request_tokens - 1 <= floor) return matches;
        const size_t ceiling = request_tokens - 1;
        const Node* parent = &root_;
        while (parent->length < ceiling) {
            const auto edge = parent->children.find(atom_at(parent->length));
            if (edge == parent->children.end()) break;
            const Node& child = *edge->second;
            if (child.length > ceiling || !(prefix_hash(child.length) == child.fingerprint)) break;
            if (child.length > floor)
                for (Id id : child.endpoints) matches.push_back({id, child.length, true});
            parent = &child;
        }
        return matches;
    }

private:
    struct Node {
        size_t length = 0;
        Fingerprint fingerprint{};
        Id representative = 0;
        Node* parent = nullptr;
        std::set<Id> endpoints;
        std::map<AtomKey, std::unique_ptr<Node>, Compare> children;

        Node() = default;
        Node(size_t n, Fingerprint hash, Id id, Node* p)
            : length(n), fingerprint(std::move(hash)), representative(id), parent(p) {}
    };

    void add_endpoint(Node& node, Id id) {
        node.endpoints.insert(id);
        sources_[id].emplace(node.length, &node);
        ++endpoint_count_;
        refresh(&node);
    }

    static void choose_representative(Node& node) {
        if (!node.endpoints.empty()) node.representative = *node.endpoints.begin();
        else if (!node.children.empty()) node.representative = node.children.begin()->second->representative;
    }

    void refresh(Node* node) {
        while (node != &root_) {
            choose_representative(*node);
            node = node->parent;
        }
    }

    void prune(Node* node) {
        while (node != &root_) {
            Node* parent = node->parent;
            if (node->endpoints.empty() && node->children.size() <= 1) {
                // Update-only metadata walk; no source data is inspected. Using
                // the parent map avoids storing a duplicate incoming atom key.
                auto edge = parent->children.begin();
                while (edge->second.get() != node) ++edge;
                if (node->children.empty()) {
                    parent->children.erase(edge);
                } else {
                    auto child = std::move(node->children.begin()->second);
                    child->parent = parent;
                    edge->second = std::move(child);
                }
            } else {
                choose_representative(*node);
            }
            node = parent;
        }
    }

    Node root_;
    std::unordered_map<Id, std::map<size_t, Node*>> sources_;
    size_t endpoint_count_ = 0;
};

}  // namespace strata::ds4
