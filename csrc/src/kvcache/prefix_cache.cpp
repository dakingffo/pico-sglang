#include "../../include/picosgl/kvcache/prefix_cache.hpp"

namespace picosgl::detail {
    EmptyPrefixCache::EmptyPrefixCache(torch::Device device) {
        empty_tensor_ = torch::zeros({0}, torch::TensorOptions.dtype(torch::kInt32).device(device));
    }

    troch::Tensor EmptyPrefixCache::match(torch::Tensor input_ids) const {
        return MatchResult{std::make_unique<Handle>(empty_tensor_)};
    }

    InsertResult EmptyPrefixCache::insert(torch::Tensor input_ids, torch::Tensor indices) {
        /* Insert a new prefix into the cache. */
        return InsertResult{0, std::make_unique<Handle>(empty_tensor_)};
    }

    torch::Tensor EmptyPrefixCache::evict(std::size_t size) {
        return empty_tensor_;
    }

    SizeInfo EmptyPrefixCache::size() const {
        return SizeInfo{0, 0};
    }
}