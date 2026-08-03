#include "../namespace.hpp"
#include "../utility/math.hpp"

namespace picosgl::detail {
    template <typename Derived>
    struct BaseKVCachePool {
        torch::Tensor k_cache() const noexcept {
            return static_cast<const Derived*>(this)->k_cache();
        }

        torch::Tensor v_cache() const noexcept {
            return static_cast<const Derived*>(this)->v_cache();
        }

        void store_kv(
            torch::Tensor k, 
            torch::Tensor v, 
            torch::Tensor out_loc,
            size_t   table_idx
        ) noexcept {
            static_cast<Derived*>(this)->store_kv(k, v, out_loc, table_idx);
        }

        torch::Device device() const noexcept {
            return static_cast<const Derived*>(this)->device();
        }

        torch::ScalarType dtype() const noexcept {
            return static_cast<const Derived*>(this)->dtype();
        }

        size_t num_layers() const noexcept {
            return static_cast<const Derived*>(this)->num_layers();
        }
    };

    class MHAKVCachePool : public BaseKVCachePool<MHAKVCachePool> {
    public: /* Implement at src/kvcache/pool.cpp */
        MHAKVCachePool(
            size_t num_kv_heads,
            size_t num_layers,
            size_t head_dim,
            size_t num_pages,
            size_t page_size,
            torch::ScalarType dtype,
            torch::Device     device
        );

        ~MHAKVCachePool() = default;

        torch::Tensor k_cache(int index) noexcept;
        torch::Tensor v_cache(int index) noexcept;
        void store_kv(
            torch::Tensor k, 
            torch::Tensor v, 
            torch::Tensor out_loc, 
            int           layer_id
        );

        torch::Device     device()     const noexcept;
        torch::ScalarType dtype()      const noexcept;
        size_t       num_layers() const noexcept;

    private:
        torch::Tensor buffer_;
        size_t   num_layers_;
    };
}

namespace picosgl {
    using detail::MHAKVCachePool;

    inline MHAKVCachePool make_mha_kvcache_pool(
        py::object        model_config,
        size_t            num_pages,
        size_t            page_size,
        torch::ScalarType dtype,
        torch::Device     device
    ) {
        return MHAKVCachePool{
            model_config.attr("num_kv_heads").cast<size_t>(),
            model_config.attr("num_layers").cast<size_t>(),
            model_config.attr("head_dim").cast<size_t>(),
            num_pages,
            page_size,
            dtype,
            device,
        };
    } 
}