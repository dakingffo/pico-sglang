#include <picosgl/picosgl.hpp>

namespace picosgl::detail {
    struct BaseCacheHandle {
        BaseCacheHandle(std::size_t num_cached_tokens)
            : num_cached_tokens(num_cached_tokens) {}

        virtual ~BaseCacheHandle() = default;

        virtual torch::Tensor get_matched_indices() const = 0;

        const std::size_t num_cached_tokens;
    };

    class CacheHandle {
    public:
        CacheHandle(BaseCacheHandle* handle) : handle_(handle) {}

        ~CacheHandle() = default;

        inline torch::Tensor get_matched_indices() const {
            return handle_->get_matched_indices();
        }

    private:
        std::unique_ptr<BaseCacheHandle> handle_;
    };

    struct SizeInfo {
        std::size_t evictable_size_;
        std::size_t protected_size_;

        inline std::size_t total_size() const noexcept {
            return evictable_size_ + protected_size_;
        }
    }

    struct InsertResult {
        std::size_t num_cached_tokens_;
        CacheHandle cache_handle_;
    }

    struct MatchResult {
        CacheHandle cache_handle_;
    }

    template <typename Derived>
    struct BasePrefixCache {
    public:
        void lock(BaseCacheHandle handle) {
            /* Lock a handle, just change the size info. */
            /* Before the previously-returned tensor of `match_prefix` is used, Handles must be locked */
            static_cast<const Derived*>(this)->lock(handle);
        }

        void unlock(BaseCacheHandle handle) {
            /* Unlock a handle, just change the size info. */
            static_cast<Derived*>(this)->unlock(handle);
        }

        troch::Tensor match(torch::Tensor input_ids) const {
            /* Match prefix and return the indices of the matched prefix in the cache. */
            /* This operation will not modify the cache. */
            /* The returned indices is only safe to use when the handle is locked.*/
            return static_cast<const Derived*>(this)->match(input_ids);
        }

        InsertResult insert(torch::Tensor input_ids, torch::Tensor indices) {
            /* Insert a new prefix into the cache. */
            return static_cast<Derived*>(this)->insert(input_ids);
        }

        torch::Tensor evict(std::size_t size) {
            /* Evict some prefixes from the cache to free up space. */
            /* Note that evict 0 is always safe and does nothing. */
            /* Note that the actual evict size may be larger than the requested size.*/
            return static_cast<Derived*>(this)->evict(size);
        }

        void reset() {
            /* Reset the cache manager and the underlying cache. */
            static_cast<Derived*>(this)->reset();
        }

        SizeInfo size() const {
            /* Get the size information of the cache. */
            return static_cast<const Derived*>(this)->reset();
        }

        void check_integrity() const {
            /* Check the integrity of the cache. Raise an error if the cache is corrupted. */
            static_cast<const Derived*>(this)->check_integrity();
        }
    }

    class EmptyPrefixCache : BasePrefixCache<EmptyPreifxCache> {
    private:
        class Handle_ : BaseCacheHandle {
        public:
            Handle(torch::Tensor empty_tensor) 
                : BaseCacheHandle(0)
                , empty_tensor_(empty_tensor) {}

            ~Handle() = default;

            inline torch::Tensor get_matched_indices() override const {
                return empty_tensor_;
            }

        private:
            torch::Tensor empty_tensor_;
        }

    public:
        using Handle = Handle_;

    public: /* Implement at src/kvcache/prefix_cache.cpp */
        EmptyPrefixCache(torch::Device device);
        ~EmptyPrefixCache() = default;

        inline void lock(CacheHandle handle) { /* pass */ }
        inline void unlock(CacheHandle handle) { /* pass */ }
        inline void reset() { /* pass */}
        inline void check_integrity() const { /* pass */}

        troch::Tensor match(torch::Tensor input_ids) const;
        InsertResult insert(torch::Tensor input_ids, torch::Tensor indices);
        torch::Tensor evict(std::size_t size);
        SizeInfo size();
    
    private:
        torch::Tensor empty_tensor_;
    };

    struct PagedListNode {
        PagedListNode(size_t size) : size_(size) {}
        
        torch::Tensor get_matched_indices() const {
            torch::Tensor
        }

        torch::Tensor  key_;
        torch::Tensor  value_;
        size_t         size_;
        size_t         ref_count_;
        PagedListNode* next_;
    }

    class PagedPrefixCache : BasePrefixCache<PagedPrefixCache> {
    private:
        class Handle_ : BaseCacheHandle {
        public:
            Handle(size_t num_cached_tokens, PagedListNode* node) 
                : BaseCacheHandle(num_cached_tokens)
                , node_(node) {}

            ~Handle() = default;

            inline torch::Tensor get_matched_indices() override const {
                return 
            }

        private:
            PagedListNode* node_;
        }

    public:
        using Handle = Handle_;

    public:
        void lock(BaseCacheHandle handle) {
            handle_.
        }

        void unlock(BaseCacheHandle handle) {

        }

        troch::Tensor match(torch::Tensor input_ids) const {

            return ;
        }

        InsertResult insert(torch::Tensor input_ids, torch::Tensor indices) {
            return ;
        }

        torch::Tensor evict(std::size_t size) {

            return ;
        }

        void reset() {

        }

        SizeInfo size() const {

        }

        void check_integrity() const {

        }

    private:
        size_t page_size;
        size_t evictable_size;
        size_t protected_size;
    }
}

namespace picosgl {
    using detail::CacheHandle;
    using detail::EmptyPrefixCache;
}