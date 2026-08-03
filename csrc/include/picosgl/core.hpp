#include <torch/extension.h>

struct SamplingParams {
    float temperature = 0.0;
    int   top_k       = -1;
    float top_p       = 1.0;
    bool  ignore_eos  = false;
    int   max_tokens  = 1024;

    inline bool is_greedy() const {
        return (temperature <= 0.0 || top_k == -1) && top_p == 1.0;
    }
};

struct Request {
    Request(
        std::size_t    id, 
        torch::Tensor  input_ids, 
        std::size_t    num_cached_tokens, 
        std::size_t    num_output_tokens, 
        std::size_t    table_idx, 
        CacheHandle    cache_handle, 
        SamplingParams sampling_params
    ) : id(id)
      , input_ids(input_ids)
      , num_cached_tokens(num_cached_tokens)
      , num_output_tokens(num_output_tokens)
      , num_device_tokens(input_ids.size())
      , max_device_tokens(input_ids.size() + num_output_tokens)
      , table_idx(table_idx)
      , cache_handle(cache_handle)
      , sampling_params(sampling_params) {}
    
    Request() = default;

    inline std::size_t num_remaining_tokens() const noexcept {
        return max_device_tokens - num_device_tokens;
    }

    inline std::size_t num_extented_tokens() const noexcept {
        return num_device_tokens - num_cached_tokens;
    }

    inline bool available() const noexcept {
        return num_remaining_tokens() > 0;
    }

    inline void complete(int num = 1) noexcept {
        num_cached_tokens = num_device_tokens;
        num_device_tokens += num;
    }

    inline void append(torch::Tensor new_tokens) noexcept {
        input_ids = torch::cat({input_ids, new_tokens}, 0);
    }

    std::size_t    id;
    torch::Tensor  input_ids;
    std::size_t    num_cached_tokens;
    std::size_t    num_output_tokens;
    std::size_t    num_device_tokens;
    std::size_t    max_device_tokens;
    std::size_t    table_idx;
    CacheHandle    cache_handle;
    SamplingParams sampling_params;
};

struct Batch {
    Batch(const std::vector<Request>& requests, const std::string& phase)
        : requests(requests), phase(phase) {}

    ~Batch() = default;

    inline bool is_prefill() const noexcept {
        return phase == "prefill";
    }

    inline bool is_decode() const noexcept {
        return phase == "decode";
    }

    inline std::size_t size() const noexcept {
        return requests.size();
    }

    inline std::size_t padded_size() const noexcept {
        return padded_reqs.size();
    }

    std::vector<Request> requests;
    std::string          phase;

    // these should be set by scheduler
    torch::Tensor        input_ids;
    torch::Tensor        positions;
    torch::Tensor        kvout_loc;
    std::vector<Request> padded_reqs;

    // this should be set by attention backend
    BaseAttnMetadata*    attn_metadata;
};

class Context {
private:
    class Manager_ {
    public:
        Manager(Context& ctx, const std::shared_ptr<Batch>& batch)
            : ctx_(ctx), saved_batch_(ctx.batch_) {
            ctx_.batch_ = batch;
        }

        ~Manager() {
            ctx_.batch_ = saved_batch_;
        }

    private:
        Context& ctx_;
        std::shared_ptr<Batch> saved_batch_;
    };

public:
    Context(std::size_t page_size)
        : page_size(page_size) {}

    ~Context() = default;

    inline Batch& batch() {
        return *batch_;
    }

    using Manager = Manager_;
    /*
    {
        Context::Manager manager(ctx, batch);
        ...
    }
    */

public:
    std::size_t      page_size;
    torch::Tensor    page_table;
    BaseAttnBackend* attn_backend;
    BaseMoeBackend*  moe_backend;
    BaseKVCachePool* kv_cache_pool;

private:
    std::shared_ptr<Batch> batch_ = nullptr;
};