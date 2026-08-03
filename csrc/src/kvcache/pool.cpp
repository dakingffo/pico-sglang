#include "../../include/picosgl/kvcache/pool.hpp"

namespace picosgl::detail {
    MHAKVCachePool::MHAKVCachePool(
        size_t num_kv_heads,
        size_t num_layers,
        size_t head_dim,
        size_t num_pages,
        size_t page_size,
        torch::ScalarType dtype,
        torch::Device     device
    ) : num_layers_(num_layers) {
        namespace py = pybind11;
        py::module_ picosgl_distributed = py::module_::import("picosgl.distributed");
        size_t tp_size = picosgl_distributed.attr("get_tp_info")().attr("size").cast<size_t>();
        size_t local_kv_heads = utility::even_div(num_kv_heads, tp_size, true);
        buffer_ = torch::empty(
            {2, num_layers, num_pages, page_size, local_kv_heads, head_dim},
            torch::TensorOptions().dtype(dtype).device(device)
        );
    }

    torch::Tensor MHAKVCachePool::k_cache(int index) noexcept {
        return buffer_.index({0, index});
    }

    torch::Tensor MHAKVCachePool::v_cache(int index) noexcept {
        return buffer_.index({1, index});
    }

    void MHAKVCachePool::store_kv(
        torch::Tensor k, 
        torch::Tensor v, 
        torch::Tensor out_loc, 
        int layer_id
    ) {
        namespace py = pybind11;
        py::module_ picosgl_kernel = py::module_::import("picosgl.kernel");
        troch::Tensor k_buffer = buffer_[0][layer_id];
        troch::Tensor v_buffer = buffer_[1][layer_id];
        troch::IntArrayRef shape = k_buffer.sizes();
        picosgl_kernel.attr("store_cache")(
            k_buffer.view({shape[0] * shape[1], shape[2], shape[3]}),
            v_buffer.view({shape[0] * shape[1], shape[2], shape[3]}),
            out_loc,
            k,
            v
        );
    }

    torch::Device MHAKVCachePool::device() const noexcept {
        return buffer_.device();
    }

    torch::ScalarType MHAKVCachePool::dtype() const noexcept {
        return buffer_.scalar_type();
    }

    size_t MHAKVCachePool::num_layers() const noexcept {
        return num_layers_;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PicoSGL C++ KV Cache Pool Bindings";

    py::class_<picosgl::MHAKVCachePool>(m, "MHAKVCachePool")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, torch::ScalarType, torch::Device>(),
            py::arg("num_kv_heads"),
            py::arg("num_layers"),
            py::arg("head_dim"),
            py::arg("num_pages"),
            py::arg("page_size"),
            py::arg("dtype"),
            py::arg("device")
        )

        .def("k_cache", &picosgl::MHAKVCachePool::k_cache, py::arg("index"))
        .def("v_cache", &picosgl::MHAKVCachePool::v_cache, py::arg("index"))
        .def("store_kv", &picosgl::MHAKVCachePool::store_kv,
            py::arg("k"), 
            py::arg("v"), 
            py::arg("out_loc"), 
            py::arg("layer_id")
        )

        .def_property_readonly("device", &picosgl::MHAKVCachePool::device)
        .def_property_readonly("dtype", &picosgl::MHAKVCachePool::dtype)
        .def_property_readonly("num_layers", &picosgl::MHAKVCachePool::num_layers);
    
    m.def("make_mha_kvcache_pool", &picosgl::make_mha_kvcache_pool,
        py::arg("model_config"),
        py::arg("num_pages"),
        py::arg("page_size"),
        py::arg("dtype"),
        py::arg("device")
    );
}