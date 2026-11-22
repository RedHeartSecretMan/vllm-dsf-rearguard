#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

void fused_deepseek_v4_kv_rope_insert(
    torch::stable::Tensor const& kv, torch::stable::Tensor& k_cache,
    torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, int64_t cache_block_size,
    std::optional<torch::stable::Tensor> fp8_scale, bool use_packed_format);

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "fused_deepseek_v4_kv_rope_insert(Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int cache_block_size, Tensor? fp8_scale=None, "
      "bool use_packed_format=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(
      "fused_deepseek_v4_kv_rope_insert",
      TORCH_BOX(&fused_deepseek_v4_kv_rope_insert));
}
