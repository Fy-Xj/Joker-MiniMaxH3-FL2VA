#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <limits>
#include <stdexcept>

#include <c10/hip/HIPGuard.h>

namespace infinicore::op::mha_impl::flashattn {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    const int64_t num_q_heads = q.size(2);
    const int64_t num_kv_heads = k.size(2);
    if (q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.size(1) == 1 && k.size(1) == 1 && v.size(1) == 1) {
        at::Tensor value = v;
        if (num_kv_heads != num_q_heads) {
            const int64_t repeat = num_q_heads / num_kv_heads;
            value = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        }
        out_work.copy_(value);
        if (out_need_copy_back) {
            p->out->copy_from(out_work_ic);
        }
        return;
    }

    at::Tensor k_attn = k;
    at::Tensor v_attn = v;
    if (num_kv_heads != num_q_heads) {
        const int64_t repeat = num_q_heads / num_kv_heads;
        k_attn = k.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
        v_attn = v.unsqueeze(3).repeat({1, 1, 1, repeat, 1}).reshape(q.sizes());
    }

    // P5: 优先用厂商 flash-attn 的 varlen 内核（与 PyTorch 参考侧同一 kernel），
    // 失败时回退到 ATen SDPA。全自注意力 = varlen 只有单序列，cu_seqlens = [0, B*S]。
    auto q_h = q.permute({0, 2, 1, 3});
    auto k_h = k_attn.permute({0, 2, 1, 3});
    auto v_h = v_attn.permute({0, 2, 1, 3});
    at::Tensor result_h;
    bool used_flash = false;
    try {
        const int64_t nb = q_h.size(0);
        const int64_t nq = q_h.size(2);
        const int64_t total = nb * nq;
        auto q_s = q_h.reshape({total, q_h.size(1), q_h.size(3)}).contiguous();
        auto k_s = k_h.reshape({total, k_h.size(1), k_h.size(3)}).contiguous();
        auto v_s = v_h.reshape({total, v_h.size(1), v_h.size(3)}).contiguous();
        auto opts = at::TensorOptions().dtype(at::kInt).device(q.device());
        auto cu = at::tensor({static_cast<int64_t>(0), total}, opts);
        std::optional<at::Tensor> out_opt = std::nullopt;
        std::optional<at::Tensor> seqused_k_opt = std::nullopt;
        std::optional<const at::Tensor> leftpad_k_opt = std::nullopt;
        std::optional<at::Tensor> block_table_opt = std::nullopt;
        std::optional<at::Tensor> alibi_opt = std::nullopt;
        auto res = flash::vllm_mha_varlen_fwd(
            q_s, k_s, v_s,
            out_opt,
            cu, cu,
            seqused_k_opt, leftpad_k_opt, block_table_opt, alibi_opt,
            static_cast<int>(total), static_cast<int>(total),
            0.0f, static_cast<float>(scale),
            false, is_causal,
            -1, -1, 0.0f, false, std::nullopt);
        if (!res.empty() && res[0].defined()) {
            result_h = res[0].reshape({nb, nq, q_h.size(1), q_h.size(3)});
            used_flash = true;
        }
    } catch (...) {
        used_flash = false;
    }
    if (!used_flash) {
        result_h = at::scaled_dot_product_attention(
            q_h, k_h, v_h,
            std::nullopt, 0.0, is_causal,
            std::optional<double>(static_cast<double>(scale)));
    }
    auto result = result_h.permute({0, 2, 1, 3});
    out_work.copy_(result);
    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
