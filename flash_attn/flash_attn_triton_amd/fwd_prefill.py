import torch
import triton
import triton.language as tl
from typing import Literal, Optional, Union
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, AUTOTUNE, compute_alibi_block, compute_fp8_scaling_factors, get_shapes_from_layout, get_strides_from_layout, is_cdna, is_fp8, is_rdna, create_dropout_mask

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = triton.language.constexpr(DROPOUT_USE_PYTORCH)
tl_DROPOUT_DUMP: tl.constexpr = triton.language.constexpr(DROPOUT_DUMP)

# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, stride_sn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, philox_ptrs, sd_mask_ptrs, dropout_mask_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope,
                    descale_q, descale_k, descale_v, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
                    IS_CAUSAL: tl.constexpr,
                    IS_SLIDING_WINDOW: tl.constexpr, window_size_left: tl.constexpr, window_size_right: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr,
                    RETURN_SCORES: tl.constexpr, ACCUMULATOR_TYPE):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
    
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        q_mask = (OFFS_M[:, None] < actual_seqlen_q)
        k_mask = ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
        p_mask = q_mask & k_mask

        # -- compute qk ----
        if IS_FP8 :
            qk += (tl.dot(q, k) * descale_q * descale_k)
        else:
            qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE

        if MASK_STEPS:
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk_scaled = tl.where(mask, qk_scaled, float("-inf"))

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        
        if IS_SLIDING_WINDOW:
            if window_size_left != -1:
                mask_left = (start_n + OFFS_N[None, :]) >= (OFFS_M[:, None] - window_size_left)
                qk_scaled = tl.where(mask_left, qk_scaled, float("-inf"))
            if window_size_right != -1:
                mask_right = (start_n + OFFS_N[None, :]) <= (OFFS_M[:, None] + window_size_right)
                qk_scaled = tl.where(mask_right, qk_scaled, float("-inf"))

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            qk_scaled += bias

        if USE_ALIBI:
            global_m_positions = OFFS_M
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
            
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        
        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        if ENABLE_DROPOUT:
            if tl_DROPOUT_USE_PYTORCH:
                dropout_mask = tl.load(dropout_mask_ptrs, mask=p_mask)
            else:
                rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
                dropout_mask = rng_output > dropout_p
                if tl_DROPOUT_DUMP:
                    tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)

            # return scores with negative values for dropped vals
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            tl.store(sd_mask_ptrs, p, mask=p_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if IS_FP8:
            scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v)
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn
            
    return acc, l_i, m_i

def get_cdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'IS_VARLEN', 'HQ', 'HK']


def get_rdna_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
        # Fall-back config.
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=2),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'IS_VARLEN', 'HQ', 'HK']


def get_autotune_configs():
    if AUTOTUNE:
        if is_rdna():
            return get_rdna_autotune_configs()
        elif is_cdna():
            return get_cdna_autotune_configs()
        else:
            raise ValueError("Unknown Device Type")
    else:
        return [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "waves_per_eu": 1, "PRE_LOAD_V": False},
                num_stages=1,
                num_warps=4,
            ),
        ], [
            "IS_CAUSAL",
            "dropout_p",
            "MAX_SEQLENS_Q",
            "MAX_SEQLENS_K",
            "ACTUAL_BLOCK_DMODEL",
            "IS_VARLEN",
            "HQ",
            "HK",
        ]


autotune_configs, autotune_keys = get_autotune_configs()

@triton.autotune(
    configs=autotune_configs,
    key=autotune_keys,
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(Q, K, V, bias, sinks, Cache_seqlens, Cache_batch_idx,
             Descale_Q, Descale_K, Descale_V, Descale_O, stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_o_z,
             SM_SCALE: tl.constexpr, LSE, Out, stride_qz, stride_qh, stride_qm, stride_qk,
             stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
             stride_oz, stride_oh, stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_sinks_h, stride_az, stride_ah,
             stride_sz, stride_sh, stride_sm, stride_sn, stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, sd_mask, dropout_mask, alibi_slopes, HQ: tl.constexpr,
             HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, IS_VARLEN: tl.constexpr, IS_INFERENCE: tl.constexpr,  IS_CAUSAL: tl.constexpr, 
             IS_SLIDING_WINDOW: tl.constexpr, window_size_left: tl.constexpr, window_size_right: tl.constexpr,
             BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             USE_SINKS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr,
             RETURN_SCORES: tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, 
             IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr):
    # set params
    ACCUMULATOR_TYPE = tl.float32

    # compute offsets
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # handle seqlen
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    elif IS_INFERENCE:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q        
        seqlen_k = tl.load(Cache_seqlens + off_z)
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    block_min_n = 0
    
    if (IS_CAUSAL):
        n_blocks_seqlen = tl.cdiv((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        
    if IS_SLIDING_WINDOW:
        if window_size_right != -1:
            k_last_seen = (start_m + 1) * BLOCK_M - 1 + window_size_right
            n_blocks_sw_right = tl.cdiv(k_last_seen + 1, BLOCK_N)
            n_blocks = min(n_blocks, n_blocks_sw_right)
        
        if window_size_left != -1:
            k_first_seen = start_m * BLOCK_M - window_size_left
            block_min_n = max(0, k_first_seen // BLOCK_N)
            
    if n_blocks <= block_min_n:
        o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
        o_ptrs_mask = offs_m[:, None] < seqlen_q
        tl.store(o_ptrs, acc, mask=o_ptrs_mask)
        
        l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
        l_ptrs = l_offset + offs_m * stride_lse_m 

        l = tl.full([BLOCK_M], value=0.0, dtype=ACCUMULATOR_TYPE)

        l_ptrs_mask = offs_m < MAX_SEQLENS_Q
        tl.store(l_ptrs, l, mask=l_ptrs_mask)
        return

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        sd_mask_offset = sd_mask + off_z * stride_sz + off_h_q * stride_sh 
        sd_mask_ptrs = sd_mask_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        sd_mask_ptrs = None

    if ENABLE_DROPOUT:
        dropout_mask_offset = dropout_mask + off_z * stride_sz + off_h_q * stride_sh
        dropout_mask_ptrs = dropout_mask_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        batch_philox_offset = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh 
        philox_ptrs = batch_philox_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        dropout_mask_ptrs = None
        philox_ptrs = 0
        
    if USE_SINKS:
        sinks_ptr = sinks + off_h_q * stride_sinks_h
        sink_val = tl.load(sinks_ptr)
        m_i = tl.full([BLOCK_M], sink_val, dtype=ACCUMULATOR_TYPE)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
        
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=ACCUMULATOR_TYPE)
    
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    if IS_FP8:
        descale_q = tl.load(Descale_Q + off_z * stride_descale_q_z + off_h_q)
        descale_k = tl.load(Descale_K + off_z * stride_descale_k_z + off_h_k)
        descale_v = tl.load(Descale_V + off_z * stride_descale_v_z + off_h_k)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0

    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)

    n_blocks_to_process = n_blocks - block_min_n 

    if IS_SLIDING_WINDOW:
        n_full_blocks_adj = 0 
    else:
        n_full_blocks = n_blocks - masked_blocks
        n_full_blocks_adj = max(0, n_full_blocks - block_min_n)
    
    masked_blocks_adj = n_blocks_to_process - n_full_blocks_adj
    
    block_min = block_min_n * BLOCK_N
    
    if n_full_blocks_adj > 0:
        block_max = (block_min_n + n_full_blocks_adj) * BLOCK_N
        
        k_ptrs_full = k_ptrs + block_min * stride_kn
        v_ptrs_full = v_ptrs + block_min * stride_vk
        bias_ptrs_full = bias_ptrs
        if USE_BIAS:
            bias_ptrs_full = bias_ptrs + block_min * stride_bn
        sd_mask_ptrs_full = sd_mask_ptrs
        if RETURN_SCORES:
            sd_mask_ptrs_full = sd_mask_ptrs + block_min * stride_sn
        dropout_mask_ptrs_full = dropout_mask_ptrs
        philox_ptrs_full = philox_ptrs
        if ENABLE_DROPOUT:
            dropout_mask_ptrs_full = dropout_mask_ptrs + block_min * stride_sn
            philox_ptrs_full = philox_ptrs + block_min * stride_sn
            
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs_full, v_ptrs_full, bias_ptrs_full, stride_kn, stride_vk, stride_bn, stride_sn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs_full,
                                        sd_mask_ptrs_full, dropout_mask_ptrs_full,
                                        block_min, block_max, 0, 0, 0, alibi_slope,
                                        descale_q, descale_k, descale_v, IS_FP8, FP8_MAX,
                                        False,
                                        IS_SLIDING_WINDOW, window_size_left, window_size_right,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        PRE_LOAD_V, False, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, RETURN_SCORES=RETURN_SCORES, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE
                                        )
        block_min = block_max
    
    block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    if (masked_blocks_adj > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
            
        k_ptrs += block_min * stride_kn
        v_ptrs += block_min * stride_vk
        if USE_BIAS:
            bias_ptrs += block_min * stride_bn
        if RETURN_SCORES:
            sd_mask_ptrs += block_min * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += block_min * stride_sn
            philox_ptrs += block_min * stride_sn
            
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, stride_sn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs,
                                        sd_mask_ptrs, dropout_mask_ptrs, block_min, block_max, offs_n_causal, masked_blocks_adj,
                                        n_extra_tokens, alibi_slope, descale_q, descale_k, descale_v, IS_FP8, FP8_MAX,
                                        IS_CAUSAL,
                                        IS_SLIDING_WINDOW, window_size_left, window_size_right,
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        PRE_LOAD_V, True, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, RETURN_SCORES=RETURN_SCORES, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE
                                        )

    # --- Epilogue ---
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale

    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    l_ptrs = l_offset + offs_m * stride_lse_m 
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL and not USE_SINKS:
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask) 
    else:
        tl.store(l_ptrs, softmax_lse) 

    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)

    if FP8_OUTPUT:
        scale_acc, descale_acc = compute_fp8_scaling_factors(acc, FP8_MAX)
        tl.store(Descale_O + off_z * stride_descale_o_z + off_h_q, descale_acc)
        tl.store(o_ptrs, (acc * scale_acc).to(Out.type.element_ty), mask=o_ptrs_mask)
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

def attention_prefill_forward_triton_impl(
                                        q: torch.Tensor,
                                        k: torch.Tensor,
                                        v: torch.Tensor,
                                        sinks: torch.Tensor,
                                        window_size_left,
                                        window_size_right,
                                        o: torch.Tensor,
                                        sm_scale: float,
                                        alibi_slopes: Optional[torch.Tensor],
                                        causal: bool,
                                        bias: Optional[torch.Tensor],
                                        layout: Literal["bshd", "bhsd", "thd"],
                                        # varlen
                                        cu_seqlens_q: Optional[torch.Tensor], 
                                        cu_seqlens_k: Optional[torch.Tensor],
                                        max_seqlens_q: int, 
                                        max_seqlens_k: int,
                                        # inference
                                        cache_seqlens: Optional[Union[(int, torch.Tensor)]],
                                        cache_batch_idx: Optional[torch.Tensor],
                                        # dropout
                                        dropout_p: float,
                                        philox_seed: Optional[int],
                                        philox_offset: Optional[int],
                                        # misc
                                        return_softmax: bool,
                                        use_exp2: bool,
                                        # fp8
                                        descale_q: Optional[torch.Tensor],
                                        descale_k: Optional[torch.Tensor],
                                        descale_v: Optional[torch.Tensor],
                                        descale_o: Optional[torch.Tensor],
):
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max

        assert is_fp8(q) and is_fp8(k) and is_fp8(v), f"Non fp8 type found: q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}. All tensors must be fp8."

        if is_fp8(o):
            FP8_OUTPUT = True
            assert descale_o is not None, "descale_o is None. In fp8, you need to pass a tensor for descale_o along with a tensor for the output."
        else:
            FP8_OUTPUT = False

        # Get strides for the kernel
        stride_descale_q_z = descale_q.stride(0) if descale_q is not None else None
        stride_descale_k_z = descale_k.stride(0) if descale_k is not None else None
        stride_descale_v_z = descale_v.stride(0) if descale_v is not None else None
        stride_descale_o_z = descale_o.stride(0) if descale_o is not None else None
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        descale_q = descale_k = descale_v = descale_o = None
        stride_descale_q_z = stride_descale_k_z = stride_descale_v_z = stride_descale_o_z = None

    # check flags
    is_varlen = layout == "thd"
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    is_inference = False if cache_seqlens is None else True
    if is_inference:
        assert layout == "bshd", f"{layout} layout is not supported with inference. Use bshd layout"
    if DEBUG:
        print("is_inference:", is_inference)

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = get_shapes_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    def grid(META):
        return triton.cdiv(max_seqlens_q, META["BLOCK_M"]), nheads_q, batch

    # sd_mask is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only. This return holds no useful output aside from debugging.
    use_dropout = (dropout_p > 0.0)
    if use_dropout or return_softmax:
        sd_mask = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(dropout_p, (batch, nheads_q, max_seqlens_q, max_seqlens_k), seed = philox_seed)
        else:
            dropout_mask = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        scores_strides = (sd_mask.stride(0), sd_mask.stride(1), sd_mask.stride(2), sd_mask.stride(3))
    else:
        sd_mask = None
        dropout_mask = None
        scores_strides = (0, 0, 0, 0)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        total_seqlen_q, _, _ = q.shape
        softmax_lse = torch.zeros((nheads_q, total_seqlen_q), device=q.device, dtype=torch.float32)
        stride_lse_h, stride_lse_m = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.zeros((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1),bias.stride(2),
                        bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if sinks is not None:
        stride_sinks_h = sinks.stride()[0]
    IS_SLIDING_WINDOW = (window_size_left != -1) or (window_size_right != -1)
    attn_fwd[grid](q, k, v, bias, sinks, cache_seqlens, cache_batch_idx,
                    descale_q, descale_k, descale_v, descale_o, stride_descale_q_z, stride_descale_k_z, stride_descale_v_z, stride_descale_o_z,
                    sm_scale, softmax_lse, o, *q_strides, *k_strides, *v_strides, *o_strides,
                    *bias_strides, stride_sinks_h, stride_az, stride_ah, *scores_strides, stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q, cu_seqlens_k,
                    dropout_p=dropout_p, philox_seed=philox_seed, philox_offset_base=philox_offset, sd_mask=sd_mask, dropout_mask=dropout_mask, alibi_slopes=alibi_slopes, 
                    HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=max_seqlens_q,
                    MAX_SEQLENS_K=max_seqlens_k, IS_CAUSAL=causal,
                    IS_SLIDING_WINDOW=IS_SLIDING_WINDOW, window_size_left=window_size_left, window_size_right=window_size_right,
                    IS_VARLEN=is_varlen, IS_INFERENCE=is_inference,
                    BLOCK_DMODEL=padded_d_model, USE_BIAS=False if bias is None else True,
                    USE_ALIBI=use_alibi, USE_SINKS=sinks is not None, ENABLE_DROPOUT=dropout_p
                    > 0.0, USE_EXP2=use_exp2, RETURN_SCORES=return_softmax, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT)

    return softmax_lse, sd_mask if return_softmax else None 
