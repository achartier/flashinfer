"""Focused host tests for the native MXFP4 x MXFP8 MMA contract."""

import pytest

cutlass = pytest.importorskip("cutlass")

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.dynamic_mainloop import (  # noqa: E402
    A_DTYPE,
    B_DTYPE,
    SF_DTYPE,
    SF_VEC_SIZE,
    align_valid_tokens,
    build_idesc_host,
    build_static_idesc_base,
    validate_mma_contract,
)


def test_swap_ab_dtype_and_scale_contract():
    validate_mma_contract(
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        sf_dtype=SF_DTYPE,
        sf_vec_size=SF_VEC_SIZE,
        mma_tiler_mnk=(256, 256, 256),
    )
    with pytest.raises(ValueError, match="instruction A"):
        validate_mma_contract(
            a_dtype=B_DTYPE,
            b_dtype=A_DTYPE,
            sf_dtype=SF_DTYPE,
            sf_vec_size=SF_VEC_SIZE,
            mma_tiler_mnk=(256, 256, 256),
        )
    with pytest.raises(ValueError, match="sf_vec_size=32"):
        validate_mma_contract(
            a_dtype=A_DTYPE,
            b_dtype=B_DTYPE,
            sf_dtype=SF_DTYPE,
            sf_vec_size=16,
            mma_tiler_mnk=(256, 256, 256),
        )


@pytest.mark.parametrize(
    ("valid", "aligned"),
    [(1, 16), (16, 16), (17, 32), (127, 128), (241, 256), (256, 256)],
)
def test_valid_token_alignment(valid, aligned):
    assert align_valid_tokens(valid) == aligned


def test_static_descriptor_is_mxf8f6f4_e2m1_e4m3_ue8m0_k32():
    # a=E2M1(5), b=E4M3(0), scale=UE8M0(1), M=(M>>7), K=32(0).
    assert build_static_idesc_base(umma_m=128) == 0x08800280
    assert build_static_idesc_base(umma_m=256) == 0x10800280


def test_runtime_descriptor_encodes_aligned_n_and_scale_data_ids():
    desc = build_idesc_host(
        umma_m=256,
        valid_tokens_in_tile=17,
        sfa_tmem_addr=0x80000000,
        sfb_tmem_addr=0x40000000,
    )
    assert (desc >> 17) & 0x3F == 4  # align16(17) / 8
    assert (desc >> 29) & 0x3 == 2
    assert (desc >> 4) & 0x3 == 1


@pytest.mark.parametrize("valid", [0, 257])
def test_runtime_descriptor_rejects_unissuable_n(valid):
    with pytest.raises(ValueError, match="dynamic N"):
        build_idesc_host(umma_m=128, valid_tokens_in_tile=valid)
