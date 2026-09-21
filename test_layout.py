"""Tests del layout de atencion de MiniMax H3.

Verifican el split QKV packed -> (q, k, v) con la geometria real y, cuando el
checkpoint esta presente, contra las formas reales de los tensores.

Ejecutar:
    .venv\\Scripts\\python.exe test_layout.py
"""

import torch

from h3runtime.layout import (
    H3_ATTENTION_HEAD_DIM,
    H3_FFN_HIDDEN_SIZE,
    H3_HIDDEN_SIZE,
    H3_NUM_ATTENTION_HEADS,
    H3_NUM_LAYERS,
    inner_size,
    packed_qkv_size,
    qkv_to_packed_heads,
    split_interleaved_qkv,
    split_interleaved_qkv_scale,
)
from h3runtime.checkpoint import H3Checkpoint, CheckpointError, find_checkpoint


def _has_checkpoint() -> bool:
    try:
        find_checkpoint()
        return True
    except CheckpointError:
        return False


# ---------------------------------------------------------------------------
# Geometria declarada
# ---------------------------------------------------------------------------

def test_geometry_matches_config():
    assert H3_NUM_ATTENTION_HEADS == 56
    assert H3_ATTENTION_HEAD_DIM == 128
    assert H3_HIDDEN_SIZE == 5376
    assert H3_FFN_HIDDEN_SIZE == 14336
    assert H3_NUM_LAYERS == 50
    assert inner_size() == 7168
    assert packed_qkv_size() == 21504  # 56 * 3 * 128


# ---------------------------------------------------------------------------
# Split QKV (puro)
# ---------------------------------------------------------------------------

def test_split_produces_three_equal_parts():
    packed = torch.randn(packed_qkv_size(), H3_HIDDEN_SIZE)
    q, k, v = split_interleaved_qkv(packed)
    for tensor in (q, k, v):
        assert tensor.shape == (inner_size(), H3_HIDDEN_SIZE), tensor.shape
    # La concatenacion en orden (q,k,v) NO debe devolver el packed directamente
    # (porque estaba interleaved), pero si recomponemos por head debe coincidir.
    stacked = qkv_to_packed_heads(q, k, v)
    assert stacked.shape == (3, H3_NUM_ATTENTION_HEADS, H3_ATTENTION_HEAD_DIM, H3_HIDDEN_SIZE)


def test_split_preserves_values_per_head():
    # Construyo un packed donde cada head/proyeccion es identificable.
    heads, head_dim = H3_NUM_ATTENTION_HEADS, H3_ATTENTION_HEAD_DIM
    packed = torch.zeros(heads * 3 * head_dim, 4)
    for h in range(heads):
        for j in range(3):
            packed[(h * 3 + j) * head_dim:(h * 3 + j + 1) * head_dim] = h * 10 + j
    q, k, v = split_interleaved_qkv(packed, heads, head_dim)
    # q debe contener los valores j=0 de cada head: 0, 10, 20, ...
    assert q.reshape(heads, head_dim, 4)[1, 0, 0].item() == 10
    assert k.reshape(heads, head_dim, 4)[1, 0, 0].item() == 11
    assert v.reshape(heads, head_dim, 4)[1, 0, 0].item() == 12


def test_split_rejects_wrong_shape():
    bad = torch.randn(100, 10)
    try:
        split_interleaved_qkv(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("deberia rechazar filas != 21504")


def test_split_scale_preserves_form():
    scale = torch.rand(packed_qkv_size(), 1)
    qs, ks, vs = split_interleaved_qkv_scale(scale)
    for s in (qs, ks, vs):
        assert s.shape == (inner_size(), 1), s.shape


def test_qkv_to_packed_heads_shape():
    q = torch.randn(inner_size(), H3_HIDDEN_SIZE)
    k = torch.randn(inner_size(), H3_HIDDEN_SIZE)
    v = torch.randn(inner_size(), H3_HIDDEN_SIZE)
    packed = qkv_to_packed_heads(q, k, v)
    assert packed.shape == (3, H3_NUM_ATTENTION_HEADS, H3_ATTENTION_HEAD_DIM, H3_HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# Contra el checkpoint real
# ---------------------------------------------------------------------------

def test_real_qkv_weight_splits_correctly():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        w = ckpt.get_weight("blocks.0.attn.qkv_proj.weight", dtype=torch.float32)
        assert w.shape == (packed_qkv_size(), H3_HIDDEN_SIZE), w.shape
        q, k, v = split_interleaved_qkv(w)
        assert q.shape == (inner_size(), H3_HIDDEN_SIZE), q.shape
        assert torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()


def test_real_out_proj_shape_matches_inner():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        out = ckpt.get_weight("blocks.0.attn.out_proj.weight", dtype=torch.float32)
        # out_proj: [hidden, inner] = [5376, 7168]
        assert out.shape == (H3_HIDDEN_SIZE, inner_size()), out.shape


def test_real_mlp_shapes():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        fc1 = ckpt.get_weight("blocks.0.mlp.fc1.weight", dtype=torch.float32)
        fc2 = ckpt.get_weight("blocks.0.mlp.fc2.weight", dtype=torch.float32)
        # Gated MLP: fc1 [2*ffn, hidden], fc2 [hidden, ffn]
        assert fc1.shape == (2 * H3_FFN_HIDDEN_SIZE, H3_HIDDEN_SIZE), fc1.shape
        assert fc2.shape == (H3_HIDDEN_SIZE, H3_FFN_HIDDEN_SIZE), fc2.shape


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} tests de layout OK.")
