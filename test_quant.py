"""Tests del de-cuantizador INT8 y la rotacion ConvRot de H3.

No requieren GPU: la des-cuantizacion y la rotacion son operaciones de CPU.
Si el checkpoint de H3 no esta presente, los tests que lo usan se saltean.

Ejecutar:
    .venv\\Scripts\\python.exe test_quant.py
"""

import os
from pathlib import Path

import torch

from h3runtime.quant import (
    build_hadamard,
    dequantize_int8,
    rotate_activations,
    unrotate_convrot_weight,
    CONVROT_GROUPSIZE,
)

# Rutas candidatas al checkpoint (se usa la primera que exista).
CHECKPOINT_CANDIDATES = [
    r"X:\Wan2GP\ckpts\MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
    r"X:\Comfyui\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors",
]


def _checkpoint_path() -> str | None:
    for path in CHECKPOINT_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Hadamard
# ---------------------------------------------------------------------------

def test_hadamard_is_orthogonal():
    for size in (4, 16, 64, 256):
        h = build_hadamard(size)
        assert h.shape == (size, size), h.shape
        identity = torch.eye(size, dtype=torch.float32)
        err = (h @ h.T - identity).abs().max().item()
        assert err < 1e-5, f"Hadamard {size} no es ortogonal: error {err}"


def test_hadamard_rejects_non_power_of_4():
    for bad in (8, 32, 128, 2, 100):
        try:
            build_hadamard(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"size {bad} deberia ser invalido (no es potencia de 4)")


def test_hadamard_is_cached():
    h1 = build_hadamard(256)
    h2 = build_hadamard(256)
    assert h1 is h2, "La matriz Hadamard deberia estar cacheada"


# ---------------------------------------------------------------------------
# De-cuantizacion
# ---------------------------------------------------------------------------

def test_dequantize_rowwise_scale():
    q = torch.tensor([[10, -20, 30], [1, 2, 3]], dtype=torch.int8)
    scale = torch.tensor([[0.5], [0.1]], dtype=torch.float32)
    out = dequantize_int8(q, scale, dtype=torch.float32)
    expected = torch.tensor([[5.0, -10.0, 15.0], [0.1, 0.2, 0.3]])
    assert torch.allclose(out, expected), out


def test_dequantize_scalar_scale():
    q = torch.tensor([[10, -20]], dtype=torch.int8)
    scale = torch.tensor(0.25, dtype=torch.float32)
    out = dequantize_int8(q, scale, dtype=torch.float32)
    assert torch.allclose(out, torch.tensor([[2.5, -5.0]])), out


def test_dequantize_rejects_non_int8():
    try:
        dequantize_int8(torch.zeros(2, 2, dtype=torch.float32), torch.tensor(1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("deberia rechazar qdata no int8")


# ---------------------------------------------------------------------------
# ConvRot: la rotacion debe cancelarse
# ---------------------------------------------------------------------------

def test_unrotate_inverts_rotation():
    # Simula el proceso: W original -> rotado -> des-rotado == W.
    torch.manual_seed(0)
    out_f, in_f = 32, CONVROT_GROUPSIZE
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    h = build_hadamard(CONVROT_GROUPSIZE, dtype=torch.float32)

    # ConvRot: W_rot = W @ H^T por grupos.
    n_groups = in_f // CONVROT_GROUPSIZE
    grouped = w.reshape(out_f, n_groups, CONVROT_GROUPSIZE)
    w_rot = torch.matmul(grouped, h.T).reshape(out_f, in_f)

    recovered = unrotate_convrot_weight(w_rot, hadamard=h)
    err = (recovered - w).abs().max().item()
    assert err < 1e-5, f"La des-rotacion no recupera el peso: error {err}"


def test_rotation_cancels_in_matmul():
    # Propiedad clave: x_rot @ W_rot^T == x @ W^T (sin des-rotar el peso).
    torch.manual_seed(1)
    out_f, in_f = 64, CONVROT_GROUPSIZE
    w = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(8, in_f, dtype=torch.float32)
    h = build_hadamard(CONVROT_GROUPSIZE, dtype=torch.float32)

    n_groups = in_f // CONVROT_GROUPSIZE
    w_rot = torch.matmul(w.reshape(out_f, n_groups, CONVROT_GROUPSIZE), h.T).reshape(out_f, in_f)
    x_rot = rotate_activations(x, hadamard=h)

    ref = x @ w.T
    fast = x_rot @ w_rot.T
    err = (ref - fast).abs().max().item()
    assert err < 1e-4, f"La rotacion no se cancela en el matmul: error {err}"


def test_unrotate_requires_divisible_groupsize():
    w = torch.randn(8, 100, dtype=torch.float32)  # 100 no es divisible por 256
    try:
        unrotate_convrot_weight(w)
    except ValueError:
        pass
    else:
        raise AssertionError("deberia rechazar in_features no divisible por group_size")


# ---------------------------------------------------------------------------
# Integracion con el checkpoint real
# ---------------------------------------------------------------------------

def test_real_checkpoint_block_dequantizes():
    path = _checkpoint_path()
    if path is None:
        print("SKIP - no se encontro el checkpoint de H3.")
        return

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        w = f.get_tensor("blocks.0.mlp.fc1.weight")
        s = f.get_tensor("blocks.0.mlp.fc1.weight_scale")

    assert w.dtype == torch.int8, w.dtype
    assert s.shape[0] == w.shape[0], (s.shape, w.shape)

    dq = dequantize_int8(w, s, dtype=torch.float32)
    assert dq.shape == w.shape, dq.shape
    # Valores finitos y en un rango plausible.
    assert torch.isfinite(dq).all()
    assert dq.abs().max().item() < 10.0, dq.abs().max().item()

    # in_features debe ser divisible por el group_size de H3.
    assert w.shape[1] % CONVROT_GROUPSIZE == 0, w.shape[1]


def test_real_checkpoint_unrotate_preserves_finiteness():
    path = _checkpoint_path()
    if path is None:
        print("SKIP - no se encontro el checkpoint de H3.")
        return

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        w = f.get_tensor("blocks.0.attn.out_proj.weight")  # [5376, 7168]
        s = f.get_tensor("blocks.0.attn.out_proj.weight_scale")

    dq = dequantize_int8(w, s, dtype=torch.float32)
    unrotated = unrotate_convrot_weight(dq)
    assert unrotated.shape == dq.shape
    assert torch.isfinite(unrotated).all()
    # La rotacion ortogonal preserva la norma global (aprox.) y la energia media.
    assert abs(unrotated.abs().mean().item() - dq.abs().mean().item()) < 0.05


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} tests de cuantizacion OK.")
