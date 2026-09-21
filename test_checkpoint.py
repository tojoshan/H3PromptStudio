"""Tests del lector de checkpoint de MiniMax H3.

E1: verifican metadata, introspeccion, de-cuantizacion y geometria de atencion
sobre el checkpoint real, sin GPU.

Ejecutar:
    .venv\\Scripts\\python.exe test_checkpoint.py
"""

from pathlib import Path

import torch

from h3runtime.checkpoint import (
    H3Checkpoint,
    CheckpointError,
    find_checkpoint,
    DEFAULT_CHECKPOINT_CANDIDATES,
)

EXPECTED_BLOCKS = 50


def _has_checkpoint() -> bool:
    try:
        find_checkpoint()
        return True
    except CheckpointError:
        return False


# ---------------------------------------------------------------------------
# Localizacion y errores explicitos
# ---------------------------------------------------------------------------

def test_find_checkpoint_raises_when_missing():
    try:
        find_checkpoint("Z:/no/existe.safetensors")
    except CheckpointError as exc:
        assert "no encontrado" in str(exc).lower() or "no se encontro" in str(exc).lower(), exc
    else:
        raise AssertionError("deberia lanzar CheckpointError con ruta inexistente")


def test_open_rejects_missing_path():
    try:
        with H3Checkpoint.open("Z:/no/existe.safetensors"):
            pass
    except CheckpointError:
        pass
    else:
        raise AssertionError("deberia fallar al abrir una ruta inexistente")


def test_get_raw_rejects_unknown_tensor():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        try:
            ckpt.get_raw("no.existe.tensor")
        except CheckpointError:
            pass
        else:
            raise AssertionError("deberia fallar con un tensor inexistente")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_is_declared():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        meta = ckpt.meta
        assert meta.partition == "FL2VA", meta.partition
        assert meta.quantization_format == "int8_convrot", meta.quantization_format
        assert meta.adaln_curve_rank == 8, meta.adaln_curve_rank
        assert meta.adaln_curve_grid == 1025, meta.adaln_curve_grid
        assert meta.source_revision, "falta source_revision (necesaria para reproducibilidad)"
        assert meta.is_convrot(), "el checkpoint deberia declararse convrot"


def test_additional_metadata_fields():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        # Campos utiles para trazabilidad del render.
        assert ckpt.meta.repo_id, "falta repo_id"
        assert ckpt.meta.raw.get("adaln_curve_centered") in ("true", "True"), ckpt.meta.raw


# ---------------------------------------------------------------------------
# Estructura
# ---------------------------------------------------------------------------

def test_block_count():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        keys = ckpt.keys()
        import re

        blocks = {int(m.group(1)) for m in (re.match(r"blocks\.(\d+)\.", k) for k in keys) if m}
        assert len(blocks) == EXPECTED_BLOCKS, len(blocks)
        assert min(blocks) == 0 and max(blocks) == EXPECTED_BLOCKS - 1


def test_attention_geometry():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        geo = ckpt.attention_qkv_shapes(0)
        assert geo["head_dim"] == 128, geo
        # 21504 = 42 heads * 4 proyecciones * 128
        assert geo["packed_heads"] == 168, geo
        assert geo["out_features"] == 5376, geo


# ---------------------------------------------------------------------------
# Lectura de pesos
# ---------------------------------------------------------------------------

def test_get_weight_dequantizes_and_unrotates():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        w = ckpt.get_weight("blocks.0.mlp.fc1.weight", dtype=torch.float32)
        assert w.dtype == torch.float32
        assert w.shape == (28672, 5376), w.shape
        assert torch.isfinite(w).all()
        assert w.abs().max().item() < 10.0, w.abs().max().item()


def test_get_weight_passthrough_for_unquantized():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        # norm1 esta en BF16: no debe intentar decuantizar.
        norm = ckpt.get_weight("blocks.0.norm1.weight", dtype=torch.float32)
        assert norm.dtype == torch.float32
        assert norm.shape == (5376,), norm.shape


def test_quant_info_is_json():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    with H3Checkpoint.open() as ckpt:
        info = ckpt.quant_info("blocks.0.mlp.fc1.weight")
        assert info.get("format") == "int8_tensorwise", info
        assert info.get("convrot") is True, info
        assert int(info.get("convrot_groupsize", 0)) == 256, info


def test_all_quantized_weights_are_finite():
    if not _has_checkpoint():
        print("SKIP - no hay checkpoint.")
        return
    # Muestreo de varios bloques para no cargar 20 GB.
    sample = [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.25.mlp.fc1.weight",
        "blocks.49.mlp.fc2.weight",
    ]
    with H3Checkpoint.open() as ckpt:
        for name in sample:
            w = ckpt.get_weight(name, dtype=torch.float32)
            assert torch.isfinite(w).all(), name
            assert w.abs().mean().item() < 1.0, (name, w.abs().mean().item())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} tests de checkpoint OK.")
