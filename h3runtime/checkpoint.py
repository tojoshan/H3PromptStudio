"""Lector del checkpoint de MiniMax H3 (formato pruned INT8 + ConvRot).

Responsabilidades:
  - localizar y abrir el checkpoint (safetensors);
  - exponer su metadata declarada;
  - entregar tensores ya decuantizados y des-rotados, como tensores float;
  - reconstruir el layout de atencion (QKV) cuando corresponda.

No requiere GPU: devuelve tensores en CPU. La colocacion en GPU es
responsabilidad de la capa de offload (E3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .quant import CONVROT_GROUPSIZE, build_hadamard, dequantize_int8, unrotate_convrot_weight

# Rutas candidatas (se usa la primera existente). Configurable por env en el futuro.
DEFAULT_CHECKPOINT_CANDIDATES = [
    r"X:\Wan2GP\ckpts\MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
    r"X:\Comfyui\ComfyUI-Easy-Install\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors",
]


class CheckpointError(RuntimeError):
    """Error explicito al cargar el checkpoint."""


@dataclass
class H3CheckpointMeta:
    """Metadata declarada por el propio checkpoint."""

    raw: dict = field(default_factory=dict)

    @property
    def partition(self) -> str:
        return self.raw.get("partition", "")

    @property
    def precision(self) -> str:
        return self.raw.get("precision", "")

    @property
    def quantization_format(self) -> str:
        return self.raw.get("quantization_format", "")

    @property
    def convrot_format(self) -> str:
        return self.raw.get("convrot_format", "")

    @property
    def adaln_curve_rank(self) -> int:
        return int(self.raw.get("adaln_curve_rank", 0))

    @property
    def adaln_curve_grid(self) -> int:
        return int(self.raw.get("adaln_curve_grid", 0))

    @property
    def source_revision(self) -> str:
        return self.raw.get("source_revision", "")

    @property
    def repo_id(self) -> str:
        return self.raw.get("repo_id", "")

    def is_convrot(self) -> bool:
        return self.quantization_format == "int8_convrot" or "quant" in self.convrot_format


def find_checkpoint(explicit: str | Path | None = None) -> Path:
    """Devuelve la ruta del checkpoint o lanza un error explicito."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint no encontrado: {path}")
        return path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path

    raise CheckpointError(
        "No se encontro ningun checkpoint de MiniMax H3. Copias buscadas:\n  "
        + "\n  ".join(DEFAULT_CHECKPOINT_CANDIDATES)
    )


def _per_tensor_quant_info(raw: bytes | None) -> dict:
    """Parsea el payload `comfy_quant` (JSON) de un tensor cuantizado."""
    if not raw:
        return {}
    data = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
    try:
        return json.loads(bytes(data).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


class H3Checkpoint:
    """Acceso de alto nivel a los tensores del checkpoint de H3.

    Uso:
        with H3Checkpoint.open() as ckpt:
            w = ckpt.get_weight("blocks.0.mlp.fc1.weight")   # float, des-rotado
    """

    def __init__(self, path: Path):
        self.path = path
        self._handle = None
        self.meta = H3CheckpointMeta()

    # -- ciclo de vida -----------------------------------------------------

    def __enter__(self) -> "H3Checkpoint":
        from safetensors import safe_open

        if not self.path.is_file():
            raise CheckpointError(f"Checkpoint no encontrado: {self.path}")
        self._handle = safe_open(str(self.path), framework="pt")
        self.meta = H3CheckpointMeta(raw=dict(self._handle.metadata() or {}))
        return self

    def __exit__(self, *exc) -> None:
        self._handle = None

    @classmethod
    def open(cls, path: str | Path | None = None) -> "H3Checkpoint":
        return cls(find_checkpoint(path))

    # -- introspeccion -----------------------------------------------------

    def keys(self) -> list[str]:
        if self._handle is None:
            raise CheckpointError("El checkpoint no esta abierto (usar 'with').")
        return list(self._handle.keys())

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle.get_slice(name).get_shape())

    def dtype(self, name: str) -> torch.dtype:
        return self._handle.get_slice(name).get_dtype()

    def quant_info(self, name: str) -> dict:
        """Devuelve el JSON comfy_quant del tensor, si existe."""
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        quant_key = base + ".comfy_quant"
        if quant_key not in self.keys():
            return {}
        raw = bytes(self._handle.get_tensor(quant_key).tolist())
        return _per_tensor_quant_info(raw)

    # -- lectura -----------------------------------------------------------

    def get_raw(self, name: str) -> torch.Tensor:
        """Devuelve el tensor tal cual esta guardado (sin decuantizar)."""
        if name not in self._handle.keys():
            raise CheckpointError(f"Tensor inexistente en el checkpoint: {name}")
        return self._handle.get_tensor(name)

    def get_weight(
        self,
        name: str,
        dtype: torch.dtype = torch.float32,
        hadamard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Devuelve el peso utilizable: decuantizado y des-rotado si aplica.

        Si el tensor no esta cuantizado (BF16/F32), se devuelve convertido a
        `dtype`. Si esta cuantizado (I8 + weight_scale), se decuantiza y, si el
        payload declara convrot, se des-rota.
        """
        raw = self.get_raw(name)

        if raw.dtype != torch.int8:
            return raw.to(dtype)

        scale_name = name + "_scale"
        if scale_name not in self._handle.keys():
            raise CheckpointError(f"Tensor cuantizado sin escala: {name}")
        scale = self.get_raw(scale_name)

        dequantized = dequantize_int8(raw, scale, dtype=torch.float32)

        info = self.quant_info(name)
        if info.get("convrot"):
            group_size = int(info.get("convrot_groupsize", CONVROT_GROUPSIZE))
            if hadamard is None:
                hadamard = build_hadamard(group_size, dtype=torch.float32)
            dequantized = unrotate_convrot_weight(
                dequantized, group_size=group_size, hadamard=hadamard
            )

        return dequantized.to(dtype)

    # -- layout de atencion ------------------------------------------------

    def attention_qkv_shapes(self, block: int = 0) -> dict:
        """Devuelve las dimensiones relevantes del bloque de atencion.

        Util para derivar la geometria de heads sin asumir constantes.
        """
        prefix = f"blocks.{block}.attn"
        qkv_shape = self.shape(f"{prefix}.qkv_proj.weight")
        head_dim = self.shape(f"{prefix}.q_norm.weight")[0]
        packed = qkv_shape[0]
        if packed % head_dim != 0:
            raise CheckpointError(
                f"qkv_proj {qkv_shape} no es divisible por head_dim {head_dim}"
            )
        heads_x_proj = packed // head_dim  # = heads * num_projections
        return {
            "qkv_shape": qkv_shape,
            "head_dim": head_dim,
            "packed_heads": heads_x_proj,
            "out_features": qkv_shape[1],
        }
