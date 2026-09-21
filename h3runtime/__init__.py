"""Runtime propio de MiniMax H3 para H3 Prompt Studio.

Motor aislado, con versiones congeladas y sin depender de ComfyUI ni Wan2GP.
Ver docs/RUNTIME_DESIGN.md para el diseno completo.

Estado de implementacion:
  E1 (hecho)     quant.py + checkpoint.py   - cargar y decuantizar pesos
  E1 (hecho)     layout.py                  - split QKV (56 heads x 3 x 128)
  E2 (pendiente) model.py                   - transformer H3
  E3 (pendiente) offload.py                 - politica VRAM/RAM
  E4 (pendiente) pipeline.py                - forward FL2VA minimo
"""

from .quant import (
    build_hadamard,
    dequantize_int8,
    unrotate_convrot_weight,
    rotate_activations,
)
from .checkpoint import (
    H3Checkpoint,
    H3CheckpointMeta,
    CheckpointError,
    find_checkpoint,
)
from .layout import (
    split_interleaved_qkv,
    split_interleaved_qkv_scale,
    qkv_to_packed_heads,
    packed_qkv_size,
    inner_size,
)

__all__ = [
    "build_hadamard",
    "dequantize_int8",
    "unrotate_convrot_weight",
    "rotate_activations",
    "H3Checkpoint",
    "H3CheckpointMeta",
    "CheckpointError",
    "find_checkpoint",
    "split_interleaved_qkv",
    "split_interleaved_qkv_scale",
    "qkv_to_packed_heads",
    "packed_qkv_size",
    "inner_size",
]
