"""Runtime propio de MiniMax H3 para H3 Prompt Studio.

Motor aislado, con versiones congeladas y sin depender de ComfyUI ni Wan2GP.
Ver docs/RUNTIME_DESIGN.md para el diseno completo.

Estado de implementacion:
  E1 (en curso)  quant.py + checkpoint.py   - cargar y decuantizar pesos
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

__all__ = [
    "build_hadamard",
    "dequantize_int8",
    "unrotate_convrot_weight",
    "rotate_activations",
    "H3Checkpoint",
    "H3CheckpointMeta",
    "CheckpointError",
    "find_checkpoint",
]
