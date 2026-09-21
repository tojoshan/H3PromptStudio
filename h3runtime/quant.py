"""De-cuantizacion INT8 y rotacion ConvRot de MiniMax H3.

Implementacion propia basada en el comportamiento documentado del formato
`int8_convrot` (comfy_quant_v1):

  format            = int8_tensorwise
  convrot           = true
  convrot_groupsize = 256

Dos operaciones:

1. `dequantize_int8`: reconstruye el peso en float como `q * scale`.
   La escala puede ser escalar (tensorwise) o `[N, 1]` (rowwise).

2. `unrotate_convrot_weight`: deshace la rotacion ConvRot del peso.
   ConvRot aplica una matriz Hadamard regular normalizada por grupos de
   `group_size` canales de entrada:

       W_rot = W @ H^T      (por grupos)

   Como H es ortogonal (H @ H^T = I), la inversa es:

       W = W_rot @ H

   La misma H se usa en runtime para rotar la activacion, de modo que el
   producto interno cancela la rotacion: x_rot @ W_rot^T = x @ W^T.
   Aqui des-rotamos el peso explicitamente para poder usar matmul normal.

Notas de performance:
  - La matriz Hadamard se cachea por (size, dtype, device).
  - La operacion se hace por grupos para no materializar matrices gigantes.
"""

import math

import torch

# Tamano de grupo declarado por el checkpoint (potencia de 4).
CONVROT_GROUPSIZE = 256

_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construye una matriz Hadamard regular normalizada de tamano `size`.

    `size` debe ser potencia de 4 (4, 16, 64, 256, ...). Se arma por producto
    de Kronecker del bloque H4 base y se normaliza dividiendo por sqrt(size),
    de modo que la matriz sea ortogonal (H @ H^T = I).
    """
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard regular: el tamano debe ser potencia de 4, recibido {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4

    h_normalized = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h_normalized
    return h_normalized


def dequantize_int8(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruye un peso INT8 como `qdata.float() * scale`.

    Acepta escala escalar (tensorwise) o `[N, 1]` / `[N]` (rowwise).
    """
    if qdata.dtype != torch.int8:
        raise ValueError(f"Se esperaba qdata int8, recibido {qdata.dtype}")

    q = qdata.to(torch.float32)

    if scale.numel() == 1:
        out = q * scale.to(torch.float32)
    else:
        s = scale.to(torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        if s.shape[0] != q.shape[0]:
            raise ValueError(
                f"Escala rowwise incompatible: scale {tuple(s.shape)} vs qdata {tuple(q.shape)}"
            )
        out = q * s

    return out.to(dtype)


def unrotate_convrot_weight(
    weight: torch.Tensor,
    group_size: int = CONVROT_GROUPSIZE,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deshace la rotacion ConvRot de un peso ya decuantizado.

    ConvRot guarda `W_rot = W @ H^T` por grupos de `group_size` canales de
    entrada. Esta funcion devuelve el peso real `W = W_rot @ H`.

    Args:
        weight: peso decuantizado, shape `[out, in]`.
        group_size: tamano de grupo (256 en los checkpoints de H3).
        hadamard: matriz H opcional ya construida (se cachea si es None).
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} no es divisible por group_size {group_size}")

    dtype = weight.dtype
    if hadamard is None:
        hadamard = build_hadamard(group_size, device=weight.device, dtype=torch.float32)

    n_groups = in_f // group_size
    grouped = weight.reshape(out_f, n_groups, group_size).to(torch.float32)
    h = hadamard.to(device=weight.device, dtype=torch.float32)

    unrotated = torch.matmul(grouped, h)
    return unrotated.reshape(out_f, in_f).to(dtype)


def rotate_activations(
    x: torch.Tensor,
    group_size: int = CONVROT_GROUPSIZE,
    hadamard: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rota la activacion con la misma H, para el matmul optimizado.

    Permite calcular `x @ W^T` usando el peso rotado sin des-rotarlo:
    `x_rot @ W_rot^T = (x @ H) @ (W @ H^T)^T = x @ W^T`.
    """
    features = x.shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} no es divisible por group_size {group_size}")

    orig_shape = x.shape
    dtype = x.dtype
    if hadamard is None:
        hadamard = build_hadamard(group_size, device=x.device, dtype=torch.float32)

    n_groups = features // group_size
    grouped = x.reshape(-1, n_groups, group_size).to(torch.float32)
    h = hadamard.to(device=x.device, dtype=torch.float32)

    rotated = torch.matmul(grouped, h)
    return rotated.reshape(orig_shape).to(dtype)
