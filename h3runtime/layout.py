"""Reconstruccion del layout de atencion de MiniMax H3.

El checkpoint guarda `qkv_proj` como un solo tensor con las 3 proyecciones
**interleaved por head**:

    qkv_proj.weight: [heads * 3 * head_dim, hidden]
        -> reshape [heads, 3, head_dim, hidden]

Con la geometria confirmada del config oficial de H3:

    num_attention_heads = 56
    attention_head_dim  = 128
    hidden_size         = 5376
    ffn_hidden_size     = 14336
    num_layers          = 50
    token_refiner_num_layers = 2
    text_dim            = 5120

de modo que:

    qkv_proj  [21504, 5376] = [56 * 3 * 128, 5376]
    q_proj    [ 7168, 5376] = [56 * 128, 5376]
    out_proj  [ 5376, 7168] = [5376, 56 * 128]

El checkpoint NO guarda q_proj/k_proj/v_proj por separado: guarda el packed y hay
que partirlo. La escala [N, 1] debe partirse con la MISMA permutacion.
"""

import torch

# Geometria confirmada del config.json oficial de MiniMax H3.
H3_NUM_LAYERS = 50
H3_TOKEN_REFINER_LAYERS = 2
H3_NUM_ATTENTION_HEADS = 56
H3_ATTENTION_HEAD_DIM = 128
H3_HIDDEN_SIZE = 5376
H3_FFN_HIDDEN_SIZE = 14336
H3_TEXT_DIM = 5120

QKV_PROJECTIONS = 3  # q, k, v


def packed_qkv_size() -> int:
    return H3_NUM_ATTENTION_HEADS * QKV_PROJECTIONS * H3_ATTENTION_HEAD_DIM


def inner_size() -> int:
    return H3_NUM_ATTENTION_HEADS * H3_ATTENTION_HEAD_DIM


def split_interleaved_qkv(
    qkv_weight: torch.Tensor,
    num_heads: int = H3_NUM_ATTENTION_HEADS,
    head_dim: int = H3_ATTENTION_HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parte un `qkv_proj` packed [heads*3*head_dim, hidden] en (q, k, v).

    El layout de entrada es `[heads, 3, head_dim, hidden]`, de modo que la
    proyeccion `j` de la head `h` ocupa las filas `(h*3 + j)*head_dim`.

    Devuelve tres tensores `[heads*head_dim, hidden]` contiguos.
    """
    if qkv_weight.dim() < 2:
        raise ValueError(f"qkv_weight debe tener al menos 2 dims, recibido {tuple(qkv_weight.shape)}")

    total = num_heads * QKV_PROJECTIONS * head_dim
    if qkv_weight.shape[0] != total:
        raise ValueError(
            f"qkv_proj tiene {qkv_weight.shape[0]} filas, se esperaban {total} "
            f"(heads={num_heads} * 3 * head_dim={head_dim})"
        )

    trailing = qkv_weight.shape[1:]
    grouped = qkv_weight.reshape(num_heads, QKV_PROJECTIONS, head_dim, *trailing)
    q, k, v = (grouped[:, index].reshape(num_heads * head_dim, *trailing).contiguous() for index in range(QKV_PROJECTIONS))
    return q, k, v


def split_interleaved_qkv_scale(
    scale: torch.Tensor,
    num_heads: int = H3_NUM_ATTENTION_HEADS,
    head_dim: int = H3_ATTENTION_HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parte una escala [N, 1] (o [N]) de qkv_proj con la MISMA permutacion.

    Devuelve tres tensores con la forma de la entrada (incluida la columna 1 si
    la escala era [N, 1]).
    """
    total = num_heads * QKV_PROJECTIONS * head_dim
    if scale.shape[0] != total:
        raise ValueError(
            f"weight_scale tiene {scale.shape[0]} filas, se esperaban {total}"
        )

    col = scale.shape[1:]
    grouped = scale.reshape(num_heads, QKV_PROJECTIONS, head_dim, *col)
    q, k, v = (grouped[:, index].reshape(num_heads * head_dim, *col).contiguous() for index in range(QKV_PROJECTIONS))
    return q, k, v


def qkv_to_packed_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int = H3_NUM_ATTENTION_HEADS,
    head_dim: int = H3_ATTENTION_HEAD_DIM,
) -> torch.Tensor:
    """Empaqueta (q, k, v) en `[3, heads, head_dim, ...]` para atencion.

    Es el layout que consume `scaled_dot_product_attention` tras reshape
    a [3, batch, heads, seq, head_dim].
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.shape[0] != num_heads * head_dim:
            raise ValueError(f"{name} tiene {tensor.shape[0]} filas, se esperaban {num_heads * head_dim}")
    trailing = q.shape[1:]
    packed = torch.stack(
        [
            t.reshape(num_heads, head_dim, *trailing)
            for t in (q, k, v)
        ],
        dim=0,
    )
    return packed
