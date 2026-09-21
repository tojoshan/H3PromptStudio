# Diseño del runtime H3 propio

Documento de diseño del runtime de MiniMax H3 de H3 Prompt Studio.
Objetivo: un motor **aislado, reproducible y con versiones congeladas**, sin
depender de ComfyUI ni de Wan2GP.

Estado: **propuesta** — pendiente de aprobación antes de implementar.

---

## 1. Problema que resuelve

Los runners existentes (ComfyUI, Wan2GP) son funcionales pero mutan bajo los
pies del usuario: una actualización rompe el modelo, y una misma sesión puede
fallar en la segunda generación al dejar de reconocer el text encoder como
compatible.

El runtime propio ataca la **clase** de problema, no la instancia:

- código propio → nadie lo actualiza sin nuestro control;
- versiones pineadas → el entorno no deriva;
- errores explícitos → cada fallo dice **qué** falló y **dónde**.

---

## 2. Hechos verificados que condicionan el diseño

Todos medidos sobre los checkpoints reales en esta máquina.

### 2.1 Checkpoint FL2VA pruned INT8

```text
archivo   MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors
tamaño    ~19.6 GB
tensores  932
```

Metadata declarada por el checkpoint:

```text
partition            = FL2VA
precision            = bf16
quantization_format  = int8_convrot
convrot_format       = comfy_quant_v1
adaln_curve_rank     = 8
adaln_curve_grid     = 1025
adaln_curve_centered = true
source_revision      = 73372e6c...   (fijable para reproducibilidad)
```

Estructura de tensores:

```text
adaln_t_table                 [1025, 8]       F32   tabla AdaLN comprimida
blocks.0..49.adaln_proj.linear [96768, 8]     F32   proyección desde la curva
blocks.N.attn.qkv_proj.weight [21504, 5376]   I8    cuantizado
blocks.N.attn.qkv_proj.weight_scale [21504,1] F32   escala por fila
blocks.N.mlp.fc1.weight       [28672, 5376]   I8
blocks.N.mlp.fc2.weight       [5376, 14336]   I8
blocks.N.norm1/norm2.weight   [5376]          BF16
blocks.N.attn.q_norm/k_norm   [128]           BF16
... (+ token_refiner, patch projs, final_layer, rope.inv_freq)
```

Distribución de dtypes en el archivo: `F32: 312`, `BF16: 220`, `U8: 200`, `I8: 200`.

### 2.2 Cuantización

El payload `comfy_quant` de cada tensor cuantizado es JSON:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

Dos consecuencias directas:

1. **INT8 tensorwise**: cada fila se reconstruye como `weight.float() * weight_scale`
   (la escala tiene shape `[N, 1]`).
2. **ConvRot**: los pesos están rotados con una transformación tipo Hadamard por
   grupos de 256 canales. Hay que **des-rotar** antes de usarlos. Esta es la razón
   por la que "cargar el checkpoint" no es un `load_state_dict` directo.

### 2.3 Layout de QKV

El checkpoint guarda `qkv_proj` con las heads **agrupadas**
(`[heads, 3, head_dim, ...]`), pero el modelo espera el layout **interleaved**
(`[3, heads, head_dim, ...]`). La permutación debe aplicarse al peso **y** a su
escala. `head_dim` sale de `q_norm.weight.shape[0]` (128) y
`heads = qkv.shape[0] // (3 * head_dim)`.

### 2.4 Versiones de referencia

Versiones validadas en esta máquina (Wan2GP) para congelar:

```text
torch        2.10.0+cu130
torchaudio   2.10.0+cu130
torchvision  0.25.0+cu130
diffusers    0.36.0
transformers 4.54.0
accelerate   1.15.0
```

Nota: el venv actual de H3 Prompt Studio tiene `torch 2.14.0+cu130` y
`transformers 5.17.0`. El salto `transformers 4 → 5` es un cambio mayor y es
precisamente el tipo de deriva que este diseño busca evitar. **Se decidirá un
venv separado y congelado para el runtime.**

### 2.5 Offload

Opciones evaluadas:

| Librería | Licencia | Estado | Nota |
|---|---|
| `accelerate` | Apache-2.0 | ya instalada (1.15.0) | `cpu_offload`, `dispatch_model`, `init_empty_weights`, `load_checkpoint_and_dispatch` |
| `mmgp` | **non-commercial + atribución, sin SPDX** | no instalada | la que usa Wan2GP para caber en 12 GB |

`mmgp` es la que resuelve el caso extremo, pero su licencia es un riesgo para uso
comercial. **Decisión propuesta: empezar con `accelerate`** (licencia limpia) y
evaluar `mmgp` sólo si `accelerate` no alcanza para 12 GB de VRAM.

---

## 3. Arquitectura propuesta

```text
h3runtime/
├── __init__.py         API pública: load(), generate()
├── config.py           H3Config: dimensiones, rutas, versiones esperadas
├── checkpoint.py       lector safetensors + reintento de errores explícitos
├── quant.py            dequant INT8 tensorwise + des-rotación ConvRot
├── layout.py           permutación QKV (grouped -> interleaved)
├── model.py            definición del transformer H3 (blocks, norms, attn, mlp)
├── text_encoder.py     carga de Qwen3-VL-32B (sólo embeddings de H3)
├── vaes.py             video VAE + audio VAE
├── offload.py          política de VRAM/RAM (accelerate), unload/load
├── scheduler.py        pasos de denoising, schedule de sigmas
└── pipeline.py         orquestación FL2VA: prompt -> latentes -> VAE -> mp4
```

### Contrato de VRAM

```text
Fase interpretación:  Qwen3-VL-4B en GPU
    ↓ (unload)
Fase render:          H3 transformer en GPU con offload de bloques a RAM
    ↓ (unload)
Vuelta a reposo
```

Los dos modelos **nunca** conviven en VRAM. El runtime expone
`free_vram()` y verifica con `torch.cuda.mem_get_info()` antes y después.

### Etapas de implementación (verificable en cada paso)

| Etapa | Entregable | Cómo se verifica |
|---|---|---|
| E1 | `checkpoint.py` + `quant.py` + `layout.py` | cargar 1 bloque, comparar forma y rango de valores; no usa GPU |
| E2 | `model.py` | construir 50 bloques con `init_empty_weights`, contar parámetros |
| E3 | `offload.py` | cargar el modelo completo y medir VRAM/RAM pico sin generar |
| E4 | `pipeline.py` T2V mínimo | un clip de 5 s (124 frames) con seed fijo |
| E5 | reproducibilidad | mismo seed → mismo resultado, dos veces seguidas |

E1 no toca la GPU y es donde se concentra el riesgo real (dequant + layout). E4/E5
son el criterio de finalización del README.

---

## 4. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Des-rotación ConvRot mal implementada | test numérico: reconstruir y comparar contra el tensor BF16 equivalente del token_refiner (que no está cuantizado) |
| 12 GB de VRAM insuficientes | offload por bloques a RAM; medir en E3 antes de invertir en E4 |
| RAM de sistema (31 GB) insuficiente | medir en E3; si no alcanza, evaluar `mmgp` o más RAM |
| Deriva de dependencias | venv separado para el runtime, versiones pineadas, `requirements-runtime.txt` |
| Licencia H3 (territorial) | documentada en el README; revisar antes de uso comercial |

---

## 5. Qué NO se hace

- No se copia código de Wan2GP ni de ComfyUI. Se usan como **referencia de
  comportamiento**; la implementación es propia.
- No se incluye `diffusers` como orquestador (es justamente la dependencia que
  queremos evitar). Se usa sólo si una etapa puntual lo justifica.
- No se promete velocidad: la prioridad es **estabilidad y reproducibilidad**.
