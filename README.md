# H3 Prompt Studio

Aplicación local para Windows + NVIDIA RTX 5070 orientada a convertir instrucciones naturales en escenas estructuradas y, finalmente, generar video con MiniMax H3 sin depender de ComfyUI como runtime principal.

## Objetivo final

El objetivo completo de H3 Prompt Studio es permitir escribir una escena en lenguaje natural y producir video de forma reproducible con MiniMax H3, manteniendo control de:

- intención semántica;
- identidad de personajes;
- entorno;
- estilo visual;
- cámara;
- acción principal y secundaria;
- duración;
- resolución;
- seed;
- referencias visuales;
- continuidad entre planos;
- generación de secuencias de 30–60 segundos formadas por varios clips coherentes.

La aplicación debe soportar cuatro modos principales:

1. **Text to Video (T2V)**
2. **Image to Video (I2V)**
3. **First / Last Frame**
4. **Reference to Video**

---

## Arquitectura actual

```text
Texto libre del usuario
        ↓
Qwen3-VL-4B-Instruct local
        ↓
SceneSpec semántico canónico en inglés
        ↓
Normalización / validación en Python
        ↓
Compilador determinista
        ↓
Prompt estructurado para MiniMax H3
```

El prompt compilado sigue la sintaxis oficial de H3 (variante T2VA/FL2VA), con tres
campos obligatorios y en orden fijo:

```text
integrated_multimodal_description: [Shot 1] <escena + acción + cámara + diálogo>
overall_soundscape: <ambiencia física> | N/A
non_diegetic_music: <música sólo audible al público> | N/A
```

El compilador también expone `compile_scene_spec_prompt`, la variante legada por
secciones (`SUBJECT` / `ACTION` / `CAMERA` / …), conservada para inspección y debug.

La arquitectura actual separa deliberadamente dos responsabilidades:

- **Qwen interpreta intención.**
- **Python valida y compila.**

Qwen no decide resolución, FPS, duración, seed, modo de video ni rutas de assets.

---

## Decisión de idioma

El usuario puede escribir en español u otros idiomas.

Qwen debe entender el pedido original pero devolver todos los campos semánticos del `SceneSpec` en **inglés canónico, natural y conciso**.

No se usa una traducción literal separada.

El objetivo es que Qwen transforme directamente:

```text
Entrada en español
    ↓
Interpretación semántica
    ↓
SceneSpec en inglés
    ↓
Prompt H3 en inglés
```

Esto evita una segunda inferencia y reduce ambigüedades en campos como `action`, `secondary_action`, `environment` y `must_preserve`.

Los nombres propios nunca deben traducirse.

---

## Estado actual

Actualmente funciona:

- aplicación local en Windows;
- Gradio en `http://127.0.0.1:7860`;
- CUDA en NVIDIA RTX 5070;
- Qwen3-VL-4B-Instruct cargado localmente;
- caché del modelo entre interpretaciones;
- entrada natural en español;
- interpretación semántica mediante Qwen;
- `SceneSpec` validado mediante Pydantic;
- presets de resolución H3;
- 24 FPS como FPS nativo;
- duración configurable;
- aspect ratios preseteados;
- compilador determinista de prompt H3;
- normalización de `action` / `secondary_action`;
- reglas para evitar pronombres ambiguos;
- continuidad básica mediante `must_preserve`;
- restricciones mediante `must_avoid`;
- normalización en Python que repara salidas defectuosas de Qwen:
  - si `secondary_action` no contiene un verbo, se fusiona en `action` y se mueve a `must_preserve`;
  - los rasgos de `subject.appearance` e `identity_constraints` se propagan a `must_preserve`;
  - deduplicación de `must_preserve` y `must_avoid`;
- SceneSpec mostrado en la interfaz (JSON);
- prompt H3 compilado mostrado en la interfaz;
- estado dinámico del modelo en la interfaz (detecta si los archivos están presentes);
- resolución mostrada en vivo al cambiar aspect ratio / calidad;
- los modos i2v / first_last / reference están definidos en el esquema pero todavía desconectados en la interfaz (devuelven un aviso de "pendiente").

Rendimiento observado con el modelo ya cargado:

```text
Qwen model ready: ~0.0 s
Qwen generation: ~5–8 s
```

La primera carga del modelo tarda más, pero después queda en memoria.

---

## Instalación local

### Requisitos

- Windows 11
- NVIDIA RTX 5070 12 GB
- CUDA compatible
- Python 3.12 recomendado
- PyTorch con CUDA
- ~10 GB libres para Qwen3-VL-4B-Instruct

### Inicio rápido

1. Descomprimir la carpeta del proyecto.
2. Ejecutar:

```text
setup.bat
```

3. Copiar el repositorio completo de Qwen dentro de:

```text
models/Qwen3-VL-4B-Instruct/
```

Alternativamente:

```text
download_model.bat
```

4. Verificar CUDA:

```text
.venv\Scripts\python.exe check_gpu.py
```

5. Verificar carga del modelo:

```text
.venv\Scripts\python.exe test_model.py
```

6. Correr los tests (no requieren GPU ni el modelo):

```text
run_tests.bat
```

7. Ejecutar:

```text
run.bat
```

8. Abrir:

```text
http://127.0.0.1:7860
```

---

## Modelo Qwen

Modelo:

```text
Qwen/Qwen3-VL-4B-Instruct
```

El modelo debe estar completo dentro de:

```text
models/Qwen3-VL-4B-Instruct/
```

Archivos principales esperados:

```text
config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer.json
tokenizer_config.json
preprocessor_config.json
```

`interpreter.py` trabaja en modo local y no necesita consultar Hugging Face durante una interpretación.

---

## SceneSpec

`SceneSpec` es el contrato central de la aplicación.

Actualmente contempla:

```text
mode
user_request
content
generation
first_frame
last_frame
references
```

`content` contiene:

```text
subject
environment
action
secondary_action
style
lighting
camera
soundscape             # campo H3: overall_soundscape
non_diegetic_music     # campo H3: non_diegetic_music
dialogue               # lista de Dialogue (habla de H3)
must_preserve
must_avoid
```

`content.dialogue` es una lista de objetos `Dialogue` con:

```text
speaker      # S1, S2, ...
language     # English, Spanish, ...
text
offscreen    # bool: voz en off (la guía pide cerrar los labios)

`content.subject` es un objeto `Subject` con:

```text
name
appearance
identity_constraints   # lista de rasgos visuales de continuidad
```

`content.camera` es un objeto `Camera` con:

```text
shot
movement
angle
lens_feel
```

La cámara se emite como una cláusula en inglés natural dentro del shot, como pide
la guía de H3 (no hay tokens especiales de cámara).

`generation` contiene:

```text
duration_seconds   # 4.0–15.0 (rango real de H3)
fps                # fijo en 24
aspect_ratio
quality
width              # derivado, no editable
height             # derivado, no editable
num_frames         # derivado: se ajusta a la regla 17n+5
seed               # -1 = aleatorio
```

`width` y `height` se derivan de `aspect_ratio` + `quality`, y el validador
exige que sean múltiplos de 32 (requisito de H3). `num_frames` se calcula con
`frames_for_duration()` redondeando hacia arriba al siguiente valor válido de la
forma `17n+5`: por ejemplo 5 s → 124 frames, 15 s → 362 frames.

`references` es una lista de objetos `Reference` con:

```text
path
role               # character | style | environment | object | other
description
```

El esquema valida por modo mediante un `model_validator`:

```text
i2v         → requiere first_frame
first_last  → requiere first_frame y last_frame
reference   → requiere al menos una referencia
```

`width` y `height` se derivan automáticamente de `aspect_ratio` + `quality` dentro del validador de `GenerationConfig`; el usuario no los define a mano.

---

## Reglas actuales del intérprete

El intérprete:

- no debe inventar elementos;
- debe conservar los hechos explícitos;
- debe eliminar pronombres ambiguos cuando puede nombrar la entidad;
- debe mantener acciones relacionales dentro de `action`;
- debe usar `secondary_action` solamente para una segunda acción real;
- no debe poner descriptores nominales en `secondary_action`;
- debe preservar color, edad, raza, tamaño, pelaje y otros rasgos visuales útiles;
- debe conservar adverbios relevantes como `slowly`, `quickly`, `gently`;
- debe devolver la semántica en inglés;
- no controla parámetros técnicos de generación.

Ejemplo:

```text
Usuario:
Una perra muy peluda de pelo blanco con ondas juega con
una gata gris de 3 meses y raza nebelung.

Salida conceptual:

SUBJECT: very fluffy dog with long wavy white fur
ACTION: plays with a 3-month-old gray Nebelung kitten
PRESERVE: long wavy white fur; very fluffy; gray 3-month-old Nebelung kitten
```

Además, la normalización en Python aplica la reparación final descrita arriba:
si Qwen devuelve un fragmento sin verbo en `secondary_action`, éste se fusiona
en `action` y se mueve a `must_preserve`; y todo rasgo estable de
`subject.appearance` / `subject.identity_constraints` se propaga a
`must_preserve` antes de compilar.

### API interna de `interpreter.py`

```text
model_files_present()                → bool: ¿están los archivos del modelo?
is_qwen_loaded()                     → bool: ¿hay componentes en caché (VRAM)?
interpret_scene(text)                → SceneContent interpretado y normalizado
get_qwen_components()                → (processor, model) con lru_cache(1)
_has_verb(text)                      → bool: ¿la frase contiene una acción?
```

La detección de acciones (`_has_verb`) combina una **whitelist** de verbos
frecuentes del dominio con un **fallback morfológico** por sufijos
(`-s`, `-es`, `-ing`, `-ed`) sobre las primeras palabras, con una lista negra de
falsos positivos (`fluffy`, `kitten`, `shining`, etc.). Esto permite separar
acciones reales de descriptores aunque el verbo no esté en la whitelist.

`interpret_scene` devuelve **sólo** `SceneContent`. El ensamblado del
`SceneSpec` completo (con `mode`, `user_request` y `generation`) lo hace `app.py`.

---

## Presets de generación

Los parámetros técnicos no dependen de Qwen.

### FPS
```text
24 FPS
```

El aumento a 30/60 FPS debe ser una etapa posterior de interpolación.

### Duración
H3 acepta clips de **4 a 15 segundos**. La cantidad de frames no es libre:
H3 decodifica longitudes de la forma **17n+5**, así que el compilador redondea
hacia arriba la duración pedida al siguiente valor válido.

```text
5 s  → 124 frames
15 s → 362 frames
```

Videos más largos que 15 s se obtienen encadenando ventanas con solapamiento
(sliding windows), no en una sola generación.

### Relaciones de aspecto

Actualmente contempladas:

```text
21:9
16:9
4:3
1:1
3:4
9:16
```

### Calidad
```text
Preview
Medium
Native 768p
```

Python resuelve automáticamente `width × height` a partir de `aspect_ratio` + `quality`.

Tabla real de presets (`scene.py`):

```text
21:9   Preview 960×416    Medium 1152×512    Native 768p 1760×768
16:9   Preview 864×480    Medium 1056×608    Native 768p 1344×768
4:3    Preview 640×480    Medium 800×608     Native 768p 1024×768
1:1    Preview 480×480    Medium 608×608     Native 768p 768×768
3:4    Preview 480×640    Medium 608×800     Native 768p 768×1024
9:16   Preview 480×864    Medium 608×1056    Native 768p 768×1344
```

La interfaz no muestra el preset crudo: resuelve las dimensiones y las muestra
directamente sobre el formulario.

---

## Runtime MiniMax H3 — realidad y decisiones
Esta sección documenta los hechos verificados sobre H3 que condicionan la
arquitectura. Son el resultado de la investigación de la Fase 2.

### Modelo
```text
MiniMax-H3 = modelo omni-modal denso de ~33B (H3-Omni-Transformer)
Text encoder = Qwen3-VL-32B completo
Salida = video 768p + audio estéreo 32 kHz en un solo paso
FPS = 24 fijo
Duración = 4–15 s por generación
Resolución = múltiplos de 32, borde corto 768 px
CFG = destilado → sin guidance_scale ni negative_prompt
Seed = reproducible vía torch.Generator
```

### Checkpoints
Dos familias, cada una un repo HF-style con `model_index.json`:

```text
FL2VA  → Text-to-Audio-Video + First/Last-Frame   (cubre T2V e I2V)
Ref2VA → Reference-to-Audio-Video
```

El repo completo pesa ~144 GB en BF16. El **footprint mínimo viable** es la
versión **pruned INT8** (checkpoint ~20B con curvas AdaLN comprimidas) más los
VAEs, el text encoder cuantizado y el latent upscaler (~42 GB en disco).

### Por qué no se depende de ComfyUI ni de Wan2GP
Ambos son funcionales pero **mutan bajo los pies del usuario**: una actualización
puede romper el modelo, y una misma sesión puede fallar en la segunda generación
al dejar de reconocer el text encoder como compatible. Este proyecto prioriza un
runtime **propio, aislado y con versiones congeladas**.

Se toma `X:\Wan2GP\models\minimax_h3` **sólo como referencia técnica**
(no como dependencia): de allí provienen los detalles de la regla `17n+5`, el
uso de offload a RAM para caber en 12 GB de VRAM, y las versiones que
funcionan en esta máquina.

### Versiones de referencia (a congelar)

```text
torch        2.10.0+cu130
torchaudio   2.10.0+cu130
torchvision  0.25.0+cu130
diffusers    0.36.0
transformers 4.54.0
accelerate   1.15.0
mmgp         3.8.0    # offload de memoria GPU/RAM
```

`mmgp` es la pieza que permite correr un modelo de este tamaño en 12 GB de VRAM
mediante offload dinámico a RAM. Requiere RAM de sistema abundante (~64 GB
recomendado para el modo offload).

### Licencia
H3 se distribuye bajo la **MiniMax H3 Community License Agreement** (no es
Apache/MIT) y su despliegue local tiene **restricciones territoriales**. Revisar
antes de cualquier uso comercial.

---

# Hoja de ruta pendiente

## Fase 1 — Prompt interpreter

### Completado

- [x] Qwen local
- [x] entrada en español
- [x] SceneSpec estructurado
- [x] salida semántica canónica en inglés
- [x] reglas de no alucinación
- [x] separación acción / acción secundaria
- [x] `must_preserve`
- [x] `must_avoid`
- [x] compilador determinista al formato real de H3
- [x] presets de resolución (múltiplos de 32)
- [x] cálculo de `num_frames` según la regla 17n+5
- [x] campos de audio H3 (`soundscape`, `non_diegetic_music`, `dialogue`)
- [x] caché del modelo

### Completado (tests)

- [x] tests deterministas del compilador (`test_compiler.py`);
- [x] tests de la normalización del intérprete (`test_interpreter.py`);
- [x] test de regresión multi-personaje (perra peluda + gata Nebelung);
- [x] mejora de la detección de acciones/verbos (whitelist + fallback morfológico);
- [x] runner `run_tests.bat` (no requiere GPU).

### Pendiente
- [ ] tests ejecutables contra Qwen real para pronombres y referencias cruzadas;
- [ ] mejorar extracción de cámara;
- [ ] mejorar extracción de iluminación;
- [ ] detectar entidades secundarias como objetos estructurados;
- [ ] definir esquema explícito de múltiples sujetos.

---

## Fase 2 — Runtime MiniMax H3

Objetivo: ejecutar H3 directamente desde Python sin depender de ComfyUI ni de
Wan2GP, con un runtime propio y versiones congeladas.

Hecho:

- [x] investigar la arquitectura y el formato de H3 (ver "Runtime MiniMax H3 —
      realidad y decisiones");
- [x] identificar las versiones que funcionan en esta máquina (a congelar);
- [x] confirmar la sintaxis real del prompt H3 y ajustar el compilador;
- [x] modelar los límites reales (4–15 s, frames 17n+5, resolución múltiplo de 32).

Pendiente:

- [ ] decidir el camino de implementación (diffusers ModularPipeline vs. runtime
      propio vendorizado);
- [ ] aislar H3 de Qwen para controlar VRAM;
- [ ] implementar carga/descarga controlada entre CPU/GPU (offload a RAM);
- [ ] primer render T2V;
- [ ] manejo de seed;
- [ ] progress callbacks;
- [ ] cancelación de generación;
- [ ] limpieza de VRAM después de errores;
- [ ] recuperación automática ante OOM;
- [ ] log de parámetros de cada render.

Criterio de finalización:

```text
Mismo prompt + mismos parámetros + mismo seed
→ generación reproducible y estable
```

---

## Fase 3 — Text to Video

Pendiente:

- [ ] conectar el prompt compilado al runtime H3;
- [ ] botón `Generate`;
- [ ] preview del video;
- [ ] guardar output;
- [ ] nombre automático de archivos;
- [ ] metadata JSON junto al video;
- [ ] historial de generaciones;
- [ ] botón regenerar;
- [ ] seed aleatorio / seed fijo;
- [ ] presets de calidad.

---

## Fase 4 — Image to Video

Pendiente:

- [ ] uploader de imagen inicial;
- [ ] preview;
- [ ] validación de formato;
- [ ] detección automática de aspect ratio;
- [ ] adaptación a resoluciones H3;
- [ ] evitar deformación/crop accidental;
- [ ] enviar first frame al runtime;
- [ ] permitir describir movimiento sin volver a describir la identidad visual;
- [ ] usar Qwen-VL para analizar referencia solamente cuando sea necesario.

La imagen debe tener prioridad sobre descripciones textuales contradictorias de identidad.

---

## Fase 5 — First / Last Frame
El esquema ya lo soporta: `SceneSpec.first_frame` / `last_frame` y la validación
por modo exigen ambos cuando `mode=first_last`. Falta la capa de interfaz y de
runtime.

Pendiente:

- [ ] uploader de first frame;
- [ ] uploader de last frame;
- [ ] validación de aspect ratios;
- [ ] preview de ambos frames;
- [ ] compilación de movimiento entre estados;
- [ ] generación H3 First/Last;
- [ ] detectar cambios imposibles/inconsistentes;
- [ ] reutilizar last frame como first frame del siguiente plano.

Esto será fundamental para continuidad multi-shot.

---

## Fase 6 — Reference to Video
El esquema ya lo soporta: `SceneSpec.references` es una lista de objetos
`Reference` (`path`, `role`, `description`) con roles
`character` / `style` / `environment` / `object` / `other`, y la validación por
modo exige al menos una referencia cuando `mode=reference`.

Pendiente:

- [ ] referencias múltiples;
- [ ] roles para cada referencia;
- [ ] `character`;
- [ ] `environment`;
- [ ] `style`;
- [ ] `object`;
- [ ] preview de referencias;
- [ ] nombres/IDs de personajes;
- [ ] selección de qué referencia debe preservar cada entidad;
- [ ] adapter/ruta correcta para Reference-to-Video H3;
- [ ] límites de cantidad/resolución;
- [ ] validación previa.

Objetivo:

```text
Metis → reference_metis
Ruby → reference_ruby
Bosque → reference_forest
Style → reference_style
```

---

## Fase 7 — Asset manager

Crear una capa persistente para manejar assets.

Pendiente:

```text
assets/
├── characters/
├── environments/
├── styles/
├── objects/
├── first_frames/
└── last_frames/
```

Cada asset debería guardar metadata:

```text
id
name
type
path
description
identity traits
created_at
```

La idea es reutilizar personajes sin volver a cargarlos/configurarlos cada vez.

---

## Fase 8 — Proyectos y escenas

Agregar persistencia.

Propuesta:

```text
projects/
└── proyecto_x/
    ├── project.json
    ├── scenes/
    ├── assets/
    ├── renders/
    └── metadata/
```

Cada escena debería guardar:

```text
SceneSpec
prompt compilado
seed
modelo
parámetros
referencias
video generado
first frame
last frame
fecha
```

---

## Fase 9 — Continuidad multi-shot

Objetivo principal del proyecto.

Permitir definir una secuencia de escenas:

```text
Shot 01
Shot 02
Shot 03
Shot 04
```

para construir:

```text
15 s × 4 clips = 60 s
```

Pendiente:

- [ ] timeline;
- [ ] lista ordenable de shots;
- [ ] continuidad de personaje;
- [ ] continuidad de vestuario/apariencia;
- [ ] continuidad de entorno;
- [ ] continuidad de cámara;
- [ ] reutilizar last frame;
- [ ] transferir `must_preserve`;
- [ ] llevar referencias entre planos;
- [ ] detectar contradicciones;
- [ ] guardar estado narrativo mínimo;
- [ ] botón `Generate next shot`.

El siguiente plano no debe reinterpretar libremente todo desde cero.

Debe heredar explícitamente el estado visual relevante del plano anterior.

---

## Fase 10 — Guión a secuencia

Agregar una capa por encima de SceneSpec.

Entrada futura:

```text
Metis camina por el bosque.
Ve un pájaro.
Corre hacia él.
El pájaro vuela hacia un árbol.
Metis se detiene debajo.
```

El sistema debería convertirlo en varios shots:

```text
Shot 01
Shot 02
Shot 03
...
```

y luego generar cada uno respetando continuidad.

Pendiente:

- [ ] parser de guión;
- [ ] segmentación en planos;
- [ ] duración estimada;
- [ ] continuidad entre shots;
- [ ] edición manual antes de renderizar;
- [ ] generación por lotes.

---

## Fase 11 — Cámara

Agregar controles explícitos y presets:

```text
shot size
camera angle
camera movement
lens feel
focus
depth of field
```

Presets posibles:

```text
close-up
medium shot
full shot
wide shot
tracking
dolly in
dolly out
pan
tilt
orbit
static
handheld
```

Qwen puede interpretar lenguaje natural, pero Python debe mantener valores normalizados.

---

## Fase 12 — Postprocesado

No debe formar parte del render H3 inicial.

Pipeline futuro:

```text
H3 24 FPS
↓
interpolación
↓
48 / 60 FPS
↓
opcional upscale
↓
encode final
```

Pendiente:

- [ ] interpolación;
- [ ] upscale opcional;
- [ ] encoding H.264/H.265/AV1;
- [ ] presets de exportación;
- [ ] audio futuro;
- [ ] concatenación automática de shots.

---

## Fase 13 — Manejo de VRAM

Importante para RTX 5070 12 GB.

Pendiente:

- [ ] visualizar VRAM en UI;
- [ ] unload de Qwen antes de H3 si es necesario;
- [ ] reload de Qwen bajo demanda;
- [ ] offload CPU;
- [ ] garbage collection controlado;
- [ ] presets de memoria;
- [ ] detectar OOM;
- [ ] recuperación automática.

Arquitectura deseada:

```text
Interpretación
Qwen → GPU
↓
SceneSpec
↓
Qwen unload/offload
↓
H3 → GPU
↓
video
```

si ambos modelos no pueden convivir cómodamente en VRAM.

---

## Fase 14 — UX

Pendiente:

- [ ] estado de carga de Qwen;
- [ ] estado de carga de H3;
- [ ] progreso de render;
- [ ] tiempo transcurrido;
- [ ] estimación aproximada;
- [ ] mensajes de error útiles;
- [ ] cancelar;
- [ ] retry;
- [ ] copiar prompt;
- [ ] editar SceneSpec;
- [ ] guardar proyecto;
- [ ] cargar proyecto;
- [ ] preview de assets;
- [ ] historial.

---

## Fase 15 — Tests y estabilidad
El objetivo es evitar la inestabilidad experimentada con workflows de ComfyUI.

Implementado (sin GPU, vía `run_tests.bat`):

- [x] tests de presets y resolución (`test_scene.py`);
- [x] tests del compiler (`test_compiler.py`);
- [x] tests de la normalización del interpreter (`test_interpreter.py`).

Pendiente:

- [ ] `requirements.lock` o versiones exactas;
- [ ] tests de SceneSpec (más allá de presets);
- [ ] tests ejecutables contra Qwen real (prompts completos);
- [ ] tests de carga de modelos;
- [ ] smoke test H3;
- [ ] test T2V;
- [ ] test I2V;
- [ ] test First/Last;
- [ ] test Reference;
- [ ] test generación consecutiva;
- [ ] test unload/reload;
- [ ] test recuperación después de error.

Regla:

**No actualizar dependencias automáticamente en una instalación de producción.**

---

## Fase 16 — Empaquetado

Objetivo final local:

```text
H3PromptStudio/
├── app/
├── models/
├── assets/
├── projects/
├── outputs/
├── logs/
├── .venv/
└── H3PromptStudio.bat
```

Idealmente:

1. doble click;
2. comprueba GPU;
3. levanta backend;
4. abre navegador;
5. carga proyecto;
6. genera.

Sin necesidad de abrir terminal o editar Python.

---

# Orden recomendado de implementación

El orden de trabajo propuesto es:

```text
1. Terminar intérprete
2. Conectar H3 T2V
3. Hacer T2V estable y repetible
4. I2V
5. First / Last Frame
6. Reference to Video
7. Asset manager
8. Proyectos / persistencia
9. Continuidad multi-shot
10. Guión → shots
11. Postprocesado
12. Empaquetado
```

No conviene avanzar a continuidad de 30–60 segundos hasta que un solo shot sea totalmente repetible y estable.

---

## Estructura actual de archivos

```text
H3PromptStudio/
├── app.py
├── interpreter.py
├── scene.py
├── compiler.py
├── test_scene.py
├── test_compiler.py
├── test_interpreter.py
├── test_model.py
├── check_gpu.py
├── download_model.py
├── requirements.txt
├── setup.bat
├── run.bat
├── run_tests.bat
├── download_model.bat
├── README.md
├── .gitignore
└── models/
    └── Qwen3-VL-4B-Instruct/
```

`models/Qwen3-VL-4B-Instruct/` está ignorado en `.gitignore` salvo un archivo
centinela `PUT_MODEL_HERE.txt`; el modelo se descarga localmente y no se versiona.

---

## Responsabilidad de cada archivo

### `app.py`

Interfaz Gradio y configuración técnica de la escena. Expone el textbox de la
petición, el selector de modo, duración, aspect ratio, calidad, el estilo visual
opcional, y muestra el SceneSpec y el prompt compilado. Los modos
i2v / first_last / reference quedan definidos pero devuelven un aviso de
"pendiente" hasta conectar los assets.

### `interpreter.py`

Carga Qwen3-VL-4B-Instruct local (caché por `lru_cache`) y convierte las
instrucciones naturales en un `SceneContent` estructurado en inglés. Fuerza
`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` y valida que existan los archivos del
modelo. Incluye la normalización en Python (`_normalize_content`) que repara
salidas defectuosas de Qwen antes de devolver el `SceneContent`.

### `scene.py`

Contrato Pydantic y validación de SceneSpec: `VideoMode`, `AspectRatio`,
`QualityPreset`, la tabla `RESOLUTION_PRESETS`, `resolve_dimensions`,
`Subject`, `Camera`, `Reference`, `SceneContent`, `GenerationConfig` y
`SceneSpec` con su `model_validator` por modo.

### `compiler.py`

Convierte SceneSpec en el **prompt estructurado real de H3**, emitiendo los tres
campos obligatorios en orden (`integrated_multimodal_description`,
`overall_soundscape`, `non_diegetic_music`), con `[Shot 1]`, cámara en inglés
natural, diálogo `(S1) says: <d>[English] …</d>` y `N/A` en los campos de audio
vacíos. Conserva además `compile_scene_spec_prompt`, la variante legada por
secciones (`SUBJECT` / `CAMERA` / …) para inspección y debug.

### `test_scene.py`

Verifica `resolve_dimensions` para varios presets y el cálculo automático de
`width` / `height` / `fps` en `GenerationConfig`.

### `test_compiler.py`

Tests deterministas del compilador de prompt H3: secciones y su orden, omisión
de secciones vacías, normalización de espacios, cámara parcial, determinismo y
validación de modo `reference`. No requiere GPU ni torch.

### `test_interpreter.py`

Tests de la capa determinista del intérprete: detección de verbos, limpieza y
deduplicación, fusión de `secondary_action` sin verbo y propagación de rasgos de
identidad a `must_preserve`, incluido un caso de regresión multi-personaje.
Importa `interpreter` (torch/transformers) pero no usa la GPU ni el modelo.

### `run_tests.bat`

Ejecuta los tres suites de tests sin GPU y devuelve error si alguno falla.

### `test_model.py`

Verifica que Qwen pueda cargar correctamente.

### `check_gpu.py`

Verifica CUDA/GPU: versión de Python y PyTorch, disponibilidad de CUDA, nombre
de la GPU y VRAM total.

### `setup.bat`

Prepara el entorno Python local.

### `run.bat`

Levanta la aplicación.

### `download_model.py` / `download_model.bat`

Descarga el repositorio completo `Qwen/Qwen3-VL-4B-Instruct` mediante
`snapshot_download` dentro de `models/Qwen3-VL-4B-Instruct/`.

---

# Principios de diseño

1. **ComfyUI no es una dependencia del producto final.**
2. **Qwen interpreta; Python controla.**
3. **Los parámetros técnicos nunca los inventa el LLM.**
4. **El usuario puede escribir en español.**
5. **La representación interna semántica se normaliza a inglés.**
6. **Cada generación debe poder reproducirse.**
7. **Las versiones del runtime deben quedar congeladas.**
8. **Los modelos se almacenan localmente.**
9. **Los errores deben ser explícitos, no silenciosos.**
10. **La continuidad debe heredarse de forma estructurada, no mediante reinterpretación libre de la imagen anterior.**
