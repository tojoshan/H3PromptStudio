**PRD**

# **Orquestador local de video generativo** **con MiniMax H3**

*Aplicación local orientada a proyectos, continuidad audiovisual y orquestación conversacional*

&nbsp;

Estado: borrador para auditoría

Versión: 0.1

Fecha: 20/09/2026

# **1\. Visión del producto**

Construir una aplicación local de escritorio para creación de video generativo orientada a proyectos, donde el usuario pueda trabajar desde una interfaz conversacional con personajes, referencias visuales y de audio, escenas, planos y continuidad temporal.

La aplicación no debe exponer al usuario la complejidad técnica de ComfyUI, WanGP/Wan2GP, loaders, LoRAs, modelos, samplers o parámetros internos. La experiencia buscada es la de una herramienta audiovisual integrada: el usuario describe lo que quiere realizar, adjunta referencias cuando corresponde y el sistema planifica, prepara y ejecuta las generaciones necesarias.

La arquitectura propuesta separa tres responsabilidades:

INTERFAZ / ORQUESTADOR  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
PROJECT CORE  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
MOTORES DE GENERACIÓN

El objetivo final es combinar:

* la facilidad conversacional de Deepy/WanGP;  
* la gestión de referencias, personajes, shots y timelines inspirada en Continuity;  
* la eficiencia de inferencia y administración de VRAM de WanGP;  
* la generación audiovisual nativa de MiniMax H3;  
* un modelo multilingüe dedicado a convertir instrucciones humanas en prompts formales específicos para H3.

Deepy se toma inicialmente como referencia funcional de UX y orquestación, pero no es un requisito que el producto final dependa de Deepy. Puede utilizarse, extenderse o reemplazarse por un orquestador propio.

# **2\. Problema**

Actualmente las capacidades necesarias están distribuidas en distintas herramientas. WanGP ofrece una interfaz relativamente sencilla, optimizaciones fuertes para hardware limitado, soporte H3 y una capa conversacional. Continuity, por su parte, aporta una abstracción más cercana al trabajo audiovisual: cast, referencias, prompting, timelines, múltiples shots y herramientas auxiliares.

Ninguna de las dos, por sí sola, representa exactamente la experiencia buscada: trabajar con un proyecto audiovisual persistente y controlar todo principalmente mediante una conversación.

El sistema propuesto agrega una capa superior que coordina ambas ideas.

# **3\. Principio central**

El chat/orquestador es el punto de entrada principal del producto.

El usuario debería poder escribir algo como: “Usá a Meche como referencia. Quiero una escena en un bar. Primero un plano general, después un plano medio y finalmente un primer plano. Conservá exactamente el vestuario y la identidad. En el segundo plano dice ‘Todavía no llegó’. Sin música. Mantené continuidad de luz y posición entre los tres planos.”

El sistema deberá convertir esa petición en una estructura de proyecto:

Proyecto  
└── Escena: Bar  
&nbsp;&nbsp;&nbsp;&nbsp;├── Shot 01  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── referencia: Meche  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── plano: general  
&nbsp;&nbsp;&nbsp;&nbsp;│   └── generación  
&nbsp;&nbsp;&nbsp;&nbsp;├── Shot 02  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── referencia: Meche  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── continuidad: Shot 01  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── plano: medio  
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── diálogo  
&nbsp;&nbsp;&nbsp;&nbsp;│   └── generación  
&nbsp;&nbsp;&nbsp;&nbsp;└── Shot 03  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── referencia: Meche  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── continuidad: Shot 02  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── plano: primer plano  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── generación

El usuario no debería tener que construir manualmente esta estructura.

# **4\. Arquitectura propuesta**

┌───────────────────────────────────────────────┐  
│                 DESKTOP APP                   │  
│   Chat / Deepy-like UI     Preview            │  
│   Cast / References        Timeline           │  
│   Scenes / Shots           Project Browser    │  
└──────────────────────┬────────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
┌───────────────────────────────────────────────┐  
│                 ORCHESTRATOR                  │  
│  Natural Language → Intent                    │  
│  Project awareness                            │  
│  Tool selection                               │  
│  Shot planning                                │  
│  Dependency / continuity planning             │  
└──────────────────────┬────────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;┌────────┴─────────┐  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼                  ▼  
┌──────────────────────┐  ┌─────────────────────┐  
│ MULTILINGUAL PROMPT  │  │    PROJECT CORE     │  
│      COMPILER        │  │ Cast / Locations    │  
│ Qwen VL / equivalent │  │ References / Scenes │  
│ Human → H3 Prompt    │  │ Shots / Generations│  
└──────────┬───────────┘  │ Continuity graph    │  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│              └──────────┬──────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───────────┬─────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
┌───────────────────────────────────────────────┐  
│              GENERATION ADAPTER               │  
│                    WanGP                      │  
│ Model selection / settings / references       │  
│ VRAM management / rendering / output          │  
└──────────────────────┬────────────────────────┘  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
┌───────────────────────────────────────────────┐  
│                MINIMAX H3                     │  
│ FL2VA / T2VA / I2VA / L2VA / Ref2VA          │  
└───────────────────────────────────────────────┘

# **5\. Tecnología base**

## **Backend / Project Core**

Primera implementación: Python \+ FastAPI.

Razones:

* WanGP y el ecosistema de generación ya son Python.  
* Simplifica reutilizar loaders, modelos y utilidades existentes.  
* Permite prototipar rápidamente el adapter.  
* Adecuado para comunicación local mediante HTTP/WebSocket.  
* Evita introducir FFI Rust/Python antes de validar el producto.

Rust no queda descartado. Puede incorporarse posteriormente en componentes donde aporte una ventaja concreta, como administración de procesos, filesystem, media indexing o empaquetado.

## **Aplicación de escritorio**

Proyección final: Tauri 2 \+ frontend web. El proceso Python permanece como servicio local.

Tauri  
&nbsp;&nbsp;↓  
Frontend  
&nbsp;&nbsp;↓  
FastAPI localhost  
&nbsp;&nbsp;↓  
Project Core  
&nbsp;&nbsp;↓  
WanGP

Esto permite una UI mucho más flexible que Gradio para timeline, drag & drop, navegador de assets, shot cards, preview, chat, cast, controles visuales y comparación de generaciones.

Para validar el concepto inicialmente puede utilizarse Gradio \+ FastAPI. Gradio no se considera necesariamente la UI definitiva.

# **6\. Motor generativo principal**

## **MiniMax H3**

Será el motor audiovisual principal. La implementación debe contemplar generación de video con audio y varios modos de condicionamiento según la familia H3 utilizada.

## **H3 FL2VA**

Debe soportar:

* T2VA — Text to Video \+ Audio.  
* I2VA — Image to Video \+ Audio.  
* L2VA — Last Frame to Video \+ Audio.  
* FL2VA — First \+ Last Frame.

First frame  
\+  
Prompt  
\+  
Last frame  
&nbsp;&nbsp;&nbsp;↓  
Video que conecta ambos frames

## **H3 Ref2VA**

Modo destinado a reutilización y consistencia. Debe utilizarse para:

* identidad de personajes;  
* vestuario;  
* objetos;  
* escenarios;  
* estilo;  
* movimiento de referencia;  
* voz;  
* audio;  
* continuidad visual basada en referencias.

El Project Core deberá poder representar múltiples referencias visuales, de video y de audio sin acoplarse a un límite específico del modelo.

# **7\. Modelo multilingüe / H3 Prompt Compiler**

Debe existir un modelo separado del motor de video. Su función no será generar video, sino convertir instrucciones humanas en prompts formales para H3.

instrucción humana  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
interpretación audiovisual  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
prompt H3 formal

Ejemplo de entrada:

“Meche mira a cámara, parece un poco cansada. Se acomoda el pelo y dice: 'No sé si fue una buena idea venir'. Quiero un plano medio sin cortes.”

Salida estructurada esperada:

subject\_definitions:  
...

summary:  
...

retention\_analysis:  
...

detailed\_description:  
\[Shot 1\] ...

overall\_soundscape:  
...

non\_diegetic\_music:  
N/A

El compilador debe preservar el diálogo textual en su idioma original y convertir el resto de la instrucción a una forma estructurada y optimizada para H3.

# **8\. Modelo candidato para el Prompt Compiler**

Familia base: Qwen3-VL.

Se requiere porque el compilador eventualmente deberá entender:

* español y otros idiomas;  
* imágenes de personajes;  
* referencias visuales;  
* intención de cámara;  
* composición;  
* continuidad.

## **Variante inicial candidata**

Qwen3-VL-4B, por tamaño y relación entre capacidad y consumo.

## **Variante con menor tendencia a rechazos**

Como opción experimental puede utilizarse una variante community/abliterated compatible, por ejemplo Qwen3-VL-4B Heretic, siempre detrás de una interfaz intercambiable.

PromptCompiler interface  
├── Qwen3-VL 4B  
├── Qwen3-VL Heretic  
├── modelo futuro  
└── proveedor externo opcional

# **9\. Project Core**

El Project Core es la pieza central que conceptualmente inspira Continuity. No debe depender directamente de H3 y debe almacenar conceptos audiovisuales abstractos.

Project  
├── metadata  
├── settings  
├── cast  
├── locations  
├── assets  
├── scenes  
├── sequences  
├── generations  
└── exports

## **Cast**

Un personaje debe existir independientemente de cualquier generación.

Character: Meche

Reference images  
├── front.jpg  
├── profile.jpg  
├── full\_body.jpg  
└── closeup.jpg

Reference video  
├── walking.mp4  
└── talking.mp4

Voice  
└── voice\_reference.wav

Properties  
├── identity  
├── default wardrobe  
├── physical description  
└── prompt aliases

# **10\. Escenas y planos**

## **Scene**

Una escena agrupa contexto:

Scene: Bar nocturno

Location  
Characters  
Lighting  
Environment  
Continuity state  
Shots

## **Shot**

Shot  
├── duration  
├── prompt  
├── camera  
├── characters  
├── references  
├── dialogue  
├── sound  
├── continuity  
├── generation mode  
├── generation settings  
├── outputs  
└── selected take

# **11\. Sistema de continuidad**

La continuidad no debe considerarse simplemente como 'usar el último frame'. Debe existir un continuity state que describa el estado relevante del plano y de los personajes.

Shot 04 State

Character Meche  
├── wardrobe: red shirt \+ jeans  
├── position: right side of table  
├── pose: seated  
├── hairstyle: unchanged  
└── emotional state: concerned

Scene  
├── table position  
├── glass position  
├── lighting  
└── camera axis

El orquestador utilizará este estado para decidir entre:

* Ref2VA;  
* first-frame;  
* first+last frame;  
* video reference;  
* frame extraído del plano anterior;  
* múltiples referencias.

# **12\. WanGP como Render Engine**

WanGP no debe convertirse en el Project Core. Su función es ejecutar trabajos de generación.

GenerationJob  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
WanGP Adapter  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
WanGP  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
MiniMax H3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
GenerationResult

El adapter deberá traducir el modelo abstracto interno hacia parámetros que WanGP entienda.

{  
&nbsp;&nbsp;"engine": "wan\_gp",  
&nbsp;&nbsp;"model": "h3\_ref2va",  
&nbsp;&nbsp;"prompt": "...",  
&nbsp;&nbsp;"references": \[\],  
&nbsp;&nbsp;"duration": 5,  
&nbsp;&nbsp;"resolution": "...",  
&nbsp;&nbsp;"sampler": "...",  
&nbsp;&nbsp;"steps": 8  
}

La implementación concreta dependerá de la interfaz estable disponible en WanGP y deberá desacoplarse del resto del sistema.

# **13\. Deepy**

Deepy se considera inicialmente como referencia de UX, referencia de tool-calling y potencial frontend/orquestador reutilizable. El Project Core no dependerá de Deepy.

IOrchestrator  
├── DeepyAdapter  
├── LocalLLMOrchestrator  
└── FutureAgent

# **14\. Interfaz final prevista**

┌──────────────────────────────────────────────────────────┐  
│ Proyecto: Video Turismo                         Render ▷ │  
├─────────────────┬────────────────────────────────────────┤  
│ CHAT            │             PREVIEW                    │  
│                 │                                        │  
│ \> Usá a Meche   │                                        │  
│   en el bar...  │                                        │  
├─────────────────┼────────────────────────────────────────┤  
│ CAST            │ TIMELINE                               │  
│ Meche           │ \[Shot 1\]\[Shot 2\]\[Shot 3\]              │  
│ Juan            │                                        │  
│ Bar             │                                        │  
├─────────────────┴────────────────────────────────────────┤  
│ Inspector / references / generation settings             │  
└──────────────────────────────────────────────────────────┘

# **15\. Principio de interacción**

El sistema debe soportar dos niveles.

## **Conversacional**

Ejemplo: “Cambiá el plano 3 por un primerísimo primer plano. Que siga diciendo la misma frase pero ahora más bajo.”

## **Manual**

El usuario puede abrir un shot y modificar directamente:

* prompt;  
* references;  
* seed;  
* duration;  
* modelo;  
* sampler;  
* steps;  
* audio;  
* cámara.

El chat no reemplaza los controles. Los orquesta.

# **16\. Historial y no destrucción**

Todas las generaciones serán inmutables.

Shot 03  
├── Take 01  
├── Take 02  
├── Take 03 ★ selected  
└── Take 04

También deberá almacenarse:

* modelo y versión;  
* seed;  
* LoRAs;  
* parámetros;  
* prompt humano original;  
* prompt H3 compilado;  
* referencias utilizadas.

Esto permitirá reproducibilidad y auditoría.

# **17\. MVP 0 — Proof of Concept**

Crear proyecto  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Crear personaje  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Agregar una referencia  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Escribir solicitud en español  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Prompt Compiler  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
H3 prompt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
WanGP  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
MP4  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Asociar resultado al proyecto

Este es el primer milestone y debe validar el loop completo.

# **18\. MVP 1**

* múltiples personajes;  
* varias referencias;  
* FL2VA;  
* Ref2VA;  
* audio de referencia;  
* selección de takes;  
* Project Core persistente;  
* preview;  
* cola de generación;  
* historial.

# **19\. MVP 2 — Continuity**

* escenas;  
* shots;  
* timeline;  
* frame extraction;  
* continuidad Shot N → Shot N+1;  
* propagation de wardrobe;  
* reference inheritance;  
* camera continuity;  
* regenerate shot;  
* replace shot;  
* extend shot.

# **20\. MVP 3 — Orquestador avanzado**

El usuario podrá describir una escena completa y el agente deberá interpretar, crear scene y shots, asignar referencias, decidir el modo H3 adecuado, generar prompts, ejecutar trabajos y presentar la timeline.

interpreta  
&nbsp;&nbsp;&nbsp;↓  
crea scene  
&nbsp;&nbsp;&nbsp;↓  
crea shots  
&nbsp;&nbsp;&nbsp;↓  
asigna referencias  
&nbsp;&nbsp;&nbsp;↓  
decide FL2VA / Ref2VA  
&nbsp;&nbsp;&nbsp;↓  
genera prompts H3  
&nbsp;&nbsp;&nbsp;↓  
ejecuta trabajos  
&nbsp;&nbsp;&nbsp;↓  
presenta timeline

Antes de generar todo, podrá mostrar el plan para edición manual.

# **21\. MVP 4 — Edición conversacional**

* Usá el take 2 del plano 3\.  
* Extendé el plano 4 dos segundos.  
* Me gusta la actuación pero no la cámara.  
* Conservá este movimiento de cámara y regenerá el personaje.  
* Usá el audio del take anterior.  
* Hacé que los planos 5 a 8 mantengan este vestuario.

Estas instrucciones modifican entidades del Project Core.

# **22\. Optimización y aceleradores**

Los aceleradores deben quedar detrás del Render Adapter y no formar parte de la semántica del proyecto.

* modelos pruned;  
* INT8;  
* FP8;  
* NVFP4;  
* Turbo LoRA;  
* PDD;  
* First Block Cache;  
* Spectrum;  
* VSA;  
* VDN;  
* futuras optimizaciones H3.

Esto permite que un proyecto siga siendo válido aunque cambie completamente el pipeline de inferencia.

# **23\. Gestión de hardware**

El sistema deberá detectar GPU, VRAM, RAM y capabilities, y seleccionar perfiles predefinidos:

Quality  
Balanced  
Fast  
Low VRAM  
Custom

El objetivo es evitar que el usuario tenga que comprender cada combinación interna.

# **24\. Prompt Compiler: separación crítica**

Deben conservarse siempre tres representaciones diferentes:

1\. User Intent

"La chica mira por la ventana..."

2\. Structured Intent

subject \= Meche  
action \= look through window  
dialogue \= ...  
camera \= medium shot

3\. H3 Prompt

subject\_definitions:  
...

Si mañana MiniMax H4 reemplaza H3, el proyecto no debería quedar atado al formato actual.

User Intent  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Structured Intent  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
H4 Compiler

# **25\. Esquema inicial de datos**

Entidades iniciales:

Project  
Character  
Location  
Asset  
Scene  
Shot  
ShotReference  
Generation  
GenerationAsset  
ContinuityState  
PromptSource  
PromptCompiled  
RenderJob  
EngineProfile

Persistencia inicial: SQLite. Assets en filesystem.

project/  
├── project.db  
├── characters/  
├── references/  
├── scenes/  
├── generations/  
├── audio/  
├── cache/  
└── exports/

# **26\. Comunicación interna**

Tauri Frontend  
&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;│ HTTP/WebSocket  
&nbsp;&nbsp;&nbsp;&nbsp;▼  
FastAPI  
&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;├── Project Service  
&nbsp;&nbsp;&nbsp;&nbsp;├── Orchestrator  
&nbsp;&nbsp;&nbsp;&nbsp;├── Prompt Compiler  
&nbsp;&nbsp;&nbsp;&nbsp;├── Media Service  
&nbsp;&nbsp;&nbsp;&nbsp;└── Render Queue  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;WanGP

WebSocket se utilizará para progreso, logs, previews, job state y errores.

# **27\. Requisitos no funcionales**

* funcionar localmente;  
* no requerir cloud para la generación principal;  
* poder recuperar un proyecto después de un crash;  
* no destruir generaciones anteriores;  
* registrar exactamente cómo se produjo cada render;  
* permitir cancelar trabajos;  
* soportar cola de trabajos;  
* evitar cargar simultáneamente modelos incompatibles con la VRAM disponible;  
* mantener modelos y orquestadores intercambiables.

# **28\. Fuera del alcance inicial**

* editor NLE completo;  
* composición tipo After Effects;  
* entrenamiento de modelos;  
* generación distribuida multi-GPU;  
* colaboración multiusuario;  
* cloud rendering;  
* marketplace de modelos;  
* edición de audio avanzada.

La timeline inicial administra shots y clips, no pretende reemplazar Premiere o Resolve.

# **29\. Riesgos principales**

## **Dependencia de WanGP**

WanGP evoluciona rápidamente. Mitigación:

Project Core  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
IRenderEngine  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
WanGPEngine

Nunca llamar WanGP directamente desde la UI.

## **Cambios en H3**

IPromptCompiler  
├── H3Compiler  
└── futuro H4Compiler

## **Compatibilidad entre aceleradores**

Turbo, PDD, VSA, caches y quantizations no deben suponerse compatibles entre sí. Cada EngineProfile deberá declarar sus combinaciones válidas.

## **Continuidad generativa**

No puede garantizarse continuidad perfecta únicamente mediante prompts. La solución deberá combinar:

* referencias;  
* keyframes;  
* previous-frame;  
* video references;  
* continuity state;  
* regeneración selectiva.

# **30\. Criterio de éxito del primer prototipo**

1. Abrir la aplicación.  
2. Crear un proyecto.  
3. Crear un personaje.  
4. Adjuntar una imagen.  
5. Escribir una instrucción en español.  
6. Ver el prompt H3 generado.  
7. Enviar el trabajo a WanGP.  
8. Recibir video \+ audio.  
9. Reproducirlo dentro del proyecto.  
10. Cerrar y volver a abrir la aplicación sin perder el estado.

Si eso funciona, la arquitectura base está validada.

# **31\. Proyección final**

El producto final no será simplemente una interfaz para WanGP ni Continuity fuera de ComfyUI.

**La proyección es un entorno local de dirección y producción audiovisual generativa donde un orquestador conversacional manipula un proyecto estructurado y utiliza distintos motores de IA como herramientas.**

H3 será inicialmente el principal motor de video, WanGP el principal backend de inferencia y Qwen el intérprete multimodal/prompt compiler. El diseño, sin embargo, debe permitir incorporar otros motores sin redefinir el proyecto.

Orchestrator  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
Project Core  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── H3  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── LTX  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── modelo de imagen  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── TTS  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── upscaler  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── frame interpolation  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── futuros motores

**La pieza que permanece en el tiempo no es el modelo. Es el proyecto.**

**Y la interacción principal no es el workflow de nodos. Es la intención del usuario.**