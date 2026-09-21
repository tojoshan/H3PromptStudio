import json
import time

import gradio as gr

from scene import SceneSpec, GenerationConfig, VideoMode, resolve_dimensions
from compiler import compile_h3_prompt
from interpreter import MODEL_DIR, interpret_scene, is_qwen_loaded, model_files_present

ASPECT_LABELS = {
    "21:9": "21:9 — Cinemático",
    "16:9": "16:9 — Horizontal",
    "4:3": "4:3 — Clásico",
    "1:1": "1:1 — Cuadrado",
    "3:4": "3:4 — Retrato",
    "9:16": "9:16 — Vertical",
}
LABEL_TO_RATIO = {v: k for k, v in ASPECT_LABELS.items()}


def resolution_text(aspect_label, quality):
    ratio = LABEL_TO_RATIO[aspect_label]
    w, h = resolve_dimensions(ratio, quality)
    return f"**Salida H3:** {w} × {h} · 24 FPS"


def initial_status():
    if model_files_present():
        return f"### Estado\nModelo local detectado en `{MODEL_DIR}`. Se cargará en la GPU con la primera interpretación."
    return f"### Estado\nFalta el modelo local. Copiá Qwen3-VL-4B-Instruct dentro de `{MODEL_DIR}`."


def run(text, mode, duration, aspect_label, quality, style, progress=gr.Progress(track_tqdm=True)):
    ratio = LABEL_TO_RATIO[aspect_label]

    if mode != "t2v":
        yield (
            "### Modo todavía no conectado",
            json.dumps({"status": "pending", "mode": mode}, ensure_ascii=False, indent=2),
            "Falta conectar assets para este modo.",
        )
        return

    if not model_files_present():
        yield (
            f"### Falta el modelo\nCopiá Qwen3-VL-4B-Instruct dentro de `{MODEL_DIR}`.",
            "",
            "",
        )
        return

    started = time.time()
    first_load = not is_qwen_loaded()
    if first_load:
        progress(0.05, desc="Cargando Qwen3-VL en la GPU...")
        yield ("### Cargando Qwen3-VL…\nLa primera carga puede tardar unos segundos.", "", "")
    else:
        progress(0.2, desc="Interpretando escena...")
        yield ("### Qwen ya está cargado\nInterpretando la escena…", "", "")

    try:
        content = interpret_scene(text)
    except Exception as exc:
        yield (
            f"### Error\n`{type(exc).__name__}: {exc}`",
            "",
            "",
        )
        return

    progress(0.85, desc="Compilando SceneSpec...")

    if style and style.strip():
        content.style = style.strip()

    spec = SceneSpec(
        mode=VideoMode(mode),
        user_request=text,
        content=content,
        generation=GenerationConfig(
            duration_seconds=duration,
            aspect_ratio=ratio,
            quality=quality,
            seed=-1,
        ),
    )

    prompt = compile_h3_prompt(spec)
    elapsed = time.time() - started
    progress(1.0, desc="Listo")

    yield (
        f"### Listo\nInterpretación + compilación: **{elapsed:.1f} s**",
        spec.model_dump_json(indent=2),
        prompt,
    )


with gr.Blocks(title="H3 Prompt Studio") as demo:
    gr.Markdown(
        "# H3 Prompt Studio\n"
        "Texto libre → **Qwen3-VL-4B-Instruct local** → SceneSpec → prompt MiniMax H3"
    )

    status = gr.Markdown(initial_status())

    text = gr.Textbox(
        label="Qué querés generar",
        lines=5,
        placeholder="Una pequeña gata gris camina por un bosque y se encuentra con un colibrí.",
    )

    with gr.Row():
        mode = gr.Dropdown(
            choices=["t2v", "i2v", "first_last", "reference"],
            value="t2v",
            label="Modo",
        )
        duration = gr.Slider(4, 15, value=5, step=1, label="Duración (s)")

    with gr.Row():
        aspect = gr.Dropdown(
            choices=list(ASPECT_LABELS.values()),
            value=ASPECT_LABELS["16:9"],
            label="Relación de aspecto",
        )
        quality = gr.Dropdown(
            choices=["Preview", "Medium", "Native 768p"],
            value="Native 768p",
            label="Calidad",
        )

    resolution = gr.Markdown(resolution_text(ASPECT_LABELS["16:9"], "Native 768p"))
    aspect.change(resolution_text, [aspect, quality], resolution)
    quality.change(resolution_text, [aspect, quality], resolution)

    style = gr.Textbox(
        label="Estilo visual (opcional)",
        placeholder="Ej.: 3D animado estilo largometraje",
    )

    button = gr.Button("Interpretar con Qwen y compilar", variant="primary")
    spec_out = gr.Code(label="SceneSpec interpretado", language="json")
    prompt_out = gr.Textbox(label="Prompt compilado H3", lines=12)

    button.click(
        run,
        [text, mode, duration, aspect, quality, style],
        [status, spec_out, prompt_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
