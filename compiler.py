"""Compilador determinista de SceneSpec a prompt MiniMax H3.

Formato real de H3 (guia oficial de prompts, variante T2VA/FL2VA):

    integrated_multimodal_description: [Shot 1] ...
    overall_soundscape: ...
    non_diegetic_music: ...

Los tres campos son OBLIGATORIOS y en ese orden exacto. Cuando no hay contenido
se escribe el token literal `N/A` (solo para los campos de audio).

Tambien expone `compile_scene_spec_prompt`, el compilador legado por secciones
(SUBJECT/ACTION/CAMERA...), conservado para inspeccion y debug.
"""

from scene import SceneSpec, Dialogue

# Token literal que H3 exige cuando un campo de audio esta vacio.
NA = "N/A"


def _clean(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _join_clauses(parts: list[str]) -> str:
    # Une fragmentos en una sola oracion terminada en punto.
    clauses = [_clean(p).rstrip(".") for p in parts if _clean(p)]
    if not clauses:
        return ""
    return ", ".join(clauses) + "."


def _camera_clause(scene: SceneSpec) -> str:
    cam = scene.content.camera
    parts = [_clean(cam.movement), _clean(cam.shot), _clean(cam.angle), _clean(cam.lens_feel)]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    # Descripcion natural en ingles, como pide la guia (no tokens especiales).
    return ", ".join(parts)


def _dialogue_snippet(dialogue: Dialogue) -> str:
    # Construye la sintaxis de dialogo de H3: (S1) says: <d>[English] ...</d>.
    speaker = _clean(dialogue.speaker) or "S1"
    language = _clean(dialogue.language) or "English"
    text = _clean(dialogue.text)
    if not text:
        return ""
    verb = "says in an off-screen voiceover:" if dialogue.offscreen else "says:"
    snippet = f"({speaker}) {verb} <d>[{language}] {text}</d>"
    if dialogue.offscreen:
        snippet += " while the lips remain completely closed"
    return snippet


def _build_shot_description(scene: SceneSpec) -> str:
    # Contenido del campo integrated_multimodal_description (un solo shot).
    c = scene.content
    pieces: list[str] = []

    lead = _join_clauses([c.style, c.lighting])
    if lead:
        pieces.append(lead)

    subject = ", ".join(x for x in [_clean(c.subject.name), _clean(c.subject.appearance)] if x)
    body = _join_clauses([subject, _clean(c.environment), _clean(c.action), _clean(c.secondary_action)])
    if body:
        pieces.append(body)

    camera = _camera_clause(scene)
    if camera:
        pieces.append(f"The camera {camera}.")

    dialogue = " ".join(s for s in (_dialogue_snippet(d) for d in c.dialogue) if s)
    if dialogue:
        pieces.append(dialogue)

    preserve = [_clean(x) for x in (c.subject.identity_constraints + c.must_preserve)]
    preserve = [x for x in preserve if x]
    if preserve:
        pieces.append("Maintain consistent appearance throughout: " + "; ".join(preserve) + ".")

    if c.must_avoid:
        avoid = "; ".join(_clean(x) for x in c.must_avoid if _clean(x))
        pieces.append(f"Avoid showing: {avoid}.")

    return " ".join(p for p in pieces if p)


def compile_h3_prompt(scene: SceneSpec) -> str:
    # Compila un SceneSpec al prompt estructurado de MiniMax H3.
    description = _build_shot_description(scene)
    shot_body = description or "A static scene with no specified action."

    # [Shot 1] sin timestamp, segun la guia.
    description_field = f"[Shot 1] {shot_body}"

    soundscape = _clean(scene.content.soundscape) or NA
    music = _clean(scene.content.non_diegetic_music) or NA

    return (
        f"integrated_multimodal_description: {description_field}\n"
        f"overall_soundscape: {soundscape}\n"
        f"non_diegetic_music: {music}"
    )


def compile_scene_spec_prompt(scene: SceneSpec) -> str:
    # Compilador legado por secciones. Conservado para inspeccion y debug.
    c = scene.content
    sections = []
    if c.subject.name or c.subject.appearance:
        subject = ", ".join(x for x in [_clean(c.subject.name), _clean(c.subject.appearance)] if x)
        sections.append(f"SUBJECT: {subject}")
    if c.environment:
        sections.append(f"ENVIRONMENT: {_clean(c.environment)}")
    if c.action:
        sections.append(f"ACTION: {_clean(c.action)}")
    if c.secondary_action:
        sections.append(f"SECONDARY ACTION: {_clean(c.secondary_action)}")
    camera = [_clean(c.camera.shot), _clean(c.camera.angle), _clean(c.camera.movement), _clean(c.camera.lens_feel)]
    camera = [x for x in camera if x]
    if camera:
        sections.append("CAMERA: " + ", ".join(camera))
    if c.style:
        sections.append(f"VISUAL STYLE: {_clean(c.style)}")
    if c.lighting:
        sections.append(f"LIGHTING: {_clean(c.lighting)}")
    preserve = c.subject.identity_constraints + c.must_preserve
    if preserve:
        sections.append("PRESERVE: " + "; ".join(_clean(x) for x in preserve if _clean(x)))
    if c.must_avoid:
        sections.append("AVOID: " + "; ".join(_clean(x) for x in c.must_avoid if _clean(x)))
    return "\n".join(sections)
