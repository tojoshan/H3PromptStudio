from scene import SceneSpec

def _clean(value: str) -> str:
    return " ".join((value or "").split()).strip()

def compile_h3_prompt(scene: SceneSpec) -> str:
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
