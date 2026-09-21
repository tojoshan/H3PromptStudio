"""Tests deterministas del compilador de prompt H3.

No requieren GPU, torch ni el modelo de Qwen: solo validan que un SceneSpec
produce siempre el mismo prompt y con las secciones correctas.

Ejecutar:
    .venv\\Scripts\\python.exe test_compiler.py
"""

from scene import (
    Camera,
    GenerationConfig,
    Reference,
    SceneContent,
    SceneSpec,
    Subject,
    VideoMode,
)
from compiler import compile_h3_prompt


def _spec(content: SceneContent, **kwargs) -> SceneSpec:
    return SceneSpec(
        user_request="test",
        content=content,
        generation=GenerationConfig(),
        **kwargs,
    )


def test_full_prompt_sections():
    content = SceneContent(
        subject=Subject(
            name="Metis",
            appearance="gray 3-month-old Nebelung kitten",
            identity_constraints=["gray fur", "green eyes"],
        ),
        environment="misty pine forest at dawn",
        action="walks slowly along a mossy path",
        secondary_action="sniffs the air",
        camera=Camera(shot="medium shot", angle="eye level", movement="slow dolly in"),
        style="3D animated feature film",
        lighting="soft diffused morning light",
        must_preserve=["long whiskers"],
        must_avoid=["no people", "no text overlay"],
    )

    prompt = compile_h3_prompt(_spec(content))

    assert prompt.startswith("SUBJECT: "), prompt
    for expected in [
        "SUBJECT: Metis, gray 3-month-old Nebelung kitten",
        "ENVIRONMENT: misty pine forest at dawn",
        "ACTION: walks slowly along a mossy path",
        "SECONDARY ACTION: sniffs the air",
        "CAMERA: medium shot, eye level, slow dolly in",
        "VISUAL STYLE: 3D animated feature film",
        "LIGHTING: soft diffused morning light",
        "AVOID: no people; no text overlay",
    ]:
        assert expected in prompt, f"Falta: {expected}\n---\n{prompt}"

    # PRESERVE combina subject.identity_constraints + must_preserve, en ese orden.
    assert "PRESERVE: gray fur; green eyes; long whiskers" in prompt, prompt

    # Orden fijo de secciones.
    order = [
        "SUBJECT:",
        "ENVIRONMENT:",
        "ACTION:",
        "SECONDARY ACTION:",
        "CAMERA:",
        "VISUAL STYLE:",
        "LIGHTING:",
        "PRESERVE:",
        "AVOID:",
    ]
    positions = [prompt.index(s) for s in order]
    assert positions == sorted(positions), f"Orden incorrecto:\n{prompt}"


def test_minimal_prompt_omits_empty_sections():
    content = SceneContent(subject=Subject(name="Metis"), action="sits")
    prompt = compile_h3_prompt(_spec(content))

    assert "SUBJECT: Metis" in prompt
    assert "ACTION: sits" in prompt
    # Las secciones vacías no deben aparecer.
    for absent in ["ENVIRONMENT:", "SECONDARY ACTION:", "CAMERA:", "PRESERVE:", "AVOID:"]:
        assert absent not in prompt, f"No deberia aparecer {absent}:\n{prompt}"


def test_whitespace_is_normalized():
    content = SceneContent(
        subject=Subject(appearance="  very   fluffy   white   fur  "),
        environment="  forest\nat   night ",
        action="walks",
    )
    prompt = compile_h3_prompt(_spec(content))

    assert "SUBJECT: very fluffy white fur" in prompt, prompt
    assert "ENVIRONMENT: forest at night" in prompt, prompt
    # El valor no debe contener espacios dobles ni saltos de linea internos.
    env_value = prompt.split("ENVIRONMENT: ", 1)[1].split("\n", 1)[0]
    assert "  " not in env_value and "\n" not in env_value, repr(env_value)


def test_camera_partial_fields():
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="runs",
        camera=Camera(shot="wide shot", lens_feel="35mm"),
    )
    prompt = compile_h3_prompt(_spec(content))

    # Solo se incluyen los campos no vacios, unidos por coma.
    assert "CAMERA: wide shot, 35mm" in prompt, prompt


def test_prompt_is_deterministic():
    content = SceneContent(
        subject=Subject(name="Metis", appearance="gray kitten"),
        environment="forest",
        action="walks",
    )
    spec = _spec(content)
    first = compile_h3_prompt(spec)
    second = compile_h3_prompt(spec)
    assert first == second, "El compilador no es determinista."


def test_reference_mode_requires_references():
    from pydantic import ValidationError

    content = SceneContent(subject=Subject(name="Metis"), action="walks")
    try:
        SceneSpec(
            mode=VideoMode.REFERENCE,
            user_request="test",
            content=content,
            references=[Reference(path="assets/metis.png", role="character")],
        )
    except ValidationError as exc:  # pragma: no cover - no deberia fallar
        raise AssertionError(f"reference con referencias no deberia fallar: {exc}")

    try:
        SceneSpec(mode=VideoMode.REFERENCE, user_request="test", content=content)
    except ValidationError:
        pass
    else:
        raise AssertionError("reference sin referencias deberia fallar")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} tests de compilador OK.")
