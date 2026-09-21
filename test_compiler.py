"""Tests deterministas del compilador de prompt MiniMax H3.

Verifican el formato real de H3 (T2VA/FL2VA):

    integrated_multimodal_description: [Shot 1] ...
    overall_soundscape: ...
    non_diegetic_music: ...

No requieren GPU, torch ni el modelo de Qwen.

Ejecutar:
    .venv\\Scripts\\python.exe test_compiler.py
"""

from scene import (
    Camera,
    Dialogue,
    GenerationConfig,
    Reference,
    SceneContent,
    SceneSpec,
    Subject,
    VideoMode,
)
from compiler import compile_h3_prompt, compile_scene_spec_prompt


def _spec(content: SceneContent, **kwargs) -> SceneSpec:
    return SceneSpec(
        user_request="test",
        content=content,
        generation=GenerationConfig(),
        **kwargs,
    )


def test_three_required_fields_in_order():
    content = SceneContent(subject=Subject(name="Metis"), action="walks")
    prompt = compile_h3_prompt(_spec(content))
    lines = prompt.split("\n")

    assert lines[0].startswith("integrated_multimodal_description: "), lines[0]
    assert lines[1].startswith("overall_soundscape: "), lines[1]
    assert lines[2].startswith("non_diegetic_music: "), lines[2]
    assert len(lines) == 3, prompt


def test_audio_fields_default_to_na():
    content = SceneContent(subject=Subject(name="Metis"), action="walks")
    prompt = compile_h3_prompt(_spec(content))
    assert "overall_soundscape: N/A" in prompt, prompt
    assert "non_diegetic_music: N/A" in prompt, prompt


def test_audio_fields_are_used_when_present():
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks",
        soundscape="Birdsong and rustling leaves",
        non_diegetic_music="A soft piano motif at a slow tempo.",
    )
    prompt = compile_h3_prompt(_spec(content))
    assert "overall_soundscape: Birdsong and rustling leaves" in prompt, prompt
    assert "non_diegetic_music: A soft piano motif at a slow tempo." in prompt, prompt
    assert "N/A" not in prompt, prompt


def test_shot_1_has_no_timestamp():
    content = SceneContent(subject=Subject(name="Metis"), action="walks")
    prompt = compile_h3_prompt(_spec(content))
    assert "[Shot 1] " in prompt, prompt
    # El primer shot no lleva "At MM:SS.mmm".
    assert "At 00:" not in prompt, prompt


def test_camera_is_natural_english_clause():
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks",
        camera=Camera(movement="pushes in with small amplitude at slow speed", shot="medium shot"),
    )
    prompt = compile_h3_prompt(_spec(content))
    assert "The camera pushes in with small amplitude at slow speed, medium shot." in prompt, prompt


def test_dialogue_syntax():
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="turns to the camera",
        dialogue=[Dialogue(speaker="S1", language="English", text="Follow me.")],
    )
    prompt = compile_h3_prompt(_spec(content))
    assert "(S1) says: <d>[English] Follow me.</d>" in prompt, prompt


def test_offscreen_dialogue_syntax():
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="looks away",
        dialogue=[Dialogue(speaker="S2", language="Spanish", text="Hola.", offscreen=True)],
    )
    prompt = compile_h3_prompt(_spec(content))
    assert "(S2) says in an off-screen voiceover: <d>[Spanish] Hola.</d>" in prompt, prompt
    assert "while the lips remain completely closed" in prompt, prompt


def test_preserve_and_avoid_are_included():
    content = SceneContent(
        subject=Subject(name="Metis", identity_constraints=["gray fur"]),
        action="walks",
        must_preserve=["long whiskers"],
        must_avoid=["no people", "no text overlay"],
    )
    prompt = compile_h3_prompt(_spec(content))
    assert "Maintain consistent appearance throughout: gray fur; long whiskers." in prompt, prompt
    assert "Avoid showing: no people; no text overlay." in prompt, prompt


def test_empty_scene_does_not_crash():
    content = SceneContent()
    prompt = compile_h3_prompt(_spec(content))
    assert prompt.startswith("integrated_multimodal_description: [Shot 1] "), prompt
    assert "overall_soundscape: N/A" in prompt, prompt


def test_prompt_is_deterministic():
    content = SceneContent(
        subject=Subject(name="Metis", appearance="gray kitten"),
        environment="forest",
        action="walks",
    )
    spec = _spec(content)
    assert compile_h3_prompt(spec) == compile_h3_prompt(spec)


def test_whitespace_is_normalized():
    content = SceneContent(subject=Subject(appearance="  very   fluffy   fur  "), action="  walks ")
    prompt = compile_h3_prompt(_spec(content))
    assert "  " not in prompt.split("\n")[0].split("[Shot 1] ")[1], prompt


def test_legacy_compiler_still_works():
    content = SceneContent(
        subject=Subject(name="Metis", appearance="gray kitten"),
        action="walks",
        camera=Camera(shot="medium shot"),
    )
    legacy = compile_scene_spec_prompt(_spec(content))
    assert "SUBJECT: Metis, gray kitten" in legacy, legacy
    assert "ACTION: walks" in legacy, legacy
    assert "CAMERA: medium shot" in legacy, legacy


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
    except ValidationError as exc:  # pragma: no cover
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
