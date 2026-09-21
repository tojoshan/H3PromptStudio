"""Tests del interprete: normalizacion en Python y reglas de continuidad.

Estos tests NO necesitan GPU ni el modelo de Qwen cargado. Cubren la capa
determinista que repara la salida de Qwen antes de compilar:

  - separacion accion / accion secundaria;
  - descartar fragmentos sin verbo en secondary_action;
  - propagacion de rasgos de identidad a must_preserve;
  - deduplicacion de must_preserve / must_avoid.

Requieren que `torch` y `transformers` puedan importarse (lo hace `interpreter`
a nivel modulo). Si no estan instalados, se saltean con un aviso.

Ejecutar:
    .venv\\Scripts\\python.exe test_interpreter.py
"""

from scene import Camera, SceneContent, Subject

try:
    import interpreter as it
    INTERPRETER_IMPORTABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depende del entorno
    INTERPRETER_IMPORTABLE = False
    IMPORT_ERROR = exc


def _require_interpreter():
    if not INTERPRETER_IMPORTABLE:
        print(f"SKIP - no se pudo importar interpreter ({IMPORT_ERROR!r}).")
        return False
    return True


# ---------------------------------------------------------------------------
# _has_verb: deteccion de verbos para separar accion de descriptor
# ---------------------------------------------------------------------------

def test_has_verb_detects_actions():
    if not _require_interpreter():
        return
    for phrase in [
        "walks along the path",
        "runs quickly",
        "plays with a kitten",
        "the cat is sleeping",
        "she jumps over the fence",
    ]:
        assert it._has_verb(phrase), f"Deberia detectar verbo en: {phrase!r}"


def test_has_verb_rejects_descriptors():
    if not _require_interpreter():
        return
    for phrase in [
        "long wavy white fur",
        "gray 3-month-old Nebelung kitten",
        "",
        "   ",
        "very fluffy",
    ]:
        assert not it._has_verb(phrase), f"NO deberia detectar verbo en: {phrase!r}"


# ---------------------------------------------------------------------------
# _clean / deduplicacion
# ---------------------------------------------------------------------------

def test_clean_collapses_whitespace():
    if not _require_interpreter():
        return
    assert it._clean("  a   b \n c  ") == "a b c"
    assert it._clean("trailing,; ") == "trailing"
    assert it._clean(None) == ""


def test_dedupe_preserve_keeps_order_and_uniqueness():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(appearance="white fur"),
        action="walks",
        must_preserve=["white fur", "white fur", "green eyes", "  green eyes  "],
    )
    out = it._normalize_content(content)
    assert out.must_preserve == ["white fur", "green eyes"], out.must_preserve


# ---------------------------------------------------------------------------
# Normalizacion de secondary_action
# ---------------------------------------------------------------------------

def test_secondary_action_with_verb_is_kept():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks slowly along the path",
        secondary_action="sniffs the air",
    )
    out = it._normalize_content(content)
    assert out.action == "walks slowly along the path", out.action
    assert out.secondary_action == "sniffs the air", out.secondary_action


def test_secondary_action_descriptor_is_merged_and_preserved():
    if not _require_interpreter():
        return
    # Un fragmento sin verbo NO debe quedar en secondary_action.
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks along the path",
        secondary_action="long wavy white fur",
    )
    out = it._normalize_content(content)

    assert out.secondary_action == "", "El descriptor no debe quedar como accion."
    assert "long wavy white fur" in out.action, out.action
    assert "long wavy white fur" in out.must_preserve, out.must_preserve


def test_secondary_action_descriptor_not_duplicated_in_action():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks slowly, long wavy white fur",
        secondary_action="long wavy white fur",
    )
    out = it._normalize_content(content)
    # No se debe repetir el descriptor al final si ya esta presente.
    assert out.action.count("long wavy white fur") == 1, out.action
    assert out.secondary_action == ""
    assert "long wavy white fur" in out.must_preserve


# ---------------------------------------------------------------------------
# Continuidad: rasgos de identidad -> must_preserve
# ---------------------------------------------------------------------------

def test_identity_traits_flow_into_must_preserve():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(
            name="Metis",
            appearance="gray 3-month-old Nebelung kitten",
            identity_constraints=["gray fur", "green eyes"],
        ),
        action="walks",
        must_preserve=["long whiskers"],
    )
    out = it._normalize_content(content)

    for trait in ["gray 3-month-old Nebelung kitten", "gray fur", "green eyes", "long whiskers"]:
        assert trait in out.must_preserve, f"Falta {trait!r} en {out.must_preserve}"


def test_must_avoid_is_deduplicated():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="walks",
        must_avoid=["no people", "no people", "  no text overlay "],
    )
    out = it._normalize_content(content)
    assert out.must_avoid == ["no people", "no text overlay"], out.must_avoid


def test_action_and_secondary_action_are_cleaned():
    if not _require_interpreter():
        return
    content = SceneContent(
        subject=Subject(name="Metis"),
        action="  walks   along   the path  ",
        secondary_action="  sniffs  the  air ",
    )
    out = it._normalize_content(content)
    assert out.action == "walks along the path", out.action
    assert out.secondary_action == "sniffs the air", out.secondary_action


# ---------------------------------------------------------------------------
# Regresion: caso multi-personaje del README (perra + gata Nebelung)
# ---------------------------------------------------------------------------

def test_multi_entity_regression_case():
    if not _require_interpreter():
        return
    # Simula una salida plausible de Qwen para el ejemplo del README.
    content = SceneContent(
        subject=Subject(
            name="",
            appearance="very fluffy dog with long wavy white fur",
            identity_constraints=["long wavy white fur", "very fluffy"],
        ),
        environment="forest clearing",
        action="plays with a 3-month-old gray Nebelung kitten",
        must_preserve=["gray 3-month-old Nebelung kitten"],
    )
    out = it._normalize_content(content)

    assert out.action == "plays with a 3-month-old gray Nebelung kitten", out.action
    assert "gray 3-month-old Nebelung kitten" in out.must_preserve
    assert "long wavy white fur" in out.must_preserve
    # Sin duplicados tras la propagacion.
    assert len(out.must_preserve) == len(set(out.must_preserve)), out.must_preserve


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} tests de interprete OK.")
