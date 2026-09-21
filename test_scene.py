from scene import GenerationConfig, frames_for_duration, resolve_dimensions

expected = {
    ("16:9", "Native 768p"): (1344, 768),
    ("9:16", "Native 768p"): (768, 1344),
    ("1:1", "Native 768p"): (768, 768),
}
for key, value in expected.items():
    assert resolve_dimensions(*key) == value
# Todas las resoluciones deben ser multiplos de 32 (requisito de H3).
for ratio, qualities in __import__("scene").RESOLUTION_PRESETS.items():
    for quality, (w, h) in qualities.items():
        assert w % 32 == 0 and h % 32 == 0, f"{ratio}/{quality} = {w}x{h} no es multiplo de 32"

cfg = GenerationConfig(aspect_ratio="16:9", quality="Native 768p")
assert cfg.width == 1344 and cfg.height == 768 and cfg.fps == 24
assert cfg.num_frames == 124, cfg.num_frames  # 17*7+5 para 5 s
# Regla de frames 17n+5.
for frames in (frames_for_duration(4.0), frames_for_duration(5.0), frames_for_duration(15.0)):
    assert (frames - 5) % 17 == 0, frames
assert frames_for_duration(5.0) == 124
assert frames_for_duration(15.0) == 362
# Duracion fuera de rango debe fallar.
for bad in (3.0, 16.0):
    try:
        GenerationConfig(duration_seconds=bad)
    except Exception:
        pass
    else:
        raise AssertionError(f"duracion {bad}s deberia ser invalida")

print("OK - presets H3, frames 17n+5 y limites automaticos funcionando.")
