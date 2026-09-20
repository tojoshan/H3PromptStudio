from scene import GenerationConfig, resolve_dimensions

expected = {
    ("16:9", "Native 768p"): (1344, 768),
    ("9:16", "Native 768p"): (768, 1344),
    ("1:1", "Native 768p"): (768, 768),
}
for key, value in expected.items():
    assert resolve_dimensions(*key) == value

cfg = GenerationConfig(aspect_ratio="16:9", quality="Native 768p")
assert cfg.width == 1344 and cfg.height == 768 and cfg.fps == 24
print("OK — presets H3 y cálculo automático funcionando.")
