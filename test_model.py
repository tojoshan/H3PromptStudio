from time import perf_counter
from interpreter import MODEL_DIR, get_qwen_components, model_files_present
print("Modelo:", MODEL_DIR)
print("Archivos completos:", model_files_present())
if not model_files_present():
    raise SystemExit("Falta copiar el modelo dentro de models/Qwen3-VL-4B-Instruct")

t0 = perf_counter()
print("Cargando Qwen...")
processor, model = get_qwen_components()
print(f"Qwen cargado en {perf_counter() - t0:.1f} s")
print("Dispositivo:", next(model.parameters()).device)
