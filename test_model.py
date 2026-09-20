from time import perf_counter
from interpreter import MODEL_DIR, get_qwen_pipe, model_files_present

print("Modelo:", MODEL_DIR)
print("Archivos completos:", model_files_present())
if not model_files_present():
    raise SystemExit("Falta copiar el modelo dentro de models/Qwen3-VL-4B-Instruct")

t0 = perf_counter()
print("Cargando Qwen...")
get_qwen_pipe()
print(f"Qwen cargado en {perf_counter() - t0:.1f} s")
