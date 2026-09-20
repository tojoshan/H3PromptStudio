from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
out = root / "models" / "Qwen3-VL-4B-Instruct"
out.mkdir(parents=True, exist_ok=True)

print("Descargando Qwen/Qwen3-VL-4B-Instruct en:", out)
snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    local_dir=str(out),
)
print("Listo.")
