import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("CUDA PyTorch:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch no detecta CUDA.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
