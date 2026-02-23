"""Environment verification script for Ambient Clinical Scribe."""
import patch_torch  # Must be first — patches torch.int1-int7 for Windows
import sys

print(f"Python: {sys.version}")
print()

packages = [
    ("torch", "torch", "__version__"),
    ("transformers", "transformers", "__version__"),
    ("unsloth", "unsloth", None),
    ("sentence-transformers", "sentence_transformers", "__version__"),
    ("peft", "peft", "__version__"),
    ("trl", "trl", "__version__"),
    ("chromadb", "chromadb", "__version__"),
    ("llama-index", "llama_index", None),
    ("openai", "openai", "__version__"),
    ("datasets", "datasets", "__version__"),
    ("accelerate", "accelerate", "__version__"),
    ("bitsandbytes", "bitsandbytes", "__version__"),
]

failed = []
for display_name, import_name, version_attr in packages:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, version_attr, "OK") if version_attr else "OK"
        print(f"  {display_name:25s} {ver}")
    except Exception as e:
        print(f"  {display_name:25s} FAILED - {e}")
        failed.append(display_name)

# CUDA check
print()
try:
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA:                    {torch.version.cuda}")
        print(f"  GPU:                     {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:                    {vram:.1f} GB")
    else:
        print("  CUDA:                    NOT AVAILABLE")
        failed.append("CUDA")
except Exception as e:
    print(f"  CUDA check:              FAILED - {e}")
    failed.append("CUDA")

print()
if failed:
    print(f"WARNING: {len(failed)} package(s) failed: {', '.join(failed)}")
else:
    print("All packages verified successfully!")
