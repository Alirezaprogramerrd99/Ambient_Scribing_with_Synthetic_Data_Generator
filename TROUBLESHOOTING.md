# Troubleshooting Guide — Ambient Clinical Scribe Environment

**Last updated:** 23 February 2026

---

## Quick Diagnosis

Run `python verify_env.py` and look for FAILED packages. Then find your error below.

---

## Error: `module 'torch' has no attribute 'int1'`

**Affects:** unsloth, bitsandbytes, peft, sentence-transformers, transformers  
**Cause:** bitsandbytes >= 0.45 and torchao expect PyTorch 2.6+ which introduced `torch.int1`–`torch.int7` dtypes. You're on PyTorch 2.5.1 which doesn't have them.  
**Fix:** Ensure `import patch_torch` is the FIRST import in every script. The patch adds mock `int1`–`int7` attributes.

```python
# CORRECT — every .py file must start with this
import patch_torch
from unsloth import FastLanguageModel
```

If you're seeing this in `verify_env.py`, make sure `verify_env.py` has `import patch_torch` as its first non-comment line.

---

## Error: `cannot import name 'PreTrainedModel' from 'transformers'`

**Affects:** peft, sentence-transformers, unsloth  
**Cause:** Two possible reasons:
1. `transformers >= 4.52` with `torch < 2.6` (tensor_parallel module needs newer torch)
2. `bitsandbytes` crashes during import, which prevents `transformers.modeling_utils` from loading, which breaks `PreTrainedModel`

**Fix:** This is almost always caused by the `torch.int1` error above. Fix that first (import patch_torch) and this resolves.

If patch_torch is already imported and this still happens: your transformers and torch versions are incompatible. Pin `transformers<4.52` or upgrade torch to 2.6+. The proven working combination is `torch==2.5.1+cu121` + `transformers==4.57.6` + `patch_torch.py`.

---

## Error: `cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler'`

**Affects:** unsloth, peft, sentence-transformers (anything importing transformers)  
**Cause:** `triton-windows` package doesn't include `AttrsDescriptor` symbol that `torch >= 2.6` expects. The `triton-windows` fork is incomplete compared to official Linux Triton wheels.  
**Fix:** Do NOT use `torch >= 2.6` with `triton-windows`. Use `torch==2.5.1+cu121` instead. The CUDA 12.1 index has torch up to 2.5.1, which works with `triton-windows==3.6.0.post25`.

If you absolutely need torch 2.6+, you'd need CUDA 12.4+ and would need to build Triton from source on Windows — not worth it for this project.

---

## Error: `module 'torch._inductor' has no attribute 'config'`

**Affects:** unsloth  
**Cause:** Unsloth tries to patch `torch._inductor.config` but this module structure changed between torch versions. Usually means torch version is too old (< 2.4) or there's a version mismatch.  
**Fix:** Use `torch==2.5.1+cu121`. If you see this with 2.5.1, it's usually accompanied by the `int1` error — fix that first with patch_torch.

---

## Error: `No module named 'triton.ops'`

**Affects:** bitsandbytes  
**Cause:** bitsandbytes tries to import triton.ops which doesn't exist in triton-windows.  
**Fix:** This usually resolves when patch_torch.py is imported first, because the import chain changes. If it persists:
```powershell
pip install bitsandbytes==0.43.3 --no-deps
```
Note: downgrading bitsandbytes may lose some features but will work for QLoRA training.

---

## Error: `torch.cuda.is_available()` returns `False`

**Cause:** CPU-only torch was installed from PyPI instead of CUDA version from PyTorch index.  
**How it happens:** Any `pip install` or `--force-reinstall` command that touches torch without `--index-url` will pull from PyPI.  
**Fix:**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Error: `resolution-too-deep` when installing llama-index

**Cause:** llama-index has 15+ sub-packages that create an enormous dependency tree. Pip's resolver gives up.  
**Fix:** Install sub-packages one at a time:
```powershell
pip install "llama-index-core>=0.10.0"
pip install "llama-index-embeddings-huggingface>=0.6.1"
pip install "llama-index-vector-stores-chroma>=0.4.0"
# ... etc (see setup_env.bat for full list)
pip install "llama-index>=0.10.0,<0.11.0" --no-deps
```

---

## Error: `trl requires transformers>=X.Y.Z`

**Cause:** You installed a trl version that requires a newer transformers than what's installed.  
**Fix:** Don't install trl manually. Unsloth installs the correct trl version (0.24.0 with transformers 4.57.6). If you accidentally changed trl:
```powershell
pip install trl==0.24.0 --no-deps
```

---

## Error: `unsloth-zoo requires transformers>=4.51.3` (or similar version complaints)

**Cause:** You manually downgraded transformers or other packages that Unsloth depends on.  
**Fix:** These are pip WARNINGS, not errors. If packages actually work (pass verify_env.py), ignore them. If they don't work, rebuild the environment from scratch.

---

## Error: `--force-reinstall` broke everything

**What happened:** `--force-reinstall` re-resolves ALL dependencies of the target package. This pulls torch from PyPI (CPU version) instead of the PyTorch CUDA index, which cascades into everything failing.  
**Fix:** Never use `--force-reinstall` without `--no-deps`. If you already did:
```powershell
# Reinstall CUDA torch
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Then check what else broke
python verify_env.py
```

---

## Error: echo/import "is not recognized as an internal or external command"

**Cause:** Windows CMD cannot handle multi-line Python code in `python -c "..."`. Each line gets interpreted as a separate CMD command.  
**Fix:** Use a separate `.py` file (like `verify_env.py`) instead of inline Python in batch scripts. For single-line checks, keep everything on one line:
```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Nuclear Option: Complete Environment Rebuild

If nothing works and you want to start completely fresh:

```powershell
conda deactivate
conda remove -n ambient_311 --all -y
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311
```

Then follow SETUP_GUIDE.md from Step 2, or run `setup_env.bat`.

**Before rebuilding:** Check if you have `requirements_lock_working.txt` from a previous working setup. If yes, you can restore from it instead (see "Restoring from Lock File" in SETUP_GUIDE.md).

---

## Version Compatibility Matrix

This matrix shows which version combinations work on Windows 11 with triton-windows:

| torch | CUDA index | transformers | bitsandbytes | triton-windows | patch_torch needed? | Status |
|---|---|---|---|---|---|---|
| 2.5.1+cu121 | cu121 | 4.57.6 | 0.49.2 | 3.6.0.post25 | YES | **WORKING** |
| 2.4.1+cu121 | cu121 | 4.51.3 | 0.43.3 | 3.6.0.post25 | No | Unsloth too old |
| 2.4.1+cu121 | cu121 | 4.57.6 | 0.49.2 | 3.6.0.post25 | YES | torch.int1 + PreTrainedModel errors |
| 2.6.0+cu124 | cu124 | 4.57.6 | 0.49.2 | 3.6.0.post25 | No | AttrsDescriptor error |
| 2.6.0+cu124 | cu124 | 4.57.6 | 0.49.2 | official triton | No | Linux only |

**Conclusion:** `torch==2.5.1+cu121` + `patch_torch.py` is the only combination that works on Windows with the full stack.

---

## Unsloth Warnings (Safe to Ignore)

These warnings appear during `import unsloth` and are **harmless**:

- `WARNING: Unsloth should be imported before [transformers]` — Only matters if you want Unsloth optimizations during training. For inference with merged models, irrelevant.
- `Could not patch trl.trainer.grpo_trainer: ... triton.enable_persistent_tma_matmul` — This is a torch.inductor config option from torch 2.6+ that doesn't exist in 2.5.1. Unsloth gracefully skips it.
- `Failed editing tqdm to replace Inductor Compilation` — Cosmetic patch that failed. No impact on functionality.
