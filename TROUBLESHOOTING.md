# Troubleshooting Guide — Ambient Clinical Scribe Environment

**Last Updated:** March 2026 (v3)

---

## Error: `ImportError: cannot import name 'sanitize_logprob' from 'trl'`

**When:** Importing unsloth or running any script after installing Unsloth from git.

**Cause:** You installed Unsloth from `git+https://github.com/unslothai/unsloth.git`.
The git main branch is bleeding-edge and expects unreleased trl functions (like
`sanitize_logprob`) that don't exist in any published trl version.

**Fix:**
```powershell
pip uninstall unsloth unsloth-zoo trl -y
pip install trl==0.24.0 --no-deps
pip install unsloth==2026.2.1 unsloth-zoo==2026.2.1 --no-deps
pip install transformers==4.57.6 huggingface_hub accelerate==1.12.0 peft==0.18.1 datasets==4.3.0 bitsandbytes==0.49.2
pip install triton-windows sentencepiece protobuf
```

**Prevention:** NEVER install Unsloth from git. Always use pinned PyPI versions.

---

## Error: `torch.int1` / `torch.int2` ... `torch.int7` AttributeError

**When:** Importing bitsandbytes, unsloth, or torchao on Windows.

**Cause:** bitsandbytes 0.49.2 and torchao reference `torch.int1`-`torch.int7` attributes
that don't exist in torch 2.5.1 (they were added in torch 2.6).

**Fix:** Import `patch_torch` as the FIRST import in every script:
```python
import patch_torch  # Must be first line
import unsloth
from transformers import AutoModelForCausalLM
```

The `patch_torch.py` file adds these attributes as aliases for `torch.int8`.

---

## Error: `PreTrainedModel` import failure after Unsloth install

**When:** `from transformers import PreTrainedModel` fails.

**Cause:** Unsloth replaced transformers with an incompatible version.

**Fix:**
```powershell
pip install transformers==4.57.6
```

---

## Error: `torch.cuda.is_available()` returns False

**When:** After installing Unsloth or other packages.

**Cause:** pip replaced CUDA torch with CPU-only torch during dependency resolution.

**Fix:**
```powershell
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Must show `2.5.1+cu121 True`.

---

## Error: `CUDA error: an illegal memory access was encountered`

**When:** During evaluation, especially when switching between model configurations
or RAG backends.

**Cause:** Loading and unloading multiple large models in the same Python process
corrupts CUDA memory on Windows. Once CUDA is corrupted, every subsequent operation
fails, including `torch.cuda.empty_cache()`.

**Fix:** The evaluator (v3, March 2026) loads each model ONCE and swaps only the
RAG retriever. If you still see this error:

1. Make sure you're using the updated `evaluator.py` (checks: model loaded once
   in `_run_comparative_experiment`, RAG backends share one model instance)
2. Restart Python completely between evaluation runs
3. Close any other GPU-using processes (Jupyter notebooks, other Python sessions)

**Prevention:** Never load/unload models in a loop. Load once, reuse.

---

## Error: `resolution-too-deep` when installing packages

**When:** Installing multiple LlamaIndex packages at once.

**Cause:** pip's dependency resolver gets stuck in circular resolution.

**Fix:** Install LlamaIndex packages one at a time:
```powershell
pip install "llama-index-core>=0.10.0"
pip install "llama-index-embeddings-huggingface>=0.6.1"
pip install "llama-index-vector-stores-chroma>=0.4.0"
# ... etc, one per line
pip install "llama-index>=0.10.0,<0.11.0" --no-deps
```

---

## Error: `AttrsDescriptor` / `triton.ops` missing

**When:** Running Unsloth inference on Windows.

**Cause:** Missing or wrong triton version. Windows needs the special `triton-windows`
package (not the Linux `triton` package).

**Fix:**
```powershell
pip install triton-windows
```

---

## Error: `--force-reinstall` broke everything

**When:** After running `pip install --force-reinstall <package>`.

**Cause:** `--force-reinstall` rebuilds ALL dependencies from scratch, ignoring the
careful install order. It typically replaces CUDA torch with CPU torch.

**Fix:** Nuclear option — full rebuild:
```powershell
conda deactivate
conda remove -n ambient_311 --all -y
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311
# Follow setup_env.bat from Step 1
```

**Prevention:** Always use `--no-deps` with `--force-reinstall`:
```powershell
pip install --force-reinstall --no-deps <package>
```

---

## Error: Evaluation produces no output files

**When:** Running the evaluator — it runs for hours but no JSON/report files appear
in the output directory.

**Cause (old evaluator):** If any experiment crashed (e.g., CUDA error in RAG backend
comparison), the exception propagated up and the save/report code never executed.

**Fix:** The updated evaluator (v3, March 2026) wraps each experiment in try/except
and always saves results, even if one experiment failed. Make sure you're using the
latest `evaluator.py`.

---

## Error: Evaluation stalls overnight / no progress until login

**When:** Running long evaluation (50 samples), left overnight, no progress when
you check in the morning, but it resumes when you log in.

**Cause:** Windows power management puts the GPU into low-power state when the
screen locks. CUDA operations stall (don't crash) until the GPU wakes up on login.

**Fix:** Before starting long runs:
```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

Or in Windows Settings: System > Power & Sleep > set everything to "Never".

Remember to restore defaults afterward.

---

## Error: `ValueError: Invalid format specifier` in evaluator report

**When:** Evaluator crashes during report generation with an f-string error.

**Cause:** The old evaluator had inline conditionals inside f-strings like
`f"{rouge:.3f if isinstance(rouge, float) else rouge}"` which Python can't parse.

**Fix:** Use the updated evaluator.py (v3) which computes formatted strings
before the f-string.

---

## Warning: `unsloth requires diffusers/hf_transfer/xformers/torchao`

**When:** After installing packages, pip shows dependency warnings.

**Cause:** These are optional Unsloth dependencies for features we don't use:
- `diffusers` — for diffusion model training
- `hf_transfer` — for faster HuggingFace Hub downloads
- `xformers` — for memory-efficient attention (optional on Windows)
- `torchao` — for quantization-aware training
- `cut_cross_entropy` — for optimized loss computation
- `msgspec` — for fast serialization

**Fix:** Ignore these warnings. They don't affect inference or evaluation.
If you want to silence them: `pip install hf_transfer msgspec` (the safe ones).

---

## Version Compatibility Matrix

This is the ONLY combination proven to work on Windows + RTX 3090:

| Package | Version | Why this version |
|---|---|---|
| Python | 3.11.x | Required by unsloth |
| torch | 2.5.1+cu121 | Last version with cu121 index; Unsloth patches work |
| transformers | 4.57.6 | Pinned by unsloth-zoo 2026.2.1 |
| unsloth | 2026.2.1 | **PyPI release** — git breaks with trl |
| unsloth-zoo | 2026.2.1 | Must match unsloth version |
| trl | 0.24.0 | Pinned by unsloth 2026.2.1 |
| peft | 0.18.1 | Compatible with transformers 4.57.6 |
| bitsandbytes | 0.49.2 | Has Windows wheel; needs patch_torch.py |
| triton-windows | 3.6.0+ | Windows-specific triton fork |
| sentencepiece | 0.2.0+ | Required by unsloth tokenizers |
| protobuf | latest | Required by unsloth serialization |

---

## Nuclear Option: Complete Environment Rebuild

If nothing else works:

```powershell
# 1. Destroy the environment
conda deactivate
conda remove -n ambient_311 --all -y

# 2. Recreate
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311

# 3. If you have the lock file:
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_lock_working.txt --no-deps

# 4. If you DON'T have the lock file, follow setup_env.bat from Step 1
# Or run: .\setup_env.bat
```
