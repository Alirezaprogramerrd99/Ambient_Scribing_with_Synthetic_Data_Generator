# Environment Setup Guide — Ambient Clinical Scribe

**Author:** Alireza Rashidi  
**Last verified:** 23 February 2026  
**Platform:** Windows 11 + RTX 3090 (24 GB) + CUDA 12.1 + Python 3.11

---

## Working Stack (Proven 23 Feb 2026)

| Package | Version | Source |
|---|---|---|
| Python | 3.11.14 | Anaconda |
| torch | 2.5.1+cu121 | PyTorch CUDA 12.1 index |
| transformers | 4.57.6 | Pulled by Unsloth |
| unsloth | 2026.2.1 | Git (cu121-ampere-torch240) |
| unsloth-zoo | 2026.2.1 | Pulled by Unsloth |
| peft | 0.18.1 | Pulled by Unsloth |
| trl | 0.24.0 | Pulled by Unsloth |
| bitsandbytes | 0.49.2 | Pulled by Unsloth |
| datasets | 4.3.0 | Pulled by Unsloth |
| accelerate | 1.12.0 | Pulled by Unsloth |
| triton-windows | 3.6.0.post25 | Pulled by Unsloth |
| sentence-transformers | 5.2.3 | pip |
| chromadb | 1.5.1 | pip |
| llama-index | 0.10.x | pip |
| openai | 2.21.0 | pip |

---

## Prerequisites

Before starting, ensure these are installed:

1. **Anaconda or Miniconda** — Download from https://www.anaconda.com/ or https://docs.conda.io/en/latest/miniconda.html

2. **NVIDIA GPU Drivers** — Run `nvidia-smi` in PowerShell to confirm. You should see your GPU name and a CUDA version >= 12.1. Download latest from https://www.nvidia.com/drivers

3. **Visual Studio C++ Build Tools** — Required for compiling some packages. Install Visual Studio Community from https://visualstudio.microsoft.com/ and select "Desktop development with C++" workload. Make sure Windows 10/11 SDK is checked.

4. **Git** — Required for Unsloth installation from GitHub. Download from https://git-scm.com/

---

## Fresh Install (Step by Step)

### Step 1: Create a clean conda environment

Open **Anaconda PowerShell Prompt** (search in Start menu):

```powershell
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311
```

Verify:
```powershell
python --version
# Should show: Python 3.11.x
```

### Step 2: Install PyTorch with CUDA 12.1

**This MUST be the very first pip install.** If you install any other package first, pip may pull CPU-only torch from PyPI, and your entire stack will break.

```powershell
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA works:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should show: 2.5.1+cu121 True NVIDIA GeForce RTX 3090
```

**If CUDA shows False:** Your NVIDIA drivers may be too old. Update them and retry.

### Step 3: Install Unsloth

Unsloth is the training framework. It **controls the versions** of transformers, peft, trl, bitsandbytes, datasets, and accelerate. Do NOT install these packages individually before Unsloth — let Unsloth pull the versions it needs.

```powershell
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

This will install approximately 30 packages. **Wait for it to finish completely.**

After installation, verify torch is still the CUDA version (Unsloth sometimes replaces it with CPU torch):

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should show: 2.5.1+cu121 True
```

**If torch was replaced (shows a different version or False):**
```powershell
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install sentence-transformers

```powershell
pip install "sentence-transformers>=4.0.0"
```

### Step 5: Install the RAG stack

LlamaIndex has 15+ sub-packages that cause pip's resolver to fail with "resolution-too-deep" if installed together. Install them one at a time:

```powershell
pip install "chromadb>=0.4.22" "qdrant-client>=1.7.0"

pip install "llama-index-core>=0.10.0"
pip install "llama-index-embeddings-huggingface>=0.6.1"
pip install "llama-index-vector-stores-chroma>=0.4.0"
pip install "llama-index-vector-stores-qdrant>=0.1.0"
pip install "llama-index-llms-ollama>=0.1.0"
pip install "llama-index-llms-openai>=0.1.0"
pip install "llama-index-llms-anthropic>=0.1.0"

pip install "llama-index>=0.10.0,<0.11.0" --no-deps
```

The final `--no-deps` is intentional — all dependencies are already installed.

### Step 6: Install LLM providers, utilities, and evaluation

```powershell
pip install "pydantic>=2.5.0,<3.0" "pydantic-settings>=2.1.0" "python-dotenv>=1.0.0"
pip install "openai>=1.0.0" "anthropic>=0.28.0" "httpx>=0.27.0"
pip install "langchain>=0.1.0" "langchain-community>=0.0.20" "litellm>=1.0.0"
pip install "tenacity>=8.2.0" "rich>=13.7.0"
pip install "pypdf>=3.17.0" "python-docx>=1.1.0" "beautifulsoup4>=4.12.0" "instructor>=0.5.0"
pip install "ragas>=0.1.0" "rouge-score>=0.1.2" "bert-score>=0.3.13" "nltk>=3.8.0"
pip install "mlflow>=2.10.0"
pip install "pytest>=7.4.0" "jupyter>=1.0.0" "ipykernel>=6.28.0"
```

### Step 7: Place patch_torch.py in project root

Copy `patch_torch.py` to `D:\ambient-scribe\patch_torch.py`. This file adds missing `torch.int1`–`torch.int7` attributes that bitsandbytes 0.49.2 expects (they were added in PyTorch 2.6, but we're on 2.5.1).

**Every Python script in the project must import this FIRST:**
```python
import patch_torch  # Must be the very first import
from unsloth import FastLanguageModel  # Now safe to import
```

### Step 8: Verify and lock

```powershell
python verify_env.py
```

All 12 packages should show versions (no FAILED). Then **immediately lock**:

```powershell
pip freeze > requirements_lock_working.txt
```

**Keep `requirements_lock_working.txt` safe.** This is your insurance policy. Store it in version control.

---

## Automated Setup (Alternative)

If you have `setup_env.bat`, `patch_torch.py`, and `verify_env.py` in your project root:

```powershell
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311
.\setup_env.bat
```

The script runs all of Steps 2–8 automatically.

---

## Restoring from Lock File

If you have `requirements_lock_working.txt` from a working environment:

```powershell
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311

# Install CUDA torch first (lock file has the +cu121 version)
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install everything else from lock file, skipping torch (already installed)
pip install -r requirements_lock_working.txt --no-deps
```

The `--no-deps` flag is critical — it prevents pip from re-resolving and potentially pulling CPU torch.

---

## Critical Rules (Learned the Hard Way)

### Rule 1: Install Order Matters
PyTorch CUDA → Unsloth → sentence-transformers → RAG stack → everything else. Breaking this order will cause version conflicts.

### Rule 2: Unsloth Owns the ML Stack
Never manually install transformers, peft, trl, bitsandbytes, or datasets before or after Unsloth. Unsloth pins specific version combinations that work together. Installing a different version of any of these will break the chain.

### Rule 3: Never Use `--force-reinstall` Without `--no-deps`
`--force-reinstall` re-resolves every dependency and will pull CPU torch from PyPI (not the CUDA index). If you must reinstall a package:
```powershell
pip install some-package==1.2.3 --no-deps
```

### Rule 4: PyTorch CUDA Comes from a Special Index
PyTorch CUDA wheels are NOT on PyPI. They're on `https://download.pytorch.org/whl/cu121`. If any pip command installs torch without `--index-url`, you'll get the CPU version.

### Rule 5: patch_torch.py Must Be Imported First
Bitsandbytes 0.49.2 expects `torch.int1`–`torch.int7` (added in PyTorch 2.6). Since we use PyTorch 2.5.1, `patch_torch.py` mocks these as `torch.int8`. Without this patch, bitsandbytes, peft, sentence-transformers, and unsloth all fail to import.

### Rule 6: LlamaIndex Must Be Installed One Package at a Time
Installing `llama-index` with all sub-packages in one pip command causes "resolution-too-deep" errors. Install `llama-index-core` first, then each integration separately, then the meta-package with `--no-deps`.

### Rule 7: Lock Your Environment
After a working setup, run `pip freeze > requirements_lock_working.txt` and commit it to git. This is the only reliable way to reproduce the environment.

---

## Files Included

| File | Purpose | Location |
|---|---|---|
| `setup_env.bat` | Automated setup script | Project root |
| `patch_torch.py` | PyTorch 2.5.1 compatibility patch | Project root |
| `verify_env.py` | Verifies all packages import correctly | Project root |
| `requirements.txt` | Documents dependencies (NOT for direct install) | Project root |
| `requirements_lock_working.txt` | Exact working versions (generated by pip freeze) | Project root |
| `pyproject.toml` | Project metadata and optional deps | Project root |
| `TROUBLESHOOTING.md` | Solutions to common errors | Project root |
| `SETUP_GUIDE.md` | This file | Project root |
