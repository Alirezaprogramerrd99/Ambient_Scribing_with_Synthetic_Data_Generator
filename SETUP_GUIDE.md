# Environment Setup Guide — Ambient Clinical Scribe

**Author:** Alireza Rashidi
**Last Updated:** March 2026 (v3 — post-wipe recovery tested)
**Platform:** Windows 11 + RTX 3090 + CUDA 12.1 + Python 3.11

---

## Quick Recovery (Machine Was Wiped)

If you already have `requirements_lock_working.txt` from a previous setup, use this
**fast-path** — it takes ~5 minutes instead of 30:

```powershell
# 1. Create and activate conda env
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311

# 2. Install PyTorch CUDA FIRST (must come from special index)
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Restore everything else from the lock file
pip install -r requirements_lock_working.txt --no-deps

# 4. Verify
python verify_env.py
```

If this works — you're done. If not, follow the full setup below.

---

## Prerequisites

Before starting, ensure you have:

1. **Anaconda or Miniconda** — [Download](https://docs.conda.io/en/latest/miniconda.html)
2. **NVIDIA GPU drivers** — Run `nvidia-smi` to confirm. [Download](https://www.nvidia.com/drivers)
3. **Visual Studio C++ Build Tools** — Required for compiling native extensions. Install with "Desktop development with C++" workload selected. Also select Windows 10/11 SDK.
4. **Git** — For cloning repos if needed.

---

## Working Stack (Proven February–March 2026)

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.x | Via conda |
| torch | 2.5.1+cu121 | From pytorch.org CUDA 12.1 index |
| transformers | 4.57.6 | Pinned by unsloth-zoo |
| unsloth | 2026.2.1 | **From PyPI, NOT from git** |
| unsloth-zoo | 2026.2.1 | **From PyPI, NOT from git** |
| trl | 0.24.0 | Pinned by unsloth |
| peft | 0.18.1 | |
| bitsandbytes | 0.49.2 | Windows wheel |
| accelerate | 1.12.0 | |
| sentence-transformers | 5.2.3 | |
| triton-windows | 3.6.0+ | Required by unsloth on Windows |
| sentencepiece | 0.2.0+ | Required by unsloth |
| protobuf | latest | Required by unsloth |

---

## Full Setup (Step by Step)

### Step 1: Create conda environment

```powershell
conda create -n ambient_311 python=3.11 -y
conda activate ambient_311
```

### Step 2: Install PyTorch with CUDA 12.1

**CRITICAL:** This must be the FIRST pip install. The CUDA version comes from a
special index URL that only has wheels up to torch 2.5.1.

```powershell
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.5.1+cu121 True NVIDIA GeForce RTX 3090`

### Step 3: Install Unsloth + ML stack

**WARNING: Use pinned PyPI versions, NEVER install from git.**

The Unsloth git main branch updates daily and frequently breaks against released
versions of trl and transformers. The `sanitize_logprob` error (March 2026) was
caused by installing from git — the git version expected unreleased trl functions.

```powershell
# Install trl first with --no-deps to prevent version conflicts
pip install trl==0.24.0 --no-deps

# Install unsloth + unsloth-zoo from PyPI (NOT from git)
pip install unsloth==2026.2.1 unsloth-zoo==2026.2.1 --no-deps

# Install the ML stack dependencies that --no-deps skipped
pip install transformers==4.57.6 huggingface_hub accelerate==1.12.0 peft==0.18.1 datasets==4.3.0 bitsandbytes==0.49.2

# Install unsloth's required system dependencies
pip install triton-windows sentencepiece protobuf
```

Verify torch is still CUDA (Unsloth installs can sometimes overwrite it):
```powershell
python -c "import torch; assert 'cu12' in torch.__version__; print('OK:', torch.__version__)"
```

If this fails (CUDA torch was replaced), reinstall it:
```powershell
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install sentence-transformers

```powershell
pip install sentence-transformers==5.2.3
```

### Step 5: Install RAG stack

Install one at a time to avoid dependency resolution failures:

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

### Step 6: Install LLM providers and utilities

```powershell
pip install "pydantic>=2.5.0,<3.0" "pydantic-settings>=2.1.0" "python-dotenv>=1.0.0"
pip install "openai>=1.0.0" "anthropic>=0.28.0" "httpx>=0.27.0"
pip install "langchain>=0.1.0" "langchain-community>=0.0.20" "litellm>=1.0.0"
pip install "tenacity>=8.2.0" "rich>=13.7.0"
pip install "pypdf>=3.17.0" "python-docx>=1.1.0" "beautifulsoup4>=4.12.0" "instructor>=0.5.0"
```

### Step 7: Install evaluation packages

```powershell
pip install "ragas>=0.1.0" "rouge-score>=0.1.2" "bert-score>=0.3.13" "nltk>=3.8.0"
pip install "mlflow>=2.10.0"
pip install "pytest>=7.4.0" "jupyter>=1.0.0" "ipykernel>=6.28.0"
```

### Step 8: Verify and lock

```powershell
python verify_env.py
```

All 12 packages should pass. Then **immediately lock**:

```powershell
pip freeze > requirements_lock_working.txt
```

**Commit this file to git.** It is your insurance policy against machine wipes.

```powershell
git add requirements_lock_working.txt
git commit -m "Lock working environment (March 2026)"
```

---

## Critical Rules (Learned from Debugging)

1. **NEVER install Unsloth from git.** Always use `pip install unsloth==2026.2.1`.
   The git version changes daily and breaks against released trl/transformers.

2. **PyTorch CUDA must come from the special index.** Always use
   `--index-url https://download.pytorch.org/whl/cu121`. Without it, pip installs
   CPU-only torch.

3. **Install order matters.** PyTorch first, then Unsloth+trl (--no-deps), then ML deps,
   then everything else. Reversing this causes version conflicts.

4. **Never use --force-reinstall without --no-deps.** Force-reinstall rebuilds
   the entire dependency tree and will replace CUDA torch with CPU torch.

5. **patch_torch.py must be imported FIRST** in every script that uses unsloth,
   transformers, peft, or sentence-transformers. It patches torch.int1-torch.int7
   which bitsandbytes 0.49.2 requires on Windows.

6. **Install LlamaIndex packages one at a time.** Installing them all together
   triggers pip's resolution-too-deep error.

7. **Lock immediately after verification.** Run `pip freeze > requirements_lock_working.txt`
   and commit to git. Next time the machine is wiped, recovery takes 5 minutes.

8. **Disable sleep for long evaluations.** Windows puts the GPU to sleep when the
   screen locks, stalling all CUDA operations. Before long runs:
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change monitor-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   ```

---

## Safe-to-Ignore Warnings

These warnings appear during normal operation and do NOT indicate errors:

- `Unsloth: WARNING: triton.enable_persistent_tma_matmul not found` — config option
  from torch 2.6+ that doesn't exist in 2.5.1. Harmless.
- `WARNING: pip's dependency resolver does not currently take into account all the
  packages that are installed` — Normal with --no-deps installs.
- `unsloth requires diffusers/hf_transfer/xformers, which is not installed` — Optional
  dependencies for features we don't use (diffusion models, fast downloads, flash attention).
- `unsloth-zoo requires cut_cross_entropy/msgspec/torchao, which is not installed` — Optional
  performance dependencies. Not needed for inference/evaluation.
- `WARNING: Some parameters are on the meta device because they were offloaded to the cpu`
  — Normal when loading large models with accelerate.
