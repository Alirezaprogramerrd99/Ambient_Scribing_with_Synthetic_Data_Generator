@echo off
REM =============================================================================
REM Environment Setup for Ambient Clinical Scribe
REM Windows 11 + RTX 3090 + CUDA 12.1 + Python 3.11
REM
REM PROVEN WORKING STACK (February-March 2026):
REM   torch 2.5.1+cu121, transformers 4.57.6, unsloth 2026.2.1
REM   peft 0.18.1, trl 0.24.0, bitsandbytes 0.49.2
REM
REM v3 UPDATE (March 2026):
REM   - Unsloth now installed from PyPI (NOT git) to avoid sanitize_logprob error
REM   - Added triton-windows, sentencepiece, protobuf as explicit dependencies
REM   - Added lock file restore as first option
REM
REM Prerequisites:
REM   - Anaconda/Miniconda installed
REM   - NVIDIA drivers installed (nvidia-smi should work)
REM   - Visual Studio C++ Build Tools installed
REM
REM Usage:
REM   1. Place patch_torch.py and verify_env.py in project root
REM   2. Open Anaconda PowerShell Prompt (or cmd)
REM   3. cd D:\ambient-scribe
REM   4. .\setup_env.bat
REM
REM Author: Alireza Rashidi
REM =============================================================================

echo ============================================================
echo  Ambient Clinical Scribe - Environment Setup
echo  Target: Python 3.11 + CUDA 12.1 + RTX 3090
echo ============================================================
echo.

REM -----------------------------------------------------------------
REM STEP 0: Check for lock file (fast recovery path)
REM -----------------------------------------------------------------
if exist requirements_lock_working.txt (
    echo [INFO] Found requirements_lock_working.txt
    echo [INFO] Attempting fast recovery from lock file...
    echo.
    set /p FAST_RESTORE="Use lock file for fast restore? [Y/N]: "
    if /i "%FAST_RESTORE%"=="Y" (
        echo [STEP 0] Installing PyTorch CUDA first...
        pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        echo [STEP 0] Restoring from lock file...
        pip install -r requirements_lock_working.txt --no-deps
        echo [STEP 0] Verifying...
        python verify_env.py
        if not errorlevel 1 (
            echo.
            echo [OK] Fast recovery successful!
            pause
            exit /b 0
        )
        echo [WARN] Fast recovery failed. Continuing with full setup...
        echo.
    )
)

REM -----------------------------------------------------------------
REM STEP 0b: Check Python version
REM -----------------------------------------------------------------
echo [STEP 0] Checking Python...

python --version 2>nul | findstr "3.11" >nul
if errorlevel 1 (
    echo [INFO] Creating conda environment...
    conda create -n ambient_311 python=3.11 -y
    echo [INFO] Please run: conda activate ambient_311
    echo [INFO] Then re-run this script.
    pause
    exit /b 1
)
echo [OK] Python 3.11 detected
echo.

REM -----------------------------------------------------------------
REM STEP 1: Install PyTorch with CUDA 12.1
REM
REM CRITICAL: This MUST be the first pip install.
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 1/7: PyTorch + CUDA 12.1
echo ============================================================

pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

python -c "import torch; assert torch.cuda.is_available(); print('[OK] PyTorch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"

if errorlevel 1 (
    echo [ERROR] CUDA not available. Check NVIDIA drivers.
    pause
    exit /b 1
)
echo.

REM -----------------------------------------------------------------
REM STEP 2: Install Unsloth + ML stack
REM
REM CRITICAL (March 2026): Install from PyPI, NOT from git.
REM The git main branch changes daily and breaks against released
REM versions of trl/transformers (e.g. sanitize_logprob error).
REM
REM We use --no-deps on unsloth/trl to prevent pip from pulling
REM incompatible transitive dependencies, then install deps
REM explicitly with pinned versions.
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 2/7: Unsloth + ML Stack (PyPI pinned versions)
echo ============================================================

echo   Installing trl (pinned, no-deps)...
pip install trl==0.24.0 --no-deps

echo   Installing unsloth + unsloth-zoo (pinned, no-deps)...
pip install unsloth==2026.2.1 unsloth-zoo==2026.2.1 --no-deps

echo   Installing ML stack dependencies...
pip install transformers==4.57.6 huggingface_hub accelerate==1.12.0 peft==0.18.1 datasets==4.3.0 bitsandbytes==0.49.2

echo   Installing unsloth system dependencies...
pip install triton-windows sentencepiece protobuf

REM Verify torch is still CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA lost'; assert 'cu12' in torch.__version__, 'Wrong torch'; print('[OK] torch', torch.__version__, 'still CUDA')"

if errorlevel 1 (
    echo [WARN] Reinstalling CUDA torch (may have been overwritten)...
    pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)
echo.

REM -----------------------------------------------------------------
REM STEP 3: Install sentence-transformers
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 3/7: Sentence-Transformers
echo ============================================================

pip install sentence-transformers==5.2.3
echo.

REM -----------------------------------------------------------------
REM STEP 4: Install RAG stack (one at a time)
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 4/7: RAG Stack (LlamaIndex + ChromaDB)
echo ============================================================

echo   Installing vector databases...
pip install "chromadb>=0.4.22" "qdrant-client>=1.7.0"

echo   Installing LlamaIndex core...
pip install "llama-index-core>=0.10.0"

echo   Installing LlamaIndex integrations...
pip install "llama-index-embeddings-huggingface>=0.6.1"
pip install "llama-index-vector-stores-chroma>=0.4.0"
pip install "llama-index-vector-stores-qdrant>=0.1.0"
pip install "llama-index-llms-ollama>=0.1.0"
pip install "llama-index-llms-openai>=0.1.0"
pip install "llama-index-llms-anthropic>=0.1.0"

echo   Installing LlamaIndex meta-package...
pip install "llama-index>=0.10.0,<0.11.0" --no-deps
echo.

REM -----------------------------------------------------------------
REM STEP 5: LLM providers and utilities
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 5/7: LLM Providers + Utilities
echo ============================================================

pip install "pydantic>=2.5.0,<3.0" "pydantic-settings>=2.1.0" "python-dotenv>=1.0.0"
pip install "openai>=1.0.0" "anthropic>=0.28.0" "httpx>=0.27.0"
pip install "langchain>=0.1.0" "langchain-community>=0.0.20" "litellm>=1.0.0"
pip install "tenacity>=8.2.0" "rich>=13.7.0"
pip install "pypdf>=3.17.0" "python-docx>=1.1.0" "beautifulsoup4>=4.12.0" "instructor>=0.5.0"
echo.

REM -----------------------------------------------------------------
REM STEP 6: Evaluation and experiment tracking
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 6/7: Evaluation + Tracking
echo ============================================================

pip install "ragas>=0.1.0" "rouge-score>=0.1.2" "bert-score>=0.3.13" "nltk>=3.8.0"
pip install "mlflow>=2.10.0"
pip install "pytest>=7.4.0" "jupyter>=1.0.0" "ipykernel>=6.28.0"
echo.

REM -----------------------------------------------------------------
REM STEP 7: Verify everything
REM -----------------------------------------------------------------
echo ============================================================
echo  STEP 7/7: Verification
echo ============================================================

python verify_env.py

echo.

REM Lock the environment
echo Locking environment...
pip freeze > requirements_lock_working.txt
echo [OK] Locked to requirements_lock_working.txt
echo [IMPORTANT] Commit this file to git: git add requirements_lock_working.txt

echo.
echo ============================================================
echo  SETUP COMPLETE!
echo ============================================================
echo.
echo  IMPORTANT NOTES:
echo    1. patch_torch.py must be imported FIRST in all scripts
echo    2. requirements_lock_working.txt is your insurance policy
echo    3. Never use --force-reinstall without --no-deps
echo    4. NEVER install unsloth from git - always use PyPI pinned version
echo    5. Commit requirements_lock_working.txt to git NOW
echo.
echo  To use the project:
echo    conda activate ambient_311
echo    python -m src.student.run_student_pipeline --help
echo.
pause
