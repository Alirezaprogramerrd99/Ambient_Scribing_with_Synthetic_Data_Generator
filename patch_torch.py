"""
Windows Compatibility Patch for PyTorch 2.5.1 + Unsloth/Bitsandbytes

This module MUST be imported BEFORE any other ML imports (unsloth, transformers,
peft, sentence-transformers, bitsandbytes). It patches missing torch attributes
that were introduced in PyTorch 2.6 but are expected by torchao/bitsandbytes.

Usage:
    import patch_torch  # Must be the VERY FIRST import
    from unsloth import FastLanguageModel
    ...

What it does:
    1. Adds torch.int1 through torch.int7 (mock as torch.int8)
       - Required by bitsandbytes >= 0.45 and torchao
       - These dtypes were added in PyTorch 2.6 for quantization support
       - Mocking as int8 is safe because the actual quantization logic
         doesn't depend on these being distinct types on Windows

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import torch

# Add missing int1-int7 dtypes (introduced in torch 2.6, needed by torchao/bitsandbytes)
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
