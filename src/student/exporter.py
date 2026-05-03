"""
Model Exporter - GGUF Conversion and Ollama Registration

Handles post-training steps:
1. Merge LoRA adapters back into base model
2. Convert merged model to GGUF format
3. Generate Ollama Modelfile
4. Register the model with Ollama

Prerequisites:
    pip install unsloth
    Ollama must be installed and running (https://ollama.ai)

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""



import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    # Input
    checkpoint_dir: str = "./checkpoints/phi35_clinical_scribe/final"
    base_model: str = "unsloth/Phi-3.5-mini-instruct"
    
    # Output
    merged_dir: str = "./models/phi35_clinical_merged"
    gguf_dir: str = "./models/phi35_clinical_gguf"
    
    # GGUF quantisation method
    # Options: q4_k_m (recommended), q5_k_m (higher quality), q8_0 (highest quality)
    quantisation_method: str = "q4_k_m"
    
    # Ollama
    ollama_model_name: str = "clinical-scribe"
    ollama_base_url: str = "http://localhost:11434"
    
    # Inference defaults (baked into Modelfile)
    default_temperature: float = 0.3
    default_num_ctx: int = 4096
    default_top_p: float = 0.9
    default_repeat_penalty: float = 1.1
    
    # Max sequence length (must match training)
    max_seq_length: int = 4096


# =============================================================================
# System Prompt (same as training)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a doctor-patient "
    "conversation and relevant clinical guidelines, produce a structured "
    "clinical summary in the specified format. Be accurate and concise. "
    "Only include information explicitly stated in the conversation. "
    "Do not fabricate symptoms, medications, or findings."
)


# =============================================================================
# Exporter
# =============================================================================

class ModelExporter:
    """
    Exports a fine-tuned LoRA model for deployment via Ollama.
    
    Pipeline:
        1. Load fine-tuned model (base + LoRA adapters)
        2. Merge adapters into base weights
        3. Convert to GGUF format with quantisation
        4. Create Ollama Modelfile
        5. Register with local Ollama instance
    
    Example:
        config = ExportConfig(
            checkpoint_dir="./checkpoints/phi35_clinical_scribe/final",
            ollama_model_name="clinical-scribe",
        )
        exporter = ModelExporter(config)
        exporter.export()
    """
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def export(self) -> dict:

        """

        Run the complete export pipeline.

        """

        logger.info("=" * 60)

        logger.info("Starting Model Export Pipeline")

        logger.info("=" * 60)

        results = {}

        # Step 1: Merge LoRA adapters (16-bit safetensors)

        logger.info("Step 1: Merging LoRA adapters into base model...")

        self._merge_adapters()

        merged_path = Path("./checkpoints/phi35_clinical_scribe/hf_merged")

        results["merged_dir"] = str(merged_path)

        # Step 2 is skipped: We rely on Ollama for native conversion

        logger.info("Step 2: Skipped (Ollama will handle conversion natively)")

        # Step 3: Create Modelfile

        logger.info("Step 3: Creating Ollama Modelfile...")

        modelfile_path = self._create_modelfile(merged_path)

        results["modelfile_path"] = str(modelfile_path)

        # Step 4: Register with Ollama (Quantization happens here)

        logger.info(f"Step 4: Registering and Quantizing '{self.config.ollama_model_name}' with Ollama...")

        registered = self._register_with_ollama(modelfile_path)

        results["ollama_registered"] = registered

        results["ollama_model_name"] = self.config.ollama_model_name

        # Step 5: Verify

        if registered:

            logger.info("Step 5: Verifying model...")

            verified = self._verify_model()

            results["verified"] = verified

        logger.info("\n" + "=" * 60)

        logger.info("Export complete!")

        logger.info(f"  Run with: ollama run {self.config.ollama_model_name}")

        logger.info("=" * 60)

        return results
 
    
    # -------------------------------------------------------------------------
    # Step 1: Merge Adapters
    # -------------------------------------------------------------------------
    
    def _merge_adapters(self):
        """Merge LoRA adapters into base model weights."""

        # ---------------------------------------------------------# 
        # WINDOWS COMPATIBILITY PATCHES# Bypasses unstable Triton/Dynamo compilers on Windows
        # ---------------------------------------------------------
 
        import os
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        # Pre-load missing config to prevent Unsloth lazy-load bug
        import torch._inductor.config




        from unsloth import FastLanguageModel
        
        checkpoint = Path(self.config.checkpoint_dir)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        
        # Load model + adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint),
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        
        # Save merged (16-bit)
        merged_dir = Path(self.config.merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained_merged(
            "./checkpoints/phi35_clinical_scribe/hf_merged",
            tokenizer,
            save_method="merged_16bit",
        )
        
        logger.info(f"  Merged model saved to {merged_dir}")
    
    # -------------------------------------------------------------------------
    # Step 2: Convert to GGUF
    # -------------------------------------------------------------------------
    
    def _convert_to_gguf(self) -> Path:
        """Convert merged model to GGUF format with quantisation."""
        from unsloth import FastLanguageModel
        
        checkpoint = Path(self.config.checkpoint_dir)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint),
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        
        gguf_dir = Path(self.config.gguf_dir)
        gguf_dir.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=self.config.quantisation_method,
        )
        
        # Find the generated GGUF file
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF file generated in {gguf_dir}")
        
        gguf_path = gguf_files[0]
        size_gb = gguf_path.stat().st_size / (1024 ** 3)
        logger.info(f"  GGUF file: {gguf_path.name} ({size_gb:.2f} GB)")
        
        return gguf_path
    
    # -------------------------------------------------------------------------
    # Step 3: Create Modelfile
    # -------------------------------------------------------------------------
    
    def _create_modelfile(self, gguf_path: Path) -> Path:
        """Create an Ollama Modelfile for the exported model."""
        
        # Detect model family from base_model name for correct template
        base_lower = self.config.base_model.lower()
        is_qwen = "qwen" in base_lower
        is_llama = "llama" in base_lower
        
        if is_llama:
            model_label = "Llama-3.2"
            stop_params = (
                'PARAMETER stop <|eot_id|>\n'
                'PARAMETER stop <|end_of_text|>\n'
                'PARAMETER stop <|start_header_id|>'
            )
            template_block = (
                'TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
                '{{ .System }}<|eot_id|>'
                '<|start_header_id|>user<|end_header_id|>\n\n'
                '{{ .Prompt }}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
                '"""'
            )
        elif is_qwen:
            model_label = "Qwen2.5-3B"
            stop_params = (
                'PARAMETER stop <|im_end|>\n'
                'PARAMETER stop <|endoftext|>\n'
                'PARAMETER stop <|im_start|>'
            )
            template_block = (
                'TEMPLATE """<|im_start|>system\n'
                '{{ .System }}<|im_end|>\n'
                '<|im_start|>user\n'
                '{{ .Prompt }}<|im_end|>\n'
                '<|im_start|>assistant\n'
                '"""'
            )
        else:
            model_label = "Phi-3.5-mini"
            stop_params = (
                'PARAMETER stop <|end|>\n'
                'PARAMETER stop <|endoftext|>\n'
                'PARAMETER stop <|user|>'
            )
            template_block = (
                'TEMPLATE """<|system|>\n'
                '{{ .System }}<|end|>\n'
                '<|user|>\n'
                '{{ .Prompt }}<|end|>\n'
                '<|assistant|>\n'
                '"""'
            )
        
        modelfile_content = f"""# Clinical Scribe - Fine-tuned {model_label}
# MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
# Author: Alireza Rashidi

FROM {gguf_path.resolve()}

# Inference parameters
PARAMETER temperature {self.config.default_temperature}
PARAMETER num_ctx {self.config.default_num_ctx}
PARAMETER top_p {self.config.default_top_p}
PARAMETER repeat_penalty {self.config.default_repeat_penalty}
{stop_params}

# System prompt
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

# Template ({model_label} ChatML format)
{template_block}
"""
        
        modelfile_path = Path(self.config.gguf_dir) / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"  Modelfile created at {modelfile_path}")
        return modelfile_path
    
    # -------------------------------------------------------------------------
    # Step 4: Register with Ollama
    # -------------------------------------------------------------------------
    
    # def _register_with_ollama(self, modelfile_path: Path) -> bool:
    #     """Register the model with Ollama."""
    #     try:
    #         # Check Ollama is running
    #         check = subprocess.run(
    #             ["ollama", "list"],
    #             capture_output=True, text=True, timeout=10
    #         )
    #         if check.returncode != 0:
    #             logger.error("Ollama is not running. Start it with: ollama serve")
    #             return False
            
    #         # Create the model
    #         result = subprocess.run(
    #             ["ollama", "create", self.config.ollama_model_name, "-f", str(modelfile_path)],
    #             capture_output=True, text=True, timeout=300
    #         )
            
    #         if result.returncode == 0:
    #             logger.info(f"  ✓ Model '{self.config.ollama_model_name}' registered with Ollama")
    #             return True
    #         else:
    #             logger.error(f"  ✗ Ollama registration failed: {result.stderr}")
    #             return False
                
    #     except FileNotFoundError:
    #         logger.error("Ollama CLI not found. Install from: https://ollama.ai")
    #         return False
    #     except subprocess.TimeoutExpired:
    #         logger.error("Ollama registration timed out")
    #         return False


    def _register_with_ollama(self, modelfile_path: Path) -> bool:
        """Register the model with Ollama."""
        try:
            # Check Ollama is running
            check = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            if check.returncode != 0:
                logger.error("Ollama is not running. Start it with: ollama serve")
                return False
            # Create the model and apply quantization natively
            logger.info(f"  (This may take 2-5 minutes as Ollama converts safetensors to {self.config.quantisation_method}...)")
            result = subprocess.run(
                [
                    "ollama", "create", self.config.ollama_model_name, 
                    "-f", str(modelfile_path),
                    "--quantize", self.config.quantisation_method
                ],
                capture_output=True, text=True, timeout=1200 # Increased timeout for quantization
            )
            if result.returncode == 0:
                logger.info(f"  ✓ Model '{self.config.ollama_model_name}' registered with Ollama")
                return True
            else:
                logger.error(f"  ✗ Ollama registration failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("Ollama CLI not found. Install from: https://ollama.ai")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Ollama registration timed out")
            return False



    
    # -------------------------------------------------------------------------
    # Step 5: Verify
    # -------------------------------------------------------------------------
    
    def _verify_model(self) -> bool:
        """Verify the registered model works with a quick test."""
        try:
            import httpx
            
            test_prompt = (
                "Summarise the following clinical consultation:\n\n"
                "Doctor: Good morning. What brings you in today?\n"
                "Patient: I've had a headache for three days.\n"
                "Doctor: Can you describe the headache?\n"
                "Patient: It's a dull pressure on both sides.\n"
                "Doctor: Any other symptoms like nausea or visual changes?\n"
                "Patient: No, just the headache.\n"
                "Doctor: Any medications you're currently taking?\n"
                "Patient: Just paracetamol, but it hasn't helped much."
            )
            
            response = httpx.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {"num_predict": 200},
                },
                timeout=120.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                output = data.get("response", "")
                if len(output) > 20:
                    logger.info(f"  ✓ Verification passed. Sample output ({len(output)} chars):")
                    logger.info(f"    {output[:150]}...")
                    return True
            
            logger.warning("  ⚠ Verification produced unexpected output")
            return False
            
        except Exception as e:
            logger.warning(f"  ⚠ Verification failed: {e}")
            return False


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument("--checkpoint", default="./checkpoints/phi35_clinical_scribe/final")
    parser.add_argument("--model-name", default="clinical-scribe", help="Ollama model name")
    parser.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0"])
    parser.add_argument("--base-model", default="unsloth/Phi-3.5-mini-instruct")
    
    args = parser.parse_args()
    
    config = ExportConfig(
        checkpoint_dir=args.checkpoint,
        ollama_model_name=args.model_name,
        quantisation_method=args.quant,
        base_model=args.base_model,
    )
    
    exporter = ModelExporter(config)
    results = exporter.export()
    
    if results.get("ollama_registered"):
        print(f"\n✓ Model ready! Run with: ollama run {args.model_name}")
    else:
        print("\n⚠ Export completed but Ollama registration failed.")
        print(f"  Register manually: ollama create {args.model_name} -f {results.get('modelfile_path')}")
