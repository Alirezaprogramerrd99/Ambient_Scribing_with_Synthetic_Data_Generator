"""
Example: Running the Pipeline with LlamaIndex RAG

This script demonstrates how to:
1. Build a knowledge base using LlamaIndex
2. Run the synthetic data pipeline with LlamaIndex RAG
3. Compare results with manual RAG implementation

Prerequisites:
    pip install llama-index llama-index-embeddings-huggingface llama-index-vector-stores-chroma

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads OPENAI_API_KEY from .env file
    print("✓ Loaded environment variables from .env")
except ImportError:
    print("python-dotenv not installed. Make sure OPENAI_API_KEY is set manually!")
    print("Install with: pip install python-dotenv")

# Verify API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment!")
    print("  Options:")
    print("  1. Create a .env file with: OPENAI_API_KEY=sk-...")
    print("  2. Set environment variable: export OPENAI_API_KEY=sk-...")
    print("  3. On Windows PowerShell: $env:OPENAI_API_KEY = 'sk-...'")
else:
    print("✓ OPENAI_API_KEY found")

from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend
from src.pipeline import SyntheticDataPipeline, PipelineConfig


def build_llama_index_knowledge_base(
    documents_dir: str = "./medical_knowledge/sample",
    persist_dir: str = "./data/llama_index_chroma_db",
):
    """
    Build knowledge base using LlamaIndex
    
    Args:
        documents_dir: Path to medical documents
        persist_dir: Path to persist the index
    """
    print("=" * 60)
    print("Building LlamaIndex Knowledge Base")
    print("=" * 60)
    
    config = RAGConfig(
        backend=RAGBackend.LLAMA_INDEX,
        vector_store="chroma",
        persist_dir=persist_dir,
        embedding_model="BAAI/bge-base-en-v1.5",
        chunk_size=512,
        chunk_overlap=50,
        similarity_top_k=8,
    )
    
    factory = RAGFactory(config)
    
    # Build knowledge base
    stats = factory.build_knowledge_base(
        documents_dir,
        clear_existing=True,
    )
    
    print(f"✓ Knowledge base built: {stats}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    retriever = factory.get_retriever()
    response = retriever.retrieve("chest pain acute coronary syndrome ECG troponin", top_k=3)
    
    print(f"Query: 'chest pain acute coronary syndrome ECG troponin'")
    print(f"Results ({len(response.results)} found):")
    for i, r in enumerate(response.results, 1):
        print(f"  {i}. Score: {r.score:.3f} - {r.source_file}")
        print(f"     Preview: {r.text[:100]}...")
    
    return factory


def run_pipeline_with_llama_index(
    num_scenarios: int = 5,
    knowledge_base_path: str = "./medical_knowledge/sample",
    persist_dir: str = "./data/llama_index_chroma_db",
    output_dir: str = "./data/synthetic_output_llama_index",
    scenario_seed : int = 42,
):
    """
    Run the synthetic data pipeline with LlamaIndex RAG
    
    Args:
        num_scenarios: Number of scenarios to generate
        knowledge_base_path: Path to medical knowledge documents
        persist_dir: Path to LlamaIndex persist directory
    """
    print("\n" + "=" * 60)
    print("Running Pipeline with LlamaIndex RAG")
    print("=" * 60)
    
    config = PipelineConfig(
        # Output settings
        output_dir=Path(output_dir),
        experiment_name="llama_index_rag_test",
        num_scenarios=num_scenarios,
        scenario_seed=scenario_seed,
        
        # RAG settings - LlamaIndex
        use_rag=True,
        rag_backend=RAGBackend.LLAMA_INDEX,
        knowledge_base_path=Path(knowledge_base_path),
        rag_top_k=8,
        
        # Teacher settings
        teacher_provider="openai",
        teacher_model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4096,
        
        # Validation
        enable_validation=True,
        enable_clinical_validation=True,
        enable_rag_validation=True,
        filter_invalid=True,
        
        # Benchmarking
        enable_benchmarking=True,
        compute_bertscore=False,
        compute_ragas=False,  # Set True if you have RAGAS installed
        generate_benchmark_report=True,
        
        # Processing
        batch_size=5,
        max_retries=4,
        
        # Tracking
        use_mlflow=True,
        use_wandb=False,
    )
    
    # Create and run pipeline
    pipeline = SyntheticDataPipeline(config)
    result = pipeline.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"  Total scenarios:     {result.total_scenarios}")
    print(f"  Generated samples:   {result.total_generated}")
    print(f"  Valid samples:       {result.total_valid}")
    print(f"  Success rate:        {result.success_rate:.1%}")
    print(f"  Total time:          {result.total_seconds:.1f}s")
    print(f"  Output:              {result.output_path}")
    
    if result.benchmark_result:
        print("\nBenchmark Scores:")
        print(f"  Overall Score:       {result.benchmark_result.overall_score:.3f}")
        print(f"  Clinical Accuracy:   {result.benchmark_result.clinical_metrics.get('clinical_accuracy', 'N/A')}")
        print(f"  RAGAS Overall:       {result.benchmark_result.ragas_metrics.get('overall_score', 'N/A')}")
        print(f"  Report:              {result.benchmark_report_path}")
    
    return result


def compare_backends(num_scenarios: int = 5):
    """
    Compare Manual vs LlamaIndex RAG backends
    
    Runs the pipeline with both backends and compares results.
    """
    print("\n" + "=" * 60)
    print("Comparing RAG Backends: Manual vs LlamaIndex")
    print("=" * 60)
    
    results = {}
    
    for backend in [RAGBackend.MANUAL, RAGBackend.LLAMA_INDEX]:
        print(f"\n--- Running with {backend.value} backend ---")
        
        persist_dir = f"./data/{backend.value}_chroma_db"
        output_dir = f"./data/synthetic_output_{backend.value}"
        
        config = PipelineConfig(
            output_dir=Path(output_dir),
            experiment_name=f"{backend.value}_comparison",
            num_scenarios=num_scenarios,
            scenario_seed=42,  # Same seed for fair comparison
            
            use_rag=True,
            rag_backend=backend,
            knowledge_base_path=Path("./medical_knowledge/sample"),
            rag_top_k=8,
            
            teacher_provider="openai",
            teacher_model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=4096,
            
            enable_benchmarking=True,
            generate_benchmark_report=True,
            
            use_mlflow=False,
        )
        
        pipeline = SyntheticDataPipeline(config)
        result = pipeline.run()
        
        results[backend.value] = {
            "success_rate": result.success_rate,
            "total_time": result.total_seconds,
            "overall_score": result.benchmark_result.overall_score if result.benchmark_result else None,
            "ragas_score": result.benchmark_result.ragas_metrics.get("overall_score") if result.benchmark_result else None,
        }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(f"{'Metric':<25} {'Manual':<15} {'LlamaIndex':<15}")
    print("-" * 55)
    
    for metric in ["success_rate", "total_time", "overall_score", "ragas_score"]:
        manual_val = results["manual"].get(metric)
        llama_val = results["llama_index"].get(metric)
        
        if isinstance(manual_val, float):
            manual_str = f"{manual_val:.3f}"
            llama_str = f"{llama_val:.3f}" if llama_val else "N/A"
        else:
            manual_str = str(manual_val)
            llama_str = str(llama_val) if llama_val else "N/A"
        
        print(f"{metric:<25} {manual_str:<15} {llama_str:<15}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline with LlamaIndex RAG")
    parser.add_argument("--build-kb", action="store_true", help="Build knowledge base first")
    parser.add_argument("--run", action="store_true", help="Run pipeline")
    parser.add_argument("--compare", action="store_true", help="Compare Manual vs LlamaIndex")
    parser.add_argument("--num-scenarios", type=int, default=5, help="Number of scenarios")
    
    args = parser.parse_args()
    
    if args.build_kb:
        build_llama_index_knowledge_base()
    
    if args.run:
        run_pipeline_with_llama_index(num_scenarios=args.num_scenarios)
    
    if args.compare:
        compare_backends(num_scenarios=args.num_scenarios)
    
    if not any([args.build_kb, args.run, args.compare]):
        # Default: build KB and run pipeline
        print("No arguments provided. Running default workflow...")
        print("Use --help to see available options.\n")
        
        build_llama_index_knowledge_base()
        # Use a different seed for the main run to ensure variability
        run_pipeline_with_llama_index(num_scenarios=250, output_dir="./data/batch_8", scenario_seed=49)