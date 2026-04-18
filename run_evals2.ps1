# Stop script on first error
$ErrorActionPreference = "Stop"

Write-Host "Starting evaluation 1: Phi-3.5" -ForegroundColor Cyan

python -m src.student.evaluator `
    --test-data ./data/training_data/test.jsonl `
    --ft-model-path ./checkpoints/phi35_clinical_scribe/hf_merged `
    --base-model-hf unsloth/Phi-3.5-mini-instruct `
    --output-dir ./final_eval_phi_rag_ablation `
    --max-samples 50 --judge-model gpt-4o-mini --no-bertscore `
    --temperature 0.1 `
    --configs ft_rag teacher `
    --rag-configs dense_only dense_rerank dense_rerank_qe full_medical

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation 1 failed. Stopping script."
    exit 1
}

Write-Host "Starting evaluation 2: LLaMA 3.2 3B" -ForegroundColor Cyan

python -m src.student.evaluator `
    --test-data ./data/training_data_llama3b/test.jsonl `
    --ft-model-path ./checkpoints/llama32_3b_clinical_scribe/hf_merged `
    --base-model-hf unsloth/Llama-3.2-3B-Instruct `
    --output-dir ./final_eval_llama3b_rag_ablation `
    --max-samples 50 --judge-model gpt-4o-mini --no-bertscore `
    --temperature 0.1 `
    --configs ft_rag teacher `
    --rag-configs dense_only dense_rerank dense_rerank_qe full_medical

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation 2 failed. Stopping script."
    exit 1
}

Write-Host "Starting evaluation 3: LLaMA 3.2 1B" -ForegroundColor Cyan

python -m src.student.evaluator `
    --test-data ./data/training_data_llama1b/test.jsonl `
    --ft-model-path ./checkpoints/llama32_1b_clinical_scribe/hf_merged `
    --base-model-hf unsloth/Llama-3.2-1B-Instruct `
    --output-dir ./final_eval_llama1b_rag_ablation `
    --max-samples 50 --judge-model gpt-4o-mini --no-bertscore `
    --temperature 0.1 `
    --configs ft_rag teacher `
    --rag-configs dense_only dense_rerank dense_rerank_qe full_medical

if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation 3 failed."
    exit 1
}

Write-Host "✅ All evaluations completed successfully." -ForegroundColor Green
