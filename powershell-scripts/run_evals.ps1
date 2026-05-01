# Phi-3.5 Evaluation
# Write-Host "Starting Phi-3.5 Evaluation..." -ForegroundColor Cyan
# python -m src.student.evaluator `
#     --test-data ./data/training_data/test.jsonl `
#     --ft-model-path ./checkpoints/phi35_clinical_scribe/hf_merged `
#     --base-model-hf unsloth/Phi-3.5-mini-instruct `
#     --output-dir ./final_eval_phi `
#     --max-samples 50 `
#     --judge-model gpt-4o-mini `
#     --no-bertscore `
#     --temperature 0.1 `
#     --rag-configs dense_only dense_rerank `
#     --return-logprobs

# Llama-3.2-3B Evaluation
Write-Host "Starting Llama-3.2-3B Evaluation..." -ForegroundColor Cyan
python -m src.student.evaluator `
    --test-data ./data/training_data_llama3b/test.jsonl `
    --ft-model-path ./checkpoints/llama32_3b_clinical_scribe/hf_merged `
    --base-model-hf unsloth/Llama-3.2-3B-Instruct `
    --output-dir ./final_eval_llama3b `
    --max-samples 50 `
    --judge-model gpt-4o-mini `
    --no-bertscore `
    --temperature 0.1 `
    --rag-configs dense_only dense_rerank `
    --return-logprobs

# Llama-3.2-1B Evaluation
Write-Host "Starting Llama-3.2-1B Evaluation..." -ForegroundColor Cyan
python -m src.student.evaluator `
    --test-data ./data/training_data_llama1b/test.jsonl `
    --ft-model-path ./checkpoints/llama32_1b_clinical_scribe/hf_merged `
    --base-model-hf unsloth/Llama-3.2-1B-Instruct `
    --output-dir ./final_eval_llama1b `
    --max-samples 50 `
    --judge-model gpt-4o-mini `
    --no-bertscore `
    --temperature 0.1 `
    --rag-configs dense_only dense_rerank `
    --return-logprobs

Write-Host "All evaluations complete!" -ForegroundColor Green