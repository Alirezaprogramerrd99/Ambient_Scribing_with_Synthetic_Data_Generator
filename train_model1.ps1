
python -m src.student.trainer `
    --data-dir ./data/training_data `
    --output-dir ./checkpoints/phi35_clinical_scribe `
    --model unsloth/Phi-3.5-mini-instruct `
    --epochs 3 `
    --lr 2e-4 `
    --lora-r 32 `
    --batch-size 4
