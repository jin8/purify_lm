OUTPUT_DIR=/v3/minseon/Data/dexperts/generations/cont_gpt_last/generation_10k/
PROMPTS_DATASET=/v3/minseon/Data/dexperts/prompts/nontoxic_prompts-10k.jsonl
MODEL_DIR=/v3/minseon/Data/dexperts/models/contrastive/small/ablation_last/
NUM_GPU=4
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 4477 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type cont_gpt_last \
    --model $MODEL_DIR \
    --batch-size 10 \
    --no-perspective-api \
    $OUTPUT_DIR


