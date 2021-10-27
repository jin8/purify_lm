OUTPUT_DIR=/v3/minseon/Data/dexperts/generations/GPT_only/generation_10k/
PROMPTS_DATASET=/v3/minseon/Data/dexperts/prompts/nontoxic_prompts-10k.jsonl
#MODEL_DIR=/v3/minseon/Data/dexperts/models/contrastive/small/gpt/
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 9778 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model gpt2 \
    --batch-size 10 \
    --no-perspective-api \
    $OUTPUT_DIR
