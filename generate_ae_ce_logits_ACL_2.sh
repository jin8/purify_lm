OUTPUT_DIR=/v3/minseon/Data/dexperts/generations/ae_ce_logits/generation_10k/w05/
#PROMPTS_DATASET=/v3/minseon/iclr/data/realtoxicityprompts/prompts.jsonl
PROMPTS_DATASET=/v3/minseon/Data/dexperts/prompts/nontoxic_prompts-10k.jsonl
MODEL_DIR=/v3/minseon/Data/dexperts/models/contrastive/small/mapping_autoencoder/
NUM_GPU=4
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,4,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 8888 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type attadd \
    --model $MODEL_DIR \
    --batch-size 20 \
    --no-perspective-api \
    $OUTPUT_DIR

