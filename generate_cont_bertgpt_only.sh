OUTPUT_DIR=/v3/minseon/Data/dexperts/generations/cont_gpt_bert_only/generation/
PROMPTS_DATASET=/v3/minseon/iclr/data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=/v3/minseon/Data/dexperts/models/contrastive/small/cont_gpt_bert_only/
NUM_GPU=4
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 7488 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type cont_bert_gpt_only \
    --model $MODEL_DIR \
    --batch-size 40 \
    --no-perspective-api \
    $OUTPUT_DIR
