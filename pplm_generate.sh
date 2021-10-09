<<<<<<< Updated upstream
OUTPUT_DIR=our_generations/toxicity-bert/pplm-small
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=models/experts/toxicity/large
API_RATE=10
NUM_GPU=4
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10200 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples 10000 \
    --num_sentences 10 \
    --dataset-file $PROMPTS_DATASET \
    --model-type pplm \
    --model toxicity-small \
    --perspective-rate-limit $API_RATE \
    --batch-size 10 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    --resume \
    $OUTPUT_DIR

    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
=======
OUTPUT_DIR=our_generations/toxicity-bert/pplm-small
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=models/experts/toxicity/large
API_RATE=10
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10200 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples 10000 \
    --num_sentences 10 \
    --dataset-file $PROMPTS_DATASET \
    --model-type pplm \
    --model toxicity-small \
    --perspective-rate-limit $API_RATE \
    --batch-size 10 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    --resume \
    $OUTPUT_DIR

    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
>>>>>>> Stashed changes
