<<<<<<< Updated upstream
OUTPUT_DIR=our_generations/toxicity-bert/gpt2-large
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=models/experts/toxicity/large
API_RATE=10
=======
OUTPUT_DIR=/v3/minseon/Data/dexperts/generations/GPT_only/generation_10k/
#PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
PROMPTS_DATASET=/v3/minseon/Data/dexperts/prompts/nontoxic_prompts-10k.jsonl
#MODEL_DIR=models/experts/toxicity/large
#API_RATE=10
>>>>>>> Stashed changes
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
<<<<<<< Updated upstream
    --num_examples 10000 \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model gpt2-large \
    --perspective-rate-limit $API_RATE \
    --batch-size 16 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    $OUTPUT_DIR

    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
OUTPUT_DIR=our_generations/toxicity-bert/gpt2-medium
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=models/experts/toxicity/large
API_RATE=10
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples 10000 \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model gpt2-medium \
    --perspective-rate-limit $API_RATE \
    --batch-size 16 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    $OUTPUT_DIR


OUTPUT_DIR=our_generations/toxicity-bert/gpt2-small
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl
MODEL_DIR=models/experts/toxicity/large
API_RATE=10
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples 10000 \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model gpt2 \
    --perspective-rate-limit $API_RATE \
    --batch-size 16 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    $OUTPUT_DIR
=======
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type gpt2 \
    --model gpt2-small \ 
    --batch-size 20 \
    --no-perspective-api \
    $OUTPUT_DIR
>>>>>>> Stashed changes
