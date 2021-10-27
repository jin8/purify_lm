P=0.9
PROMPTS_DATASET=prompts/sentiment_prompts-10k/negative_prompts.jsonl
OUTPUT_DIR=our_generations/prompted_sentiment-10k/negative_prompts/contrastive-latent1-gpu8-ep100/positive/

CUDA_VISIBLE_DEVICES=0 python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type contrastive-gpt2 \
    --model models/sentiment/contrastive-latent1-gpu8-ep100/small/ \
    --positive \
    --p $P \
    $OUTPUT_DIR
