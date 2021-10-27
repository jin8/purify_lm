P=0.9
PROMPTS_DATASET=prompts/sentiment_prompts-10k/neutral_prompts.jsonl
OUTPUT_DIR=our_generations/prompted_sentiment-10k/neutral_prompts/fuse_rev_style_pred_contrast1.0_ep30/positive/

CUDA_VISIBLE_DEVICES=0 python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type style-gpt2-attr \
    --model models/sentiment/fuse_rev_style_pred_contrast1.0_ep30/  \
    --positive \
    --p $P \
    $OUTPUT_DIR


PROMPTS_DATASET=prompts/sentiment_prompts-10k/negative_prompts.jsonl
OUTPUT_DIR=our_generations/prompted_sentiment-10k/negative_prompts/fuse_rev_style_pred_contrast1.0_ep30/positive/
CUDA_VISIBLE_DEVICES= python -m scripts.run_sentiment_experiment \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model-type style-gpt2-attr \
    --model models/sentiment/fuse_rev_style_pred_contrast1.0_ep30/  \
    --positive \
    --p $P \
    $OUTPUT_DIR
