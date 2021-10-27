OUTPUT_DIR=our_generations/toxicity/100k/contrastive-small-2
#PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
PROMPTS_DATASET=../data/realtoxicityprompts/prompts.jsonl

API_RATE=10
NUM_GPU=7
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples -1 \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type contrastive-gpt2 \
    --perspective-rate-limit $API_RATE \
    --model models/contrastive/small/ \
    --batch-size 16 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    $OUTPUT_DIR
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--model ../toxic_detection/result/toxic_kaggle/cont_bert_gpt2-7/ckpts/4_6235_F.pth \
