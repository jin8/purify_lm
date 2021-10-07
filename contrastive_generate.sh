OUTPUT_DIR=our_generations/toxicity/nontoxic10k/kaggle64-contrastive-small
PROMPTS_DATASET=prompts/nontoxic_prompts-10k.jsonl
API_RATE=10
NUM_GPU=4
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=2,3,4,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
    scripts/run_toxicity_experiment.py \
    --use-dataset \
    --num_examples 10000 \
    --num_sentences 25 \
    --dataset-file $PROMPTS_DATASET \
    --model-type contrastive-gpt2 \
    --model ../toxic_detection/result/toxic_kaggle/cont_bert_gpt2-7/ckpts/4_6235_F.pth \
    --perspective-rate-limit $API_RATE \
    --batch-size 16 \
    --alpha 2.0 \
    --filter_p 0.9 \
    --no-perspective-api \
    $OUTPUT_DIR
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
    #--toxic-model $MODEL_DIR/finetuned_gpt2_toxic \
