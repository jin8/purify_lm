DATA_DIR=datasets/jigsaw-unintended-bias-in-toxicity-classification
BATCH_SIZE=8
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=4
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --master_addr 127.0.0.1  \
    --master_port 10009 \
    --nproc_per_node $NUM_GPU \
	scripts/finetuning/finetune_contrastive.py \
	--output_dir models/contrastive/small/ \
	--model_type gpt2 \
	--model_name_or_path gpt2 \
	--do_train \
	--num_train_epochs 5 \
	--block_size $BLOCK_SIZE \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_files $DATA_DIR/toxicity_eq0.txt  $DATA_DIR/toxicity_gte0.5.txt
	#--overwrite_cache
