DATA_DIR=datasets/SST-5
BATCH_SIZE=4
BLOCK_SIZE=128
GRAD_ACCUM_STEPS=16
NUM_GPU=8
# Allow multiple threads
export OMP_NUM_THREADS=20
# Use distributed data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
	--master_addr 127.0.0.1  \
    --master_port 11129 \
    --nproc_per_node $NUM_GPU \
	scripts/finetuning/finetune_style_transfer.py \
	--output_dir models/sentiment/fuse_rev_style_pred_contrast1.0_ep20_with_project_balance \
	--model_type gpt2 \
	--model_name_or_path gpt2 \
	--contrastive_factor 1.0 \
	--do_train \
	--balance \
	--num_train_epochs 20 \
	--block_size $BLOCK_SIZE \
	--save_total_limit 1 \
	--dataloader_drop_last \
	--per_device_train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
	--train_data_files $DATA_DIR/negative.txt  $DATA_DIR/positive.txt \
	--overwrite_cache
