python -m scripts.training.run_pplm_discrim_train \
    --dataset toxic \
    --save_model \
    --pretrained_model gpt2 \
    --batch_size 64 \
    --output_fp models/toxic_classifierhead_768/
