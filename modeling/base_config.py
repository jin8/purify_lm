import torch
import random


class Config():
    data_path = ''
    #data_path = '/v6/myung/iclr/data/toxic_kaggle/'

    pretrained_embed_path = './embedding/'
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    min_freq = 1
    max_length = 64
    block_size = -1
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    batch_size = 16
    lr = 5e-5 # 5e-5
    iter_D = 10
    iter_F = 5
    num_epochs = 5
    early_stop = -1
    log_steps = 5
    eval_steps = 25
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]
    inp_drop_prob = 0

    load_model = ''
    approach = ''

    log_dir = ''#'runs/{}-{}-save'
    save_path = '' #''

    gradient_accumulation_steps = 1
    adam_epsilon = 1e-8
    warmup_steps = 0
    weight_decay = 0
    # Style-Transformer :st & pretrained_model = None or 'gpt2'
    # Domain-Adaptive Pretraining : dapt & pretrained_model = 'gpt2'
    # Attribute Conditioning : atcon & pretrained_model = 'gpt2'

    num_samples = 25
    test_batch_size = 64
    #prompt_path='/v6/myung/iclr/data/realtoxicityprompts/prompts.jsonl'

    prompt_path='/v6/myung/iclr/purify_lm/prompts/nontoxic_prompts-10k.jsonl'
