<<<<<<< Updated upstream
<<<<<<< Updated upstream
import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))

print(sys.path)

import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm
import os
import json
from generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3,contrastive_gpt2
from constants import PERSPECTIVE_API_ATTRIBUTES_LOWER, OPENAI_API_KEY
from perspective_api import PerspectiveWorker, unpack_scores
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed, collate, make_generations_col
import torch.distributed as dist
from detoxify import Detoxify

ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm', 'contrastive-gpt2']



@click.command()

@click.option("--local_rank", type=int, default=-1)
@click.option("--num_examples", type=int, default=-1)
@click.option("--random_state", type=int, default=1)

@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--toxic-model', type=str, default=None, help='Anti-expert for DExperts')
@click.option('--nontoxic-model', type=str, default=None, help='Expert for DExperts')
@click.option('--perspective-rate-limit', default=25)
@click.option('--num_sentences', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=16)
@click.option('--resume/--no-resume', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for dexperts')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
@click.option('--perspective-api/--no-perspective-api', default=False)

def main(local_rank:int, num_examples:int, random_state:int, output_dir: str, dataset_file: Optional[str],
         use_eos: bool, model: str, model_type: str, nontoxic_model: str,
         toxic_model: str, perspective_rate_limit: int, num_sentences: int, max_tokens: int, batch_size: int, resume: bool,
         alpha: float, filter_p: float, p: float, perspective_api: bool):
    print('perspective_api:',perspective_api)
    if local_rank >= 0:
        dist.init_process_group('nccl', rank=local_rank)
        torch.cuda.set_device(local_rank)
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        if num_examples > 0:
            dataset = dataset.sample(n=num_examples, random_state=random_state).reset_index(drop=True)
        else:
            dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']

    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-ensemble', 'gpt2-naughty-list', 'pplm']:
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    if local_rank >= 0:
        generations_file = output_dir / 'generations_gpu{}.jsonl'.format(local_rank)
        bert_eval_file = output_dir / 'bert_eval_gpu{}.jsonl'.format(local_rank)
    else:
        generations_file = output_dir / 'generations.jsonl'
        bert_eval_file = output_dir / 'bert_eval.jsonl'

    if local_rank>=0:
        dist.barrier()
    if local_rank in [-1,0]:
        assert resume or not os.path.exists(generations_file)   # don't overwrite generations!
        ensure_dir(output_dir)
    if local_rank >= 0:
        output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}_gpu{local_rank}.jsonl'
    else:
        output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'
    num_splits = len(prompts)//torch.cuda.device_count()
    if torch.cuda.device_count()-1 == local_rank:
        prompts = prompts[num_splits*local_rank:]
        dataset = dataset.iloc[num_splits*local_rank:]
        print(torch.cuda.device_count(), local_rank, num_splits*local_rank, num_splits*local_rank+len(prompts))
    else:
        prompts = prompts[num_splits*local_rank:(num_splits*(local_rank+1))]
        dataset = dataset.iloc[num_splits*local_rank:(num_splits*(local_rank+1))]
        print(local_rank, num_splits*local_rank, num_splits*(local_rank+1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = Detoxify('original', device=device)
    classifier.model.eval()

    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'gpt3':
        generations_iter = gpt3(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'dexperts':
        generations_iter = dexperts(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            alpha=alpha,
        )
    elif model_type == 'dexperts-gpt3':
        generations_iter = dexperts_gpt3(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            alpha=alpha,
        )
    elif model_type == 'pplm':
        generations_iter = pplm(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            class_label=0,
            stepsize=0.20,
            num_iterations=10,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'contrastive-gpt2':
        generations_iter = contrastive_gpt2(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')


    # Generate and collate perspective scores
    """
    if perspective_api:
        generations = []
        for i, gen in enumerate(generations_iter):
            generations.append(gen)
            perspective(f'generation-{i}', gen)
        torch.cuda.empty_cache()
        perspective.stop()
        print('Finished generation and perspective scoring!')

    else:
    """
    generations = []

    for i, gen in enumerate(generations_iter):
        generations.append(gen)
        row = {}
        result = classifier.predict(gen)
        row = {k: float(v) for k, v in result.items()}

        with bert_eval_file.open('a') as fw:
            print(json.dumps(row), file=fw)
            fw.flush()

    if os.path.exists(bert_eval_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(bert_eval_file), output_file)


if __name__ == '__main__':
    main()
=======
=======
>>>>>>> Stashed changes
import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))

print(sys.path)

import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm
import os
import json
<<<<<<< Updated upstream
from generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3, cont
=======
from generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3, cont, mapping, att, attadd
>>>>>>> Stashed changes
from constants import PERSPECTIVE_API_ATTRIBUTES_LOWER, OPENAI_API_KEY
from perspective_api import PerspectiveWorker, unpack_scores
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed
import torch.distributed as dist
from detoxify import Detoxify
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig
)
<<<<<<< Updated upstream
ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm', 'cont_gpt', 'cont_gpt_bert', 'cont_gpt_last']
=======
ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm', 'mapping','att','attadd','cont_gpt', 'add_cont_bert_gpt','cont_bert_gpt', 'cont_bert_gpt_only','add_cont_gpt_last','cont_gpt_last', 'cont_gpt_last_only']
>>>>>>> Stashed changes


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        try:
            if response['response']:
                response = unpack_scores(response['response'])[0]
            else:
                response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        except KeyError:
            pass

        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):

    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)


@click.command()

@click.option("--local_rank", type=int, default=-1)
@click.option("--num_examples", type=int, default=-1)
@click.option("--random_state", type=int, default=1)

@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--toxic-model', type=str, default=None, help='Anti-expert for DExperts')
@click.option('--nontoxic-model', type=str, default=None, help='Expert for DExperts')
@click.option('--perspective-rate-limit', default=25)
@click.option('--num_sentences', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=16)
@click.option('--resume/--no-resume', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for dexperts')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
@click.option('--perspective-api/--no-perspective-api', default=False)
@click.option('--config_name', default=None, type=str, help='Pretrained config name or path if not the same as model_name')
@click.option('--cache_dir', default=None, type=str, help='Where do you want to store the pretrained models downloaded from s3')

def main(config_name:Optional[str], local_rank:int, num_examples:int, random_state:int, output_dir: str, dataset_file: Optional[str],
         cache_dir:Optional[str], use_eos: bool, model: str, model_type: str, nontoxic_model: str,
         toxic_model: str, perspective_rate_limit: int, num_sentences: int, max_tokens: int, batch_size: int, resume: bool,
         alpha: float, filter_p: float, p: float, perspective_api: bool):

    if config_name:
        config = AutoConfig.from_pretrained(config_name, cache_dir=cache_dir)
    elif model:
        config = AutoConfig.from_pretrained(model, cache_dir=cache_dir)
    else:
        config = CONFIG_MAPPING[model_type]()
    config.cache_dir = cache_dir
    print('perspective_api:',perspective_api)
    if local_rank >= 0:
        dist.init_process_group('nccl', rank=local_rank)
        torch.cuda.set_device(local_rank)
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        if num_examples > 0:
            dataset = dataset.sample(n=num_examples, random_state=random_state).reset_index(drop=True)
        else:
            dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']

    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
<<<<<<< Updated upstream
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-ensemble', 'gpt2-naughty-list', 'pplm', 'cont_bert_gpt', 'cont_gpt', 'cont_gpt_last']:
=======
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-ensemble', 'gpt2-naughty-list', 'pplm','mapping','att','attadd', 'add_cont_bert_gpt','cont_bert_gpt','cont_bert_gpt_only', 'cont_gpt', 'add_cont_gpt_last','cont_gpt_last', 'cont_gpt_last_only']:
>>>>>>> Stashed changes
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    if local_rank >= 0:
        generations_file = output_dir / 'generations_gpu{}.jsonl'.format(local_rank)
        perspective_file = output_dir / 'perspective_gpu{}.jsonl'.format(local_rank)
    else:
        generations_file = output_dir / 'generations.jsonl'
        perspective_file = output_dir / 'perspective.jsonl'

    if local_rank>=0:
        dist.barrier()
    if local_rank in [-1,0]:
        assert resume or not os.path.exists(generations_file)   # don't overwrite generations!
        ensure_dir(output_dir)
    if local_rank >= 0:
        output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}_gpu{local_rank}.jsonl'
    else:
        output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'
    num_splits = len(prompts)//torch.cuda.device_count()
    if torch.cuda.device_count()-1 == local_rank:
        prompts = prompts[num_splits*local_rank:]
        dataset = dataset.iloc[num_splits*local_rank:]
        print(torch.cuda.device_count(), local_rank, num_splits*local_rank, num_splits*local_rank+len(prompts))
    else:
        prompts = prompts[num_splits*local_rank:(num_splits*(local_rank+1))]
        dataset = dataset.iloc[num_splits*local_rank:(num_splits*(local_rank+1))]
        print(local_rank, num_splits*local_rank, num_splits*(local_rank+1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if perspective_api:
        # Create perspective worker thread
        perspective = PerspectiveWorker(
            out_file=perspective_file,
            total=len(prompts) * num_sentences,
            rate_limit=perspective_rate_limit
        )
    else:
        classifier = Detoxify('original', device=device)
        classifier.model.eval()


    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'gpt3':
        generations_iter = gpt3(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif model_type == 'dexperts':
        generations_iter = dexperts(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            alpha=alpha,
        )
    elif model_type == 'dexperts-gpt3':
        generations_iter = dexperts_gpt3(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=nontoxic_model,
            antiexpert_name_or_path=toxic_model,
            out_file=generations_file,
            filter_p=filter_p,
            alpha=alpha,
        )
    elif model_type == 'pplm':
        generations_iter = pplm(
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            class_label=0,
            stepsize=0.20,
            num_iterations=10,
            model_name_or_path=model,
            out_file=generations_file
        )
<<<<<<< Updated upstream
=======
    elif 'mapping' in model_type:
        generations_iter = mapping(
            config=config,
            approach=model_type,
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif 'att' in model_type:
        generations_iter = att(
            config=config,
            approach=model_type,
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    elif 'attadd' in model_type:
        generations_iter = attadd(
            config=config,
            approach=model_type,
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )

>>>>>>> Stashed changes
    elif 'cont' in model_type:
        generations_iter = cont(
            config=config,
            approach=model_type,
            local_rank=local_rank,
            prompts=prompts,
            max_len=max_tokens,
            num_samples=num_sentences,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate and collate perspective scores
    if perspective_api:
        generations = []
        for i, gen in enumerate(generations_iter):
            generations.append(gen)
            perspective(f'generation-{i}', gen)
        torch.cuda.empty_cache()
        perspective.stop()
        print('Finished generation and perspective scoring!')

    else:
        generations = []

        for i, gen in enumerate(generations_iter):
            generations.append(gen)
            row = {}
            result = classifier.predict(gen)
            row = {k: float(v) for k, v in result.items()}
            num_cached_generations = 0

            with perspective_file.open('a') as fw:
                print(json.dumps(row), file=fw)
                fw.flush()

    if os.path.exists(perspective_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
