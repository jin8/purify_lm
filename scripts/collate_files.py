import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))
import glob
from pathlib import Path
import json
import click
import pandas as pd
from constants import PERSPECTIVE_API_ATTRIBUTES_LOWER, OPENAI_API_KEY
from perspective_api import PerspectiveWorker, unpack_scores
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed, collate, make_generations_col



@click.command()
@click.argument('output-dir')
@click.option("--num_examples", type=int, default=10000)
@click.option("--random_state", type=int, default=1)
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--num_sentences', default=10, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--batch-size', default=10)

def main(output_dir: str, num_examples:int, random_state:int, dataset_file: str,
        num_sentences: int, batch_size:int):

    dataset = pd.read_json(dataset_file, lines=True)
    if num_examples > 0:
        dataset = dataset.sample(n=num_examples, random_state=random_state).reset_index(drop=True)
    else:
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    prompts = pd.json_normalize(dataset['prompt'])['text']


    output_file = output_dir + "prompted_gens.jsonl"

    perspectives = []
    for file in glob.glob(output_dir + "perspective_gpu*.jsonl"):
        with open(file) as f:
            for line in f:
                perspectives.append(json.loads(line))

    generations = []
    i = 0
    for file in glob.glob(output_dir + "generations_gpu*.jsonl"):
        for gen in enumerate(load_jsonl(file)):
            generations.append(gen)

    print('Collating output files')
    collate(dataset, generations, perspectives, output_file)


if __name__ == '__main__':
    main()
