import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))
import glob
import click
import pandas as pd
from constants import PERSPECTIVE_API_ATTRIBUTES_LOWER, OPENAI_API_KEY
from perspective_api import PerspectiveWorker, unpack_scores
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed, collate, make_generations_col
from typing import Optional
from pathlib import Path
import time


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--perspective-rate-limit', default=1)
@click.option('--num_sentences', default=25, help='Number of samples to generate for each prompt. When used with --eos')

def main(output_dir: str, dataset_file: Optional[str],
        perspective_rate_limit: int, num_sentences: int):
    output_dir = Path(output_dir)
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])['text']
    perspective_file = output_dir / 'perspective.jsonl'
    output_file = output_dir / "prompted_gens.jsonl"
    try:
        os.remove(perspective_file)
        os.remove(output_file)
    except:
        pass

    perspective = PerspectiveWorker(
    out_file=perspective_file,
    total=len(prompts) * num_sentences,
    rate_limit=perspective_rate_limit
    )

    generations = []
    i = 0
    print(sorted(glob.glob(str(output_dir) + "/generations_gpu*.jsonl")))
    for file in sorted(glob.glob(str(output_dir) + "/generations_gpu*.jsonl")):
        for i, gen in enumerate(load_jsonl(file)):

            print(gen)
            generations.append(gen)
            perspective(f'generation-{i}', gen)
            time.sleep(perspective_rate_limit)
            i+=1
    perspective.stop()
    print('Finished generation and perspective scoring!')

    if os.path.exists(perspective_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()