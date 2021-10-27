import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))
import glob
import click
import pandas as pd
from constants import PERSPECTIVE_API_ATTRIBUTES_LOWER, OPENAI_API_KEY
<<<<<<< HEAD
from perspective_api import PerspectiveWorker, unpack_scores, collate, make_generations_col
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed
=======
<<<<<<< Updated upstream
from perspective_api import PerspectiveWorker, unpack_scores
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed, collate, make_generations_col
=======
from perspective_api import PerspectiveWorker, unpack_scores, collate, make_generations_col
from utils_fn import load_jsonl, batchify, ensure_dir, set_seed
>>>>>>> Stashed changes
>>>>>>> minseon
from typing import Optional
from pathlib import Path
import time


@click.command()
<<<<<<< HEAD
@click.argument('output_dir')
@click.option('--dataset_file', required=False, type=str,
=======
<<<<<<< Updated upstream
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
>>>>>>> minseon
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--perspective_rate_limit', default=100)
@click.option('--num_sentences', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option("--random_state", type=int, default=1)

def main(output_dir: str, dataset_file: Optional[str],
        perspective_rate_limit: int, num_sentences: int, random_state:int):
    output_dir = Path(output_dir)
    dataset = pd.read_json(dataset_file, lines=True)
<<<<<<< HEAD
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

=======
=======
@click.argument('output_dir')
@click.option('--dataset_file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--perspective_rate_limit', default=100)
@click.option('--num_sentences', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option("--random_state", type=int, default=1)

def main(output_dir: str, dataset_file: Optional[str],
        perspective_rate_limit: int, num_sentences: int, random_state:int):
    output_dir = Path(output_dir)
    dataset = pd.read_json(dataset_file, lines=True)
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

>>>>>>> Stashed changes
>>>>>>> minseon
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
<<<<<<< Updated upstream
            time.sleep(perspective_rate_limit)
=======
            if i%100==0:
                time.sleep(3)
>>>>>>> Stashed changes
            i+=1
    perspective.stop()
    print('Finished generation and perspective scoring!')

    if os.path.exists(perspective_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
<<<<<<< Updated upstream
    main()
=======
    main()
>>>>>>> Stashed changes
