import json
import argparse
from glob import glob
from tqdm import tqdm
import pandas as pd
import json
import pdb
from evaluate_toxicity import compute_toxicity
from detoxify import Detoxify
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluate the generated 25 sentences')
    parser.add_argument('--eval_file', default='/v6/myung/iclr/purify_lm/generations/toxicity/dexperts/small_experts/prompted_gens_dexperts.jsonl', type=str, help='generated eval file path')
    parser.add_argument('--num_samples', default=10, type=int, help='num of generated sentence')
    args = parser.parse_args()
    #import json
    #with open(args.eval_file, 'r') as datafile:
    #    data = json.load(datafile)
    #df = pd.DataFrame(data)
    df = pd.read_json(args.eval_file, lines=True)
    res = compute_toxicity(df)
    print(args.eval_file)
    print("Toxicity Prob")
    print('toxic {:0.3f}'.format(res['toxic']['toxicity_probability']))
    print('nontoxic {:0.3f}'.format(res['nontoxic']['toxicity_probability']))
    print('full: {:0.3f}'.format(res['full']['toxicity_probability']))

    print('Expected Max Toxicity')
    print('toxic {:0.3f} {:0.3f}'.format(res['toxic']['avg_max'], res['toxic']['std_max']))
    print('nontoxic {:0.3f} {:0.3f}'.format(res['nontoxic']['avg_max'], res['nontoxic']['std_max']))
    print('full: {:0.3f} {:0.3f}'.format(res['full']['avg_max'], res['full']['std_max']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_C = Detoxify('original', device=device)
    file_path, jsonl=args.eval_file.rsplit('.',1)
    fw = open(file_path+"_BERT"+'.jsonl', 'w')
    num_of_rows = df.shape[0]
    prompt = pd.json_normalize(df['prompt'])
    continuation = pd.json_normalize(df['continuation'])
    for row_idx in tqdm(range(num_of_rows)):
        fw_gen_texts = []
        for num_idx in range(args.num_samples):
            gen = df.generations[row_idx][num_idx]['text']
            gen_result = model_C.predict(gen)
            fw_gen_texts += [{'text': gen}]
            fw_gen_texts[-1].update({k: float(v) for k, v in gen_result.items()})

        print(json.dumps({
            'filename': df.filename[row_idx],
            'prompt': {col: str(prompt[col][row_idx]) \
                        if col == 'text' \
                        else float(prompt[col][row_idx]) \
                        for col in prompt.columns},
            'continuation': {col: str(continuation[col][row_idx]) \
                        if col == 'text' \
                        else float(continuation[col][row_idx]) \
                        for col in continuation.columns},
            'gen_continuation':fw_gen_texts}), file=fw)
        fw.flush()
    fw.close()

    df = pd.read_json(file_path+"_BERT"+'.jsonl', lines=True)
    res = compute_toxicity(df)
    print(args.eval_file)
    print("=====BERT based Classifier=====")
    print("Toxicity Prob")
    print('toxic {:0.3f}'.format(res['toxic']['toxicity_probability']))
    print('nontoxic {:0.3f}'.format(res['nontoxic']['toxicity_probability']))
    print('full: {:0.3f}'.format(res['full']['toxicity_probability']))

    print('Expected Max Toxicity')
    print('toxic {:0.3f} {:0.3f}'.format(res['toxic']['avg_max'], res['toxic']['std_max']))
    print('nontoxic {:0.3f} {:0.3f}'.format(res['nontoxic']['avg_max'], res['nontoxic']['std_max']))
    print('full: {:0.3f} {:0.3f}'.format(res['full']['avg_max'], res['full']['std_max']))
