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
import csv
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluate the generated 25 sentences')
    parser.add_argument('--eval_files_or_dirs', nargs="+", help='generated eval file path')
    parser.add_argument('--num_samples', default=25, type=int, help='num of generated sentence')
    args = parser.parse_args()

    all_dfs = []

    for file_or_dir in args.eval_files_or_dirs:
        if  ".jsonl" in file_or_dir:
            all_dfs.append(pd.read_json(file_or_dir, lines=True))

        else:
            frames = []
            print(sorted(glob.glob(str(file_or_dir) + "/prompted*.jsonl")))
            for file in sorted(glob.glob(str(file_or_dir) + "/prompted*.jsonl")):
                print(file)
                frames.append(pd.read_json(file, lines=True))
            if len(frames)>1:
                df = pd.concat(frames)
            else:
                df = frames[0]
            df = df.reset_index(drop=True)
            all_dfs.append(df)
            print(df.count())

    compare_results = {}
    for df_idx in range(len(all_dfs)):
        toxicities = []

        for row_idx, row in enumerate(all_dfs[df_idx].iterrows()):

            try:
                generations = row[1]['generations']
            except:
                generations =  row[1]['gen_continuation']

            max_toxicity = float('-inf')
            max_gen_idx = 0
            #print(len(generations))
            for gen_idx, gen in enumerate(generations):
                if gen['toxicity'] is not None:
                    if max_toxicity < gen['toxicity']:
                        max_gen_idx = gen_idx
                        max_toxicity = gen['toxicity']
            if max_toxicity > 0:
                toxicities.append(max_toxicity)


            if row_idx not in compare_results:
                try:
                    compare_results[row_idx] = {
                        'filename': row[1]['filename'],
                        'prompt': row[1]['prompt']['text'],
                        '{}-text'.format(df_idx): generations[max_gen_idx]['text'],
                        '{}-toxicity'.format(df_idx): float(generations[max_gen_idx]['toxicity'])
                    }
                except:
                    compare_results[row_idx] = {
                        'filename': row[1]['filename'],
                        'prompt': row[1]['prompt']['text'],
                        '{}-text'.format(df_idx): generations[max_gen_idx]['text'],
                        '{}-toxicity'.format(df_idx): float('-1')
                    }
            else:
                try:
                    compare_results[row_idx]['{}-text'.format(df_idx)] = generations[max_gen_idx]['text']
                    compare_results[row_idx]['{}-toxicity'.format(df_idx)] = float(generations[max_gen_idx]['toxicity'])
                except:
                    compare_results[row_idx]['{}-text'.format(df_idx)] = generations[max_gen_idx]['text']
                    compare_results[row_idx]['{}-toxicity'.format(df_idx)] = float('-1')
        print(np.mean(toxicities))
    csv_columns = ['filename', 'prompt']
    for df_idx in range(len(all_dfs)):
        csv_columns+=['{}-text'.format(df_idx)]
        csv_columns+=['{}-toxicity'.format(df_idx)]
    #print(len(compare_results.keys()))
    with open('compare_results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for row_idx in compare_results.keys():
            #print(compare_results[row_idx])
            writer.writerow(compare_results[row_idx])
