"""
construct toxicity data from Jigsaw used to finetune nontoxic expert and toxic anti-expert in DExperts
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

data_dir = '/v3/minseon/Data/dexperts/datasets/jigsaw-unintended-bias-in-toxicity-classification'
jigsaw_df = pd.read_csv(f'{data_dir}/toxic_type_all_data.csv')
attributes = ['toxicity', 'severe_toxicity', 'identity_attack', 'insult', 'threat', 'obscene', 'sexual_explicit']

fos = defaultdict(dict)
for a in attributes:
    fos[a]['toxic'] = open(f'{data_dir}/toxic_type_{a}_gte0.5.txt', 'w')
    fos[a]['nontoxic'] = open(f'{data_dir}/toxic_type_{a}_eq0.txt', 'w')
fos['total']['toxic'] = open(f'{data_dir}/toxic_type_total_gte0.5.txt', 'w')
fos['total']['nontoxic'] = open(f'{data_dir}/toxic_type_total_eq0.txt', 'w')
comments_ct = {a: {'gte50': 0, 'eq0': 0} for a in attributes}

for i, row in tqdm(jigsaw_df.iterrows(), total=len(jigsaw_df.index)):
    flag = False
    toxic_labels = {'Top1':None, 'Top1Score':-1, 'Top2':None, 'Top2Score':-1}
    for a in attributes:
        if row[a] >= 0.5:
            flag = True
            if row[a]>toxic_labels['Top2Score']:
                toxic_labels['Top2Score'] = row[a]
                toxic_labels['Top2'] = a
                if toxic_labels['Top2Score'] > toxic_labels['Top1Score'] :
                    temp, temp_score = toxic_labels['Top1'], toxic_labels['Top1Score']
                    toxic_labels['Top1'], toxic_labels['Top1Score'] = toxic_labels['Top2'], toxic_labels['Top2Score']
                    toxic_labels['Top2'], toxic_labels['Top2Score'] = temp, temp_score

            fos[a]['toxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['gte50'] += 1
        if row[a] == 0.0:
            fos[a]['nontoxic'].write(f"{row['comment_text']}\n")
            comments_ct[a]['eq0'] += 1
    if flag:
        fos['total']['toxic'].write(f"{row['comment_text']}\t{toxic_labels['Top1']}\t{toxic_labels['Top1Score']}\t{toxic_labels['Top2']}\t{toxic_labels['Top2Score']}\n")

for a in attributes:
    fos[a]['toxic'].close()
    fos[a]['nontoxic'].close()
fos['total']['toxic'].close()
print(comments_ct)
