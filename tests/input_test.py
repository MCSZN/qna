# input_test.py

import json
from tqdm import tqdm
from pprint import pprint

path = "../data/teach_data/data.json"

with open(path, 'r') as f:
	data = json.load(f)

def clear(dictionnary):
    temp = dictionnary
    to_delete = []
    for key,val in tqdm(temp.items()):
        try:
            int(val['auteur_crc'])
            int(val['category'])
            int(val['date'])
            int(val['id'])
            int(val['is_best_answer'])
            int(val['is_solved'])
            int(val['last_answer_date'])
            int(val['member'])
            int(val['num_answers'])
            int(val['parent_id'])
            int(val['topic_id'])
            int(val['visits'])
            int(val['votes'])
        except:
            to_delete.append(key)
            continue
        if val['question_answer_or_comment'] not in ['C', 'A', 'Q']:
        	to_delete.append(key)
        	continue
    for key in to_delete:
        del temp[key]
    return temp

new_data = clear(data)

print('Unclear dataset nlines: ', len(data.keys()))
print('Clear dataset :', len(new_data.keys()))