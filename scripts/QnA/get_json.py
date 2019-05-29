from tqdm import tqdm
import json
import csv

path = "../data/teach_data/data.csv"


colnames = ["id","question_answer_or_comment", 
			"is_best_answer","topic_id", "parent_id",
			"votes", "titre", "message", "member", 
			"category", "state","is_solved",  
			"num_answers","country", "date",
			"last_answer_date", "auteur_crc", 
			"visits"]

thres = len(colnames)

my_json = {}

i = 0
err_count = 0

with open(path, newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in tqdm(reader):
		if len(row) == thres:
			my_json[i] = {
				colnames[col]:row[col] for col in range(thres)
			}
			i +=1
		else:
			err_count +=1
			continue

print(err_count)

SAVE=False

if SAVE:
	with open('../data/teach_data/data.json', 'w') as file:
		json.dump(my_json, file)

