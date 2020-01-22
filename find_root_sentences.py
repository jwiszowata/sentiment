import pandas as pd
from utils import ingest_train, subseries

# Import data
train_data = ingest_train('./Data - sentiment classification.csv')
train_set = set()

# Sort by length
train_data.insert(3, 'words', train_data['tokens'].map(len), allow_duplicates = False)
train_data = train_data.sort_values(by=['words'])

# Collect sentences which are not subsentences, I called the roots
def check_subseries(l):
	for s in subseries(l):
		if s in train_set:
			train_set.remove(s)
	if l != []:
		train_set.add(tuple(l))

train_data['tokens'].map(check_subseries)
print(len(train_set))

# Insert root info to the data
def find_roots(data):
	if tuple(data) in train_set:
		train_set.remove(tuple(data))
		return True
	return False

train_data.insert(4, 'roots', train_data['tokens'].map(find_roots), allow_duplicates = False)
print(sum(train_data['roots']))

# Save created data frame to file
train_data.to_csv(index=False, path_or_buf="./with_roots_new.csv", sep='\t')
