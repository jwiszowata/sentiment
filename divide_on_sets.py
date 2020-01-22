import pandas as pd
from sklearn.model_selection import train_test_split
from utils import subseries, ingest_train

# Import data
train_data = ingest_train('./with_roots_new.csv')

# Get only root sentences
def get_data(train_data, roots_val):
	y = train_data['sentiment']
	y = y[train_data['roots'] == roots_val]
	X = train_data[train_data['roots'] == roots_val]
	return X, y

X, y = get_data(train_data, roots_val=True)

# Split root sentences between train, validation and test sets
SEED = 2000

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, 
                                                                test_size=.2, random_state=SEED)
print("train: ", len(x_train), 
      ", classes: ", sum(y_train == -1), sum(y_train == 0), sum(y_train == 1),  
      ", validation: ", len(x_validation), 
      ", classes: ", sum(y_validation == -1), sum(y_validation == 0), sum(y_validation == 1),
      ", test: ", len(x_test),
      ", classes: ", sum(y_test == -1), sum(y_test == 0), sum(y_test == 1))

# Put all not root data into set
X_n, y_n = get_not_roots(train_data)

not_root_set = set()
X_n['tokens'].map(lambda x: not_root_set.add(tuple(x)))

# Put not root data into proper set
# if sentence B is a root sentence in train set and sentence A is subsentence of sentence B,
# then A should also be in train set
def distribute(data):
	chosen_set = set()
	for d in data:
		sub = subseries(d)
		for s in sub:
			if s in not_root_set:
				chosen_set.add(s)
				not_root_set.remove(s)
	return chosen_set

train_set = distribute(x_train['tokens'])
validation_set = distribute(x_validation['tokens'])
test_set = distribute(x_test['tokens'])

x_train['tokens'].map(lambda x: train_set.add(tuple(x)))
x_validation['tokens'].map(lambda x: validation_set.add(tuple(x)))
x_test['tokens'].map(lambda x: test_set.add(tuple(x)))

print(len(train_set), len(validation_set), len(test_set))

# Insert set info to the data
def find_set_type(data):
	if tuple(data) in train_set:
		return 0
	if tuple(data) in validation_set:
		return 1
	if tuple(data) in test_set:
		return 2

train_data.insert(5, 'set_type', train_data['tokens'].map(find_set_type), allow_duplicates = False)

# Save created data frame to file
train_data.to_csv(index=False, path_or_buf="./divided_on_sets.csv", sep='\t')






