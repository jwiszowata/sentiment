import pandas as pd
import re
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

def data_cleaner(text):
    try:
	    text_for_cleaning = " ".join(text)
	    lower_case = text_for_cleaning.lower()
	    letters_only = re.sub(
	        "[^AaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpQqRrSsŚśTtUuVvWwYyZzŹźŻż]", 
	        " ", 
	        lower_case)
	    tokens = tokenizer.tokenize(letters_only)
	    return tokens
    except:
        return 'NC'

def ingest_train(path):
    data = pd.read_csv(path, encoding='utf-8', sep='\t')
    data = data[data.sentiment.isnull() == False]
    data['sentiment'] = data['sentiment'].map(int)
    data = data[data['tokens'].isnull() == False]
    data['tokens'] = data['tokens'].map(eval)
    data['tokens'] = data['tokens'].map(data_cleaner)
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    data = data.sort_values(by=['id'])
    return data

train_data = ingest_train('./Data - sentiment classification.csv')

def subseries(words_list):
	return  [tuple(words_list[i:j]) for i in range(len(words_list)) for j in range(i + 1, len(words_list) + 1)]