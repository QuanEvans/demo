import pandas as pd
import numpy as np
import os
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

# Extracting folder

import requests

filename = 'aclImdb_v1.tar.gz' 
# url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename 
# r = requests.get(url)
# with open(filename, 'wb') as f: f.write(r.content)

#...extract zip file
import tarfile

tar = tarfile.open(filename, mode='r')
tar.extractall()
tar.close()

def get_precision(y_pred, y_test, debug = False):

    # deal with npdarray

    y_pred = list(y_pred)

    y_test = list(y_test)


    y_pred = list(map(int,[1 == l for l in y_pred]))# deal with None type

    y_test = list(map(int,[1 == l for l in y_test]))# deal with None type
    
    n = len(y_pred);

    true_positive = sum(y_pred[i]* y_test[i] for i in range(n))

    if (0 == sum(y_pred)): return 0

    return true_positive*1.0/sum(y_pred)

def get_recall(y_pred, y_test):

    # deal with npdarray

    y_pred = list(y_pred)

    y_test = list(y_test)

    n = len(y_pred);

    y_pred = list(map(int,[1 == l for l in y_pred]))# deal with None type

    y_test = list(map(int,[1 == l for l in y_test]))# deal with None type

    true_positive = sum(y_pred[i]*y_test[i] for i in range(n))

    if 0 == sum(y_test): return 0

    return true_positive*1.0/sum(y_test)

def get_fscore(y_pred, y_test):


    precision=get_precision(y_pred,y_test)

    recall=get_recall(y_pred,y_test)

    if precision==0 and recall==0:

        return 0

    fscore=2.0*precision*recall/(precision+recall)

    return fscore

# Extracting Data in train test files

imdb_dir = 'aclImdb'
train_dir = os.path.join(imdb_dir,'train')
test_dir = os.path.join(imdb_dir,'test')
labels = []
texts = []

test_labels = []
test_texts = []

# Tagging data as positive and negative in train data

for label_type in ['pos','neg']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname),encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


# Tagging data as positive and negative in test data

for label_type in ['pos','neg']:
    dir_name = os.path.join(test_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname),encoding="utf8")
            test_texts.append(f.read())
            f.close()
            if label_type == 'neg':
                test_labels.append(0)
            else:
                test_labels.append(1)

print(f'Length of texts is {len(texts)}')
print(f'Length of labels id {len(labels)}')
print(f'Length of test_texts is {len(test_texts)}')
print(f'Length of test_labels is {len(test_labels )}')

texts_df = pd.DataFrame({'texts': texts,
                        'labels':labels})

texts_df.head()

positive = texts_df[texts_df['labels']==1]['texts']
negative = texts_df[texts_df['labels']==0]['texts']

X_train, X_test, y_train, y_test = train_test_split(texts_df.texts,texts_df.labels, test_size=0.3, random_state=0)
print(X_test.shape,y_test.shape,X_train.shape,y_train.shape)


from sklearn.dummy import DummyClassifier       #this classifier selects the most frequent class in train data and fit on test data

dummy_majority = DummyClassifier(strategy='most_frequent',random_state=0)
dummy_majority.fit(X_train,y_train)

y_pred = dummy_majority.predict(X_test)

#using function
print("Precison:",get_precision(y_pred,y_test))
print("Recall:",get_recall(y_pred,y_test))
print("FScore:",get_fscore(y_pred,y_test))

#Using sklearn
print("Precision: %0.2f" %precision_score(y_test, y_pred , average="macro"))
print("Recall:  %0.2f" %recall_score(y_test, y_pred , average="macro"))
print("F1-score:  %0.2f" %f1_score(y_test, y_pred , average="macro"))

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

texts_df['text_length'] = texts_df['texts'].str.split().str.len()  # creating new variable for text length

texts_df.head()

texts_df['text_length'].max()  # maximum length of review in data is 2470

texts_df['text_length'].min()   #min length of review in data is 10