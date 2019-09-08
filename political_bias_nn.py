#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import codecs
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from numba import jit
import numpy as np
from pathlib import Path
import pickle
import re, string
from sklearn.decomposition import TruncatedSVD, randomized_svd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


# get input and output, or just input if it's not training
# x_train = article text, in string format
# y_train = an array of two values: a score for left leaning, and a score for right-leaning
x_train = ""
left_val = 0 / 5
right_val = 0 / 5
x_train = x_train.lower()
y_train = np.ndarray((250,2))
y_train[:,0] = left_val
y_train[:,1] = right_val


# In[3]:


# preprocessing part 1: take article text and form the bag of words
# remove punctuation
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(corpus):
    return punc_regex.sub('', corpus)

# create word counter
def to_counter(doc):
    """
    Produce word-count of document, removing all punctuation
    and removing all punctuation.

    Parameters
    ----------
    doc : str

    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    return Counter(strip_punc(doc).lower().split())

# create bag
def to_bag(counters, k=None, stop_words=None):
    """
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`

    Parameters
    ----------
    counters : Iterable[Iterable[str]]

    k : Optional[int]
        If specified, only the top-k words are returned

    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the bag
    """
    bag = Counter()
    for counter in counters:
        bag.update(counter)

    if stop_words is not None:
        for word in set(stop_words):
            bag.pop(word, None)  # if word not in bag, return None
    return list(sorted(i for i,j in bag.most_common(k)))

# get stopwords (to remove)
with open("stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

# run all the methods here
x_train = strip_punc(x_train)
x_count = to_counter(x_train)
x_bag = to_bag([x_count], k=250, stop_words=stops)
print(x_bag)
for i in range(250 - len(x_bag)):
    x_bag.append("")
print(len(x_bag))


# In[4]:


# preprocessing part 2: get the word embeddings, add the bag of words to the word embeddings
# turn words into code
if Path("word_vectors_200.pkl").is_file():
    with open("word_vectors_200.pkl", mode="rb") as opened_file:
        x_dict = pickle.load(opened_file)
else:
    x_dict = dict()

def sort_words(tokens):
    cnt = Counter(tokens)
    common = cnt.most_common()
    return list(tuple(zip(*common))[0])

def words_to_code(tokens, sorted_words):
    word2code = dict()
    for i, c in enumerate(sorted_words):
        word2code[c] = i
    return_list = list()
    for i in tokens:
        return_list.append(word2code[i])
    return return_list

@jit(nopython = True)
def generate_word_by_context(codes, max_vocab_words=1000, max_context_words=1000,
                             context_size=2, weight_by_distance=False):
    # initialize 2d array of zeros (with dtype=np.float32 to reduce required memory)
    # of shape (max_vocab_words, max_context_words)
    matrix = np.zeros((max_vocab_words, max_context_words), dtype=np.float32)

    # slide window along sequence and count "center word code" / "context word code" co-occurrences
    # Hint: let main loop index indicate the center of the window
    for i in range(context_size, len(codes) - context_size):
        if codes[i] < max_vocab_words:
            for word in range(1, context_size + 1):
                if codes[word + i] < max_context_words:
                    if weight_by_distance:
                        matrix[codes[i]][codes[word + i]] += 1.0 / np.abs(word)
                    else:
                        matrix[codes[i]][codes[word + i]] += 1.0
            for word in range(1, context_size + 1):
                if codes[i - word] < max_context_words:
                    if weight_by_distance:
                        matrix[codes[i]][codes[i - word]] += 1.0 / np.abs(word)
                    else:
                        matrix[codes[i]][codes[i - word]] += 1.0
    return matrix


# In[5]:


# SVD functions
def reduce(X, n_components, power=0.0):
    U, Sigma, VT = randomized_svd(X, n_components=n_components)
    # note: TruncatedSVD always multiplies U by Sigma, but can tune results by just using U or raising Sigma to a power
    return U * (Sigma**power)

# get embedding
def get_embedding(dictionary, text):
    """ Returns the word embedding for a given word, reshaping the word embedding array. """
    return dictionary[text][:,np.newaxis]

x_sorted = sort_words(x_bag)
x_codes = words_to_code(x_bag, x_sorted)
x_contexts = generate_word_by_context(x_codes,
                                      max_vocab_words=50000,
                                      max_context_words=5000,
                                      context_size=4,
                                      weight_by_distance=True)
x_log = np.log10(1 + x_contexts, dtype="float32")
x_vectors = reduce(x_log, n_components=200)

for i in range(len(x_bag)):
    x_dict[x_bag[i]] = x_vectors[i,:]

with open("word_vectors_200.pkl", mode="wb") as opened_file:
    pickle.dump(x_dict, opened_file)


# In[7]:


# create the model
class Model(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(dim_input, dim_hidden, 1, stride = 1)
        self.dense1 = nn.Linear(dim_hidden, dim_hidden)
        self.dense2 = nn.Linear(dim_hidden, dim_output)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = nn.functional.max_pool1d(x, (x.shape[-1],), 1)
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = torch.sigmoid(x)
        return x


# In[8]:


# pull the model, else make a new one
if Path("model.pt").is_file():
    model = torch.load("model.pt")
    model.eval()
else:
    model = Model(200,250,2)


# In[9]:


# define accuracy and binary cross entropy
def accuracy(predictions, truth):
    # np.mean(np.round(0.4) == 0)
    # >>> 1.0
    #tries to see how close predictions are to the truth
    #taking the mean will sum the amount of correct predictions and divide by total predictions returning accuracy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    return np.mean(1 - abs(predictions - truth.detach().numpy()))

def binary_cross_entropy(prediction, truth):
    # mg.mean is used is used for the summation and division by the number of summed terms
    # 1e-08 is used as a very small number to further prevent log(0)
    return torch.mean(truth * -torch.log(prediction + 1e-08) + (1 - truth) * -torch.log(1 - prediction + 1e-08))


# In[10]:


# if training = true: train
# EDIT LATER
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

arr = np.ones((len(x_bag), 1, 200)) / 10000
for i, x in enumerate(x_bag):
    try:
        arr[i] += x_dict[x]
    except Exception as e:
        continue

arr = torch.tensor(np.swapaxes(arr, -2, -1)).float()

# compute the predictions for this batch by calling on model
prediction = model.forward(arr)

# compute the true (a.k.a desired) values for this batch:
truth = torch.tensor(y_train).float()

# We pass in our truth data with a new axis. The reasoning is that our prediction is of shape (N,1)
# and our truth is of shape (N,) - adding the new axis standardizes the (N,1) shape.  Alternatively, you could
# squash them both down to shape (N,)
# compute the loss associated with our predictions(use softmax_cross_entropy)
loss = binary_cross_entropy(prediction, truth)

# back-propagate through your computational graph through your loss
loss.backward()

# Again, we want our prediction and truth data to be the same shape
# compute the accuracy between the prediction and the truth
acc = accuracy(prediction, truth)

# execute gradient descent by calling step() of optim
optimizer.step()

# null your gradients (please!)
optimizer.zero_grad()
print(acc)


# In[11]:


predicted_left_bias = prediction.detach().numpy()[:,0]
predicted_right_bias = prediction.detach().numpy()[:,1]
print(np.mean(truth.detach().numpy()[:,0])*5)
print(np.mean(truth.detach().numpy()[:,1])*5)
print(np.mean(predicted_left_bias)*5)
print(np.mean(predicted_right_bias)*5)


# In[12]:


# save model
torch.save(model, "model.pt")
