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

# constants
punctuation = re.compile('[{}]'.format(re.escape(string.punctuation)))

# create bag
def to_bag(counters, k=None, stop_words=None):
    # sorted list of top-k unique words
    # Excludes words included in `stop_words`
    # k = if specified, only the top-k words are returned
    # stop_words = a collection of words to be ignored when populating the bag
    bag = Counter()
    for counter in counters:
        bag.update(counter)

    if stop_words is not None:
        for word in set(stop_words):
            bag.pop(word, None)  # if word not in bag, return None
    return list(sorted(i for i,j in bag.most_common(k)))

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
def generate_word_by_context(codes, max_vocab_words=1000, max_context_words=1000, context_size=2, weight_by_distance=False):
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

# SVD functions
def reduce(X, n_components, power=0.0):
    U, Sigma, VT = randomized_svd(X, n_components=n_components)
    # note: TruncatedSVD always multiplies U by Sigma, but can tune results by just using U or raising Sigma to a power
    return U * (Sigma**power)

# get embedding
def get_embedding(dictionary, text):
    """ Returns the word embedding for a given word, reshaping the word embedding array. """
    return dictionary[text][:,np.newaxis]

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

def train_nn(x_train, bias):
    # get input and output, or just input if it's not training
    # x_train = article text, in string format
    # y_train = an array of two values: a score for left leaning, and a score for right-leaning
    left_val = 0
    right_val = 0
    if bias < 0:
        left_val = (-bias) / 5
    elif bias > 0:
        right_val = bias / 5
    x_train = x_train.lower()
    y_train = np.ndarray((250,2))
    y_train[:,0] = left_val
    y_train[:,1] = right_val

    # get stopwords (to remove)
    with open("stopwords.txt", 'r') as r:
        stops = []
        for line in r:
            stops += [i.strip() for i in line.split('\t')]

    x_count = Counter(punctuation.sub('', x_train).lower().split())
    x_bag = to_bag([x_count], k=250, stop_words=stops)
    for i in range(250 - len(x_bag)):
        x_bag.append("")

    # preprocessing part 2: get the word embeddings, add the bag of words to the word embeddings
    # turn words into code
    if Path("word_vectors_200.pkl").is_file():
        with open("word_vectors_200.pkl", mode="rb") as opened_file:
            x_dict = pickle.load(opened_file)
    else:
        x_dict = dict()

    x_sorted = sort_words(x_bag)
    x_codes = words_to_code(x_bag, x_sorted)
    x_contexts = generate_word_by_context(x_codes, max_vocab_words=50000, max_context_words=5000, context_size=4, weight_by_distance=True)
    x_log = np.log10(1 + x_contexts, dtype="float32")
    x_vectors = reduce(x_log, n_components=200)

    for i in range(len(x_bag)):
        x_dict[x_bag[i]] = x_vectors[i,:]

    with open("word_vectors_200.pkl", mode="wb") as opened_file:
        pickle.dump(x_dict, opened_file)

    # pull the model, else make a new one
    if Path("model.pt").is_file():
        model = torch.load("model.pt")
    else:
        model = Model(200,250,2)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    arr = np.ones((len(x_bag), 1, 200)) / 10000
    for i, x in enumerate(x_bag):
        try:
            arr[i] += x_dict[x]
        except Exception as e:
            continue

    arr = torch.tensor(np.swapaxes(arr, -2, -1)).float()
    prediction = model.forward(arr)
    truth = torch.tensor(y_train).float()
    loss = binary_cross_entropy(prediction, truth)
    loss.backward()
    acc = accuracy(prediction, truth)
    optimizer.step()
    optimizer.zero_grad()
    print("accuracy: " + str(acc))

    predicted_left_bias = prediction.detach().numpy()[:,0]
    predicted_right_bias = prediction.detach().numpy()[:,1]
    print("left actual bias: " + str(np.mean(truth.detach().numpy()[:,0])*5))
    print("right actual bias: " + str(np.mean(truth.detach().numpy()[:,1])*5))
    print("left predicted bias: " + str(np.mean(predicted_left_bias)*5))
    print("right predicted bias: " + str(np.mean(predicted_right_bias)*5))

    torch.save(model, "model.pt")
    return True
