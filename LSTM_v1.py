import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import csv
import torch
from torch import nn


# First read in all the data: training, dev, and test sentences; then the vocabulary
global first_tr_sentences
global second_tr_sentences
global first_dev_sentences
global second_dev_sentences
global first_tst_sentences
global second_tst_sentences
global vocab

def load_sentences(filename):
    lines = open(filename, "rb").readlines()
    sentences1 = []
    sentences2 = []
    for line in lines:
        sentences = line.decode('utf-8', errors='replace').split('\t')
        sentences1.append(sentences[0])
        sentences2.append(sentences[1].strip('\n'))
    return sentences1, sentences2

def load_vocab():
    lines = open("bobsue.voc.txt", "rb").readlines()
    vocab = []
    for line in lines:
        word = line.decode('utf-8', errors='replace').strip()
        vocab.append(word)
    return vocab

first_tr_sentences, second_tr_sentences = load_sentences('bobsue.seq2seq.train.tsv')
first_dev_sentences, second_dev_sentences = load_sentences('bobsue.seq2seq.dev.tsv')
first_tst_sentences, second_tst_sentences = load_sentences('bobsue.seq2seq.test.tsv')
vocab = load_vocab()
########### DATA LOADING COMPLETED ##############################

## Functions for word to vector conversion: uniform random and GloVe

# Convert words to dictionary of vectors. "file" contains words 
# vec_length is desired vector length
def words_to_vec_rand(file,vec_length):
    with open(file, encoding="utf8" ) as f:
       content = f.readlines()
    word_dict = {}
    for word in content:
        word = word.strip('\n\r')
        embedding = np.random.rand(vec_length)
        word_dict[word] = embedding
    # print ("Done.",len(word_dict)," words loaded!")  # For debugging
    return word_dict

# Form word dictionary using a glove file. "file" is the glove file;
# NOTE: The glove file MUST BE IN THE PROJECT DIRECTORY and the
# vector length DEPENDS ON the glove file being used. We will use
# the glove file glove.6B.200d.txt, hence a vector length of 200.
# The function takes about a minute to run.
# NOTE THAT glove words are all LOWERCASE, so word MUST BE CONVERTED 
# TO LOWERCASE (using word.lower()) before lookup in the glove dictionary
def words_to_vec_glove(glovefile):
    print("Loading Glove Model")
    f = open(glovefile,encoding="utf8")
    word_dict = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        word_dict[word] = embedding
    print("Done.",len(word_dict)," words loaded!")
    return word_dict


global randvec_dict
global glovevec_dict
randvec_dict =  words_to_vec_rand("bobsue.voc.txt",200)  
glovevec_dict = words_to_vec_glove("glove.6B.200d.txt")
################ Word to vector processing COMPLETED ##############

# Now define a few key RNN/LSTM-related functions

def sigmoid(x):
    return ((1.0)/(1.0 + np.exp(-x)))

def tanh(x):
    return ((np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)))

# This function performs the LSTM processing, taking a word vector 
def LSTM(word_vector, hsize):
    #hidden = torch.zeros(1, hsize)
    word_vec = torch.from_numpy(word_vector)   # converts numpy array to torch tensor
    #word_vec = torch.cat((word_v[None, :], hidden), dim=1)
    iWgt = nn.Linear(hsize * 2, hsize)
    gWgt = nn.Linear(hsize * 2, hsize)
    fWgt = nn.Linear(hsize * 2, hsize)
    oWgt = nn.Linear(hsize * 2, hsize)
    ctx = torch.zeros(1, hsize)
    f = sigmoid(fWgt(word_vec))
    i = sigmoid(iWgt(word_vec))
    g = tanh(gWgt(word_vec))
    o = sigmoid(oWgt(word_vec))
    c1 = ctx * f
    c2 = i * g
    new_ctx = c1 + c2
    new_hidden = o * tanh(new_ctx)
    return new_hidden, (new_hidden, new_ctx)
