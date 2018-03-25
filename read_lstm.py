# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from io import open
import glob


#def findFiles(path): return glob.glob(path)


#print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + "0123456789<>()[]/+ .,;-'\"#%*/"


# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = [u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = [u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ', u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [u' ', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ', u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ', u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ']


all_letters = all_letters + ''.join(CHOSUNG_LIST)+ ''.join(JUNGSUNG_LIST) + ''.join(JONGSUNG_LIST)

chars = list(sorted(set(all_letters)))
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = { i:ch for i,ch in enumerate(chars)}
print(char_to_ix)
print(len(chars))
n_letters = len(chars)

#print(set(all_letters))

for e in chars:
    sys.stdout.write(unicode(e) + ' ')
    sys.stdout.flush()
#n_letters = len(all_letters)
#print (n_letters)



# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line.split('\t') for line in lines]

lines = readLines('data/trainu.txt')
#print(lines)

# 한글은 아래와 같은 방법으로 유니코드로 조합된다.
ch = BASE_CODE + (0 * CHOSUNG + 2 * JUNGSUNG)
print '한글 : {}  /  유니코드 : {}'.format(ch, unichr(ch))

category_lines = {} # dictionary, a list of text per account
all_categories = []

# BASE_CODE(4403244) 제거

for e in sorted(set([y[1] for y in lines if int(y[1]) < 199 ])):
    category = e
    all_categories.append(category)
    category_lines[category] = [x[0] for x in lines if x[1] == category and len(x[0]) > 0]

n_categories = len(all_categories)

#print all_categories
#print all_categories[0]
#print category_lines['1']
print n_categories
#print (category_lines['24'][:5])

import torch


def lineToTensor(line):

    res = []
    #print line
    for letter in line:

        if ord(letter) >= 44032 and ord(letter) <= 55199:
            # 시작: 44032, 끝: 55199
            # print(charTemp)
            cBase = ord(letter) - BASE_CODE

            c1 = cBase / CHOSUNG
            # print '초성 : {}  /  유니코드 : {}'.format(CHOSUNG_LIST[c1], unichr(c1))
            # print CHOSUNG_LIST[c1]
            #res.append(CHOSUNG_LIST[c1])
            try:
                res.append(char_to_ix[CHOSUNG_LIST[c1]])
            except KeyError:
                # print res
                pass

            c2 = (cBase - (CHOSUNG * c1)) / JUNGSUNG
            # print '중성 : {}  /  유니코드 : {}'.format(JUNGSUNG_LIST[c2], unichr(c2))
            # print JUNGSUNG_LIST[c2]
            #res.append(JUNGSUNG_LIST[c2])
            try:
                res.append(char_to_ix[JUNGSUNG_LIST[c2]])
            except KeyError:
                # print res
                pass

            c3 = (cBase - (CHOSUNG * c1) - (JUNGSUNG * c2))
            # print '종성 : {}  /  유니코드 : {}'.format(JONGSUNG_LIST[c3], unichr(c3))
            # print JONGSUNG_LIST[c3]
            #res.append(JONGSUNG_LIST[c3])
            try:
                res.append(char_to_ix[JONGSUNG_LIST[c3]])
            except KeyError:
                # print res
                pass
        else:
            try:
                res.append(char_to_ix[letter])
            except KeyError:
                # print res
                pass

    tensor = torch.zeros(len(res), 1, n_letters)
    for li, letter in enumerate(res):
        tensor[li][0][res[li]] = 1
    #print res
    return tensor


#print(lineToTensor(u'중학교').size())
#print(lineToTensor(u'중학교'))


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, n_hidden)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(n_hidden, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

    def forward(self, line):
        out, self.hidden = self.lstm(line.view(len(line), 1, -1), self.hidden)
        class_space = self.hidden2class(out.view(len(line), -1))
        class_scores = F.log_softmax(class_space)
        return class_scores


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(5) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], top_i
    #return all_categories[category_i], top_i

#print(categoryFromOutput(output))

import random

def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    #print category, line_tensor.size()
    return category, line, category_tensor, line_tensor



n_hidden = 128
model = LSTMClassifier(n_letters, n_hidden, n_categories)

# Load the best saved model.
with open('./adam/model_100000.lstm', 'rb') as f:
    model = torch.load(f)

loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters())

#inputs = Variable(lineToTensor(u'유류비'))
#class_scores = model(inputs)
#print class_scores[-1:]
#guess, guess_i = categoryFromOutput(class_scores[-1:])

#print guess


import time
import math

n_iters = 100000
print_every = 100
plot_every = 1000
save_every = 10000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
for iter in range(1, n_iters+1):

    # print epoch
    category, line, category_tensor, line_tensor = randomTrainingExample()

    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.

    # Step 3. Run our forward pass.
    class_scores = model(line_tensor)

    # print class_scores[-1:]
    # print category_tensor

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(class_scores[-1:], category_tensor)
    loss.backward()
    optimizer.step()


    #category, line, category_tensor, line_tensor = randomTrainingExample()
    #output, loss = train(category_tensor, line_tensor)
    current_loss += loss.data[0]

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(class_scores[-1:])
        guess1 = all_categories[guess_i[0][0]]
        guess2 = all_categories[guess_i[0][1]]
        guess3 = all_categories[guess_i[0][2]]

        #correct = '✓' if  guess3 == category or guess1 == category or guess2 == category else '✗ (%s)' % category
        correct = '✓' if guess == category else '✗ (%s)' % category

        print('%d %d%% (%s) %.4f %s / %s %s %s %s %s' % (iter, iter / float(n_iters) * 100, timeSince(start), loss.data[0], line, guess, guess1, guess2, guess3, correct))

        #print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / float(n_iters) * 100, timeSince(start), loss.data[0], line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        print current_loss / plot_every
        #print output
        #print categoryFromOutput(output)
        current_loss = 0

    if iter % save_every == 0:
        file_name = "model_{}.lstm".format(iter)
        with open(file_name, 'wb') as f:
            torch.save(model, f)

exit(0)

