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
print(chars)
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

for e in sorted(set([y[1] for y in lines if int(y[1]) < 19 ])):
    category = e
    all_categories.append(category)
    category_lines[category] = [x[0] for x in lines if x[1] == category and len(x[0]) > 0]

n_categories = len(all_categories)

#print all_categories
#print all_categories[0]
#print category_lines['1']
#print n_categories
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


import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
print n_letters
print n_hidden
print n_categories


#input = Variable(lineToTensor(u'유류비'))
#hidden = Variable(torch.zeros(1, n_hidden))

#output, next_hidden = rnn(input[0], hidden)
#print(output)

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

#for i in range(10):
#    category, line, category_tensor, line_tensor = randomTrainingExample()
#    print('cateogry = ', category, '/ line =', line)

criterion = nn.NLLLoss()

learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.initHidden()

    #print line_tensor.size()[0]
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    print output
    print category_tensor

    exit()

    loss = criterion(output, category_tensor)
    loss.backward()

    loss_before = loss.data[0]
    optimizer.step()

    if math.isnan(loss.data[0]):
        print loss_before
        pass

    #for p in rnn.parameters():
    #    p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

import time
import math

n_iters = 100000
print_every = 1000
plot_every = 1000



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
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    #print line_tensor
    #exit()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        guess1 = all_categories[guess_i[0][0]]
        guess2 = all_categories[guess_i[0][1]]
        guess3 = all_categories[guess_i[0][2]]

        correct = '✓' if  guess3 == category or guess1 == category or guess2 == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s %s %s' % (iter, iter / float(n_iters) * 100, timeSince(start), loss, line, guess1, guess2, guess3, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        print current_loss / plot_every
        #print output
        #print categoryFromOutput(output)
        current_loss = 0

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

#plt.figure()
#plt.plot(all_losses)

exit(0)

