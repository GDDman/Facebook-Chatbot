import tensorflow as tf
import numpy as np
import spacy
import random
import custom_dict

'''
File for training of testing the second corpus.
'''

from datasets.facebook2 import data
import data_utils

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

# load preprocessed data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/facebook2/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

# initialize model
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/facebook2/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

# split data into training, dev, and testing sets
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)

print('loading...')

# get word frequencies
words = []
wordcount = 0
f = open('datasets/facebook/words.txt', 'r')
for line in f:
    tokens = line.split(':')
    if wordcount == 0:
        wordcount = int(tokens[1].strip())
    else:
        words.append((tokens[0].strip(), tokens[1].strip()))

def randomsentence(length):
    sentence = ''
    for i in range(0, length):
        rand = random.randint(0, wordcount)
        for word in words:
            rand -= int(word[1])
            if rand <= 0:
                sentence += word[0] + ' '
                break
    return sentence.strip()

def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

# new training, continue training, test
mode = 'test'

if mode is 'newtraining':
    # save to ckpt
    sess = model.train(train_batch_gen, val_batch_gen)

elif mode is 'continue training':
    # get last session and continue training
    sess = model.restore_last_session()
    sess = model.train(train_batch_gen, val_batch_gen, sess)

elif mode is 'test':
    # get last session
    sess = model.restore_last_session()
    # get string
    input_, answers = test_batch_gen.__next__()
    output = model.predict(sess, input_)

    modelsim = 0
    usersim = 0
    randomsim = 0
    simcount = 0
    lines = []
    # get questions, real answers and model answers
    for ii, ai, oi in zip(input_.T, answers.T, output):

        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        a = data_utils.decode(sequence=ai, lookup=metadata['idx2w'], separator=' ')
        d = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ')
        d = custom_dict.translateSentence(d).strip()

        qarr = q.split(' ')
        aarr = a.split(' ')
        darr = d.split(' ')

        # random answer
        r = randomsentence(len(darr))
        r = custom_dict.translateSentence(r)

        # remove sentences containing unk
        if darr.count('unk') == 0 and aarr.count('unk') == 0 and qarr.count('unk') == 0:

            nlp = spacy.load('en')
            qsp = nlp(q)
            asp = nlp(a)
            dsp = nlp(d)
            rsp = nlp(r)

            # cosine similarity between model and real
            msim = asp.similarity(dsp)

            # get input and format sentence
            userans = input('\nQuestion: ' + q + '\n')
            userans = userans.lower()
            userans = filter_line(userans, EN_WHITELIST)

            # cosine similarity between
            uasp = nlp(userans)
            usim = asp.similarity(uasp)

            # random and real
            rsim = asp.similarity(rsp)

            #print('q : [{0}] \na : [{1}] \nout : [{2}] \nsim : [{3}] \n'.format(q, a, d, sim))

            # update total counts
            modelsim += msim
            usersim += usim
            randomsim += rsim
            simcount += 1

            # write to file
            lines.append('question: ' + q + '\n')
            lines.append('real answer: ' + a + '\n')
            lines.append('model answer: ' + d + '\n')
            lines.append('random answer: ' + r + '\n')
            lines.append('testing answer: ' + userans + '\n\n')

    if simcount is not 0:
        # get average simlarity over all questions
        print('# questions: ' + str(simcount))
        print('Model Similarity: ' + str(modelsim/simcount) + '\n')
        print('User Similarity: ' + str(usersim/simcount) + '\n')
        print('Random Similarity: ' + str(randomsim/simcount) + '\n')
        lines.append('Model Similarity: ' + str(modelsim/simcount) + '\n')
        lines.append('User Similarity: ' + str(usersim/simcount) + '\n')
        lines.append('Random Similarity: ' + str(randomsim/simcount) + '\n')
    else:
        print('No output. \n')

    f = open("datasets/facebook2/testing.txt", "w")
    f.writelines(lines)
    f.close()
