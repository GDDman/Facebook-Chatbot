from tkinter import *
import socket
import threading
from select import select
import json
import time
import random

import tensorflow as tf
import numpy as np

'''
A gui for talking to a trained model.
'''

# preprocessed data
from datasets.facebook2 import data
import data_utils
import FacebotGui as fb
import custom_dict

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

print('LOADING FACEBOT (Might take a minute or two)...')

# load data from pickle and npy files
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

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/facebook2/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

# get model
sess = model.restore_last_session()
w2idx = data.get_w2idx()

# get label for a word
def wordnet_pos(pos):

    if pos.startswith('V'):
        return 'verb'
    elif pos.startswith('N'):
        return 'noun'
    elif pos.startswith('PR'):
        return 'pronoun'
    else:
        return None

# called when user submits text
def enter_pressed(event):

    # print input to screen
    input_get = input_field.get()
    txt.configure(state='normal')
    txt.insert(END, "{}: {}\n".format('User', input_get))
    txt.yview(END)
    txt.configure(state='disabled')
    input_user.set('')

    # get output
    output = fb.get_output(input_get, sess, w2idx, model, metadata).strip()
    if output is None:
        output = 'Your input sentence is too long (it should be 20 words or less).'
    else:
        # format output
        temp = input_get
        temp = temp.replace(' i ', ' I ')
        temp = word_tokenize(temp.strip())
        temp = pos_tag(temp)
        # if an unk is found in output, replace it with a word from the input sentence
        # the word is selected randomly from nouns -> pronouns -> verbs
        nouns = []
        pronouns = []
        verbs = []
        for j, (w, pos) in enumerate(temp):
            pos = wordnet_pos(pos)
            if pos is 'noun':
                nouns.append(w)
            elif pos is 'pronoun':
                pronouns.append(w)
            elif pos is 'verb':
                verbs.append(w)
        replacement = ' unk '
        if nouns:
            replacement = ' ' + random.choice(nouns) + ' '
        elif pronouns:
            prn = random.choice(pronouns)
            if prn is 'i':
                prn = 'you'
            elif prn is 'you':
                prn = 'I'
            replacement = ' ' + prn + ' '
        elif verbs:
            replacement = ' ' + random.choice(verbs) + ' '

        output = output.replace(' unk ', replacement)
        output = output.replace('unk ', replacement[1:])
        output = output.replace(' unk', replacement[:-1])
        output = custom_dict.translateSentence(output).strip()
        # add period
        output = output + '.'
        output = output.lower()
        # uppercase first letter
        output = "%s%s" % (output[0].upper(), output[1:])

    # write output to screen
    txt.configure(state='normal')
    txt.insert(END, "{}: {}\n".format('Facebot', output))
    txt.yview(END)
    txt.configure(state='disabled')

    return "break"

# create GUI window
window = Tk()
window.title('Facebot')
txt_frm = Frame(window, width=450, height=300)
txt_frm.pack(fill="both", expand=False)
txt_frm.grid_propagate(False)
txt_frm.grid_rowconfigure(0, weight=1)
txt_frm.grid_columnconfigure(0, weight=1)
txt = Text(txt_frm, borderwidth=3, relief="sunken")
txt.config(font=("consolas", 10), undo=True, wrap='word', state='disabled')
txt.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
scrollb = Scrollbar(txt_frm, command=txt.yview)
scrollb.grid(row=0, column=1, sticky='nsew')
txt['yscrollcommand'] = scrollb.set
input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)
input_field.bind("<Return>", enter_pressed)
window.mainloop()

print('Goodbye')
