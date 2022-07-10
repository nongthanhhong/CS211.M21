import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import glob
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np
from scipy.spatial import distance

from seq2seq import *
from dataloading import *


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=15):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == '__main__':

    # device choice
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    MAIN_PATH = "/content/drive/MyDrive/CS_project/CS211.M21/final_project"  
    corpus_name = "train"
    corpus = os.path.join(MAIN_PATH, "data", corpus_name)
    datafile = os.path.join(corpus, "formatted_dialogues_train.txt")

    # Load/Assemble voc and pairs
    save_dir = os.path.join(MAIN_PATH, "checkpoint")
    voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    pairs = trim_rare_words(voc, pairs, min_count=3)

    # Example for validation
    small_batch_size = 5
    batches = batch_2_train_data(
        voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    # Configure models
    # Configure training/optimization
    clip = 50.0
    #Configure RL model
    model_name='RL_model_seq'
    n_iteration = 100000
    print_every=100
    save_every=500
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    teacher_forcing_ratio = 0.5
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 0  # 4000
    # loadFilename = os.path.join(save_dir,'{}_{}-{}_{}'.format(model_name, encoder_n_layers, decoder_n_layers, hidden_size),
    #                             '{}_checkpoint.tar'.format(checkpoint_iter))
    list_of_files = glob.glob(os.path.join(save_dir, '{}_{}-{}_{}'.format(model_name, encoder_n_layers, decoder_n_layers, hidden_size),'*')) # * means all if need specific format then *.csv
    if list_of_files:
        loadFilename = max(list_of_files, key=os.path.getctime)
        print("load checkpoint from: ", loadFilename)
        checkpoint_iter = (int)(loadFilename.split('/')[-1].split('_')[0])
    else:
        print("Checkpoint is empty")

    # Load model if a loadFilename is provided
    if loadFilename: 
        # If loading on same machine the model was trained on
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(loadFilename, map_location=torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc_dict = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        print("Now loading saved model state dicts")
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    # encoder.train()
    # decoder.train()

    # # Initialize optimizers
    # print('Building optimizers ...')
    # encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.Adam(
    #     decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    # if loadFilename:
    #     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    #     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # if USE_CUDA:
    #     # If you have cuda, configure cuda to call
    #     for state in encoder_optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()

    #     for state in decoder_optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()


    # Initialize encoder & decoder models
    # encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    # decoder = LuongAttnDecoderRNN(
    #     attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # # Use appropriate device
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # # Set dropout layers to eval mode
    # encoder.eval()
    # decoder.eval()

    # # Initialize search module
    # searcher = GreedySearchDecoder(encoder, decoder)

    # # Begin chatting
    # #evaluateInput(encoder, decoder, searcher, voc)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting
    evaluate_input(encoder, decoder, searcher, voc)
