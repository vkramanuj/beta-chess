# -*- coding: utf-8 -*-
"""
Code for preprocessing data and a basic dataset object for minibatching,
including chess games and pgns
"""

from itertools import ifilter
from tensorflow import gfile
from tensorflow.contrib.learn.python.learn.datasets import base
from tqdm import tqdm
import numpy as np
import subprocess
import re
import pandas as pd

pieces = ['p', 'k', 'r', 'q', 'n', 'b', 'P', 'K', 'R', 'Q', 'N', 'B']
unicode_pieces = ['♟', '♚', '♜', '♛', '♞', '♝', '♙', '♔', '♖', '♕', '♘', '♗']
piece_map = dict(zip(pieces, xrange(1, len(pieces) + 1)))

# TODO use chess board embeddings, get rid of code smell
class ChessDataset(object):
    def __init__(self, csv_path, batch_size, max_chunk_size=300000):
        self.csv_path = csv_path

        assert gfile.Exists(csv_path), 'CSV file does not exist,\
                                       use data.generate_csv to create it'

        self.batch_size = batch_size
        self._title_line = map(str, range(64)) + ['Side to Move', 'Winner']
        self._types = dict([(a, np.int32) for a in self._title_line])
        self._num_examples = file_len(csv_path) - 1
        self._completed_epochs = 0
        self._max_chunk_size = max_chunk_size
        self.is_chunked = self._num_examples > self._max_chunk_size
        self._index_in_epoch = 0

        if self.is_chunked:
            self._chunk_generator = pd.read_csv(csv_path, chunksize=self._max_chunk_size,\
                                               skiprows=1,\
                                               header=None, names=self._title_line,\
                                               dtype=self._types)
            self._current_chunk = next(self._chunk_generator, None).sample(frac=1)
        else:
            self.data = pd.read_csv(csv_path, skiprows=1,\
                                    nrows=self._num_examples, header=None, names=self._title_line,\
                                    dtype=self._types)
            self.data = self.data.sample(frac=1)
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch(self):
        return self._completed_epochs

    def next_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size

        if self.is_chunked:
            if self._index_in_epoch > len(self._current_chunk):
                print "Reading in new chunk..."
                self._current_chunk = next(self._chunk_generator, None)
                if self._current_chunk is not None:
                    self._current_chunk = self._current_chunk.sample(frac=1)
                start, self._index_in_epoch = 0, self.batch_size
            
            if self._current_chunk is None:
                print "Finished epoch, generating new chunk..."
                self._chunk_generator = pd.read_csv(self.csv_path, chunksize=self._max_chunk_size,\
                                            skiprows=1,\
                                            header=None, names=self._title_line,\
                                            dtype=self._types)
                self._current_chunk = next(self._chunk_generator, None).sample(frac=1)
                start, self._index_in_epoch = 0, self.batch_size
                self._completed_epochs += 1
            
            batch_x = self._current_chunk.iloc[start:self._index_in_epoch, :-1]
            batch_y = self._current_chunk.iloc[start:self._index_in_epoch, -1]

            return batch_x, batch_y
        else:
            if self._index_in_epoch > self._num_examples:
                self._completed_epochs += 1
                self.data = self.data.sample(frac=1) # shuffle again
                start = 0
                self._index_in_epoch = self.batch_size
            
            batch_x = self._current_chunk.iloc[start:self._index_in_epoch, :-1]
            batch_y = self._current_chunk.iloc[start:self._index_in_epoch, -1]

            return batch_x, batch_y
        

def generate_datasets(csv_name, batch_size, validation_split=[0.7, 0.1, 0.2], overwrite=False):
    csv_path = 'datasets/csvs/%s.csv' % csv_name
    csv_train_path = 'datasets/csvs/%s_train.csv' % csv_name
    csv_test_path = 'datasets/csvs/%s_test.csv' % csv_name
    csv_validation_path = 'datasets/csvs/%s_validation.csv' % csv_name
    paths = [csv_train_path, csv_validation_path, csv_test_path]

    assert abs(sum(validation_split) - 1) <= 1e-5,\
            'Split of validation must be close to 1'

    dataset_size = file_len(csv_path) - 1

    train_idx = 0 
    validation_idx = int(dataset_size * validation_split[0])
    test_idx = validation_idx + int(dataset_size * validation_split[1])

    print "Creating train, validation, and test files..."
    if not (gfile.Exists(csv_train_path) and\
            gfile.Exists(csv_test_path) and\
            gfile.Exists(csv_validation_path)) or overwrite:

        csv_gen = (open(path, 'w') for path in paths)
        csv = csv_gen.next()
        for n, line in tqdm(enumerate(open(csv_path)), total=dataset_size):
            if n == validation_idx:
                csv = csv_gen.next()
            
            if n == test_idx:
                csv = csv_gen.next()
            
            csv.write(line)

    print "Creating datasets..."
    train = ChessDataset(csv_train_path, batch_size)
    validation = ChessDataset(csv_validation_path, batch_size)
    test = ChessDataset(csv_test_path, batch_size)

    return base.Datasets(train=train, validation=validation, test=test)


def generate_csv(data_name, overwrite=False, verbose=True):
    pgn_path = 'datasets/pgns/%s.pgn' % data_name
    epd_path = 'datasets/epds/%s.epd' % data_name
    csv_path = 'datasets/csvs/%s.csv' % data_name

    if not overwrite and gfile.Exists(csv_path):
        print "Loading pre-existing csv %s..." % csv_path
        return data_name
    
    print "Creating CSV at %s..." % csv_path
    if overwrite or not gfile.Exists(epd_path):
        if gfile.Exists(csv_path):
            gfile.Remove(csv_path)

        subprocess.call(['./scripts/pgn-extract', '-Wepd', pgn_path, '-o', epd_path])

    winners = []

    # Getting winners from pgn
    result_map = {
        '1-0': 0,
        '0-1': 1,
        '1/2-1/2': 2
    }

    print "Getting winners..."
    quotes = re.compile('".*"')
    for line in open(pgn_path):
        if "Result" in line:
            result = quotes.findall(line)[0][1:-1]
            winners.append(result_map[result])
    
    print "Matched %d games" % len(winners)

    print "Setting up CSV"
    csv = open(csv_path, 'w')
    title_line = ", ".join(map(str, range(64)) + ['Side to Move', 'Winner']) + '\n'
    csv.write(title_line)

    print "Progress for transcribing boards to CSV %s" % csv_path

    game_i = 0
    for line in tqdm(open(epd_path), total=file_len(epd_path)):
        parsed = line.split()
        if not parsed:
            game_i += 1
            continue

        board_raw = parsed[0]
        side_to_move = len(piece_map) + 1 if parsed[1] == 'b' else len(piece_map) + 2
        board = [0] * 66

        i = 0
        for ch in ifilter(lambda x: x != '/', board_raw):
            if not ch.isdigit():
                board[i] = piece_map[ch]
                i += 1
            else:
                i += int(ch)
            
        board[64] = side_to_move
        board[65] = winners[game_i]

        csv.write(','.join(str(ch) for ch in board) + '\n')
        # Processing the board
    
    return data_name
    
def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    
    return i + 1