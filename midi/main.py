import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
import torch.utils.data as data
from torch.autograd import Variable

from utils import midiread, midiwrite
from matplotlib import pyplot as plt

# import skimage.io as io
# from IPython.display import FileLink
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def midi_filename_to_piano_roll(midi_filename):
    midi_data = midiread(midi_filename, dt=0.3)

    piano_roll = midi_data.piano_roll.transpose()

    # Pressed notes are replaced by 1
    piano_roll[piano_roll > 0] = 1

    return piano_roll


def pad_piano_roll(piano_roll, max_length=132333, pad_value=0):
    original_piano_roll_length = piano_roll.shape[1]

    padded_piano_roll = np.zeros((88, max_length))
    padded_piano_roll[:] = pad_value

    padded_piano_roll[:, -original_piano_roll_length:] = piano_roll

    return padded_piano_roll


class NotesGenerationDataset(data.Dataset):

    def __init__(self, midi_folder_path, longest_sequence_length=1491):
        self.midi_folder_path = midi_folder_path

        midi_filenames = os.listdir(midi_folder_path)

        self.longest_sequence_length = longest_sequence_length

        midi_full_filenames = map(lambda filename: os.path.join(midi_folder_path, filename), midi_filenames)

        self.midi_full_filenames = list(midi_full_filenames)

        if longest_sequence_length is None:
            self.update_the_max_length()

    def update_the_max_length(self):
        sequences_lengths = map(lambda filename: midi_filename_to_piano_roll(filename).shape[1],
                                self.midi_full_filenames)

        max_length = max(sequences_lengths)

        self.longest_sequence_length = max_length

    def __len__(self):
        return len(self.midi_full_filenames)

    def __getitem__(self, index):
        midi_full_filename = self.midi_full_filenames[index]

        piano_roll = midi_filename_to_piano_roll(midi_full_filename)

        # Shifting by one time step
        sequence_length = piano_roll.shape[1] - 1

        # Shifting by one time step
        input_sequence = piano_roll[:, :-1]
        ground_truth_sequence = piano_roll[:, 1:]

        # padding sequence so that all of them have the same length
        input_sequence_padded = pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)

        ground_truth_sequence_padded = pad_piano_roll(ground_truth_sequence, max_length=self.longest_sequence_length,
                                                      pad_value=-100)

        input_sequence_padded = input_sequence_padded.transpose()
        ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()

        return (torch.FloatTensor(input_sequence_padded), torch.LongTensor(ground_truth_sequence_padded),
                torch.LongTensor([sequence_length]))


def post_process_sequence_batch(batch_tuple):
    input_sequences, output_sequences, lengths = batch_tuple

    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(
        *training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)

    input_sequence_batch_sorted = input_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]

    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)

    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)

    return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, n_layers=2):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers

        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)

        self.logits_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_sequences, input_sequences_lengths, hidden=None):
        batch_size = input_sequences.shape[1]

        notes_encoded = self.notes_encoder(input_sequences)

        notes_encoded_rolled = notes_encoded.permute(1, 2, 0).contiguous()
        notes_encoded_norm = self.bn(notes_encoded_rolled)

        notes_encoded_norm_drop = nn.Dropout(0.25)(notes_encoded_norm)
        notes_encoded_complete = notes_encoded_norm_drop.permute(2, 0, 1)

        # Here we run rnns only on non-padded regions of the batch
        packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded_complete, input_sequences_lengths)
        outputs, hidden = self.lstm(packed, hidden)

        # Here we unpack sequence(back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        outputs_norm = self.bn(outputs.permute(1, 2, 0).contiguous())
        outputs_drop = nn.Dropout(0.1)(outputs_norm)
        logits = self.logits_fc(outputs_drop.permute(2, 0, 1))
        logits = logits.transpose(0, 1).contiguous()

        neg_logits = (1 - logits)

        # Since the BCE loss doesn't support masking,crossentropy is used
        binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
        logits_flatten = binary_logits.view(-1, 2)
        return logits_flatten, hidden


def validate(model):
    model.eval()
    full_val_loss = 0.0
    overall_sequence_length = 0.0

    for batch in valset_loader:

        post_processed_batch_tuple = post_process_sequence_batch(batch)

        input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

        output_sequences_batch_var =  Variable( output_sequences_batch.contiguous().view(-1))#.cuda() )

        input_sequences_batch_var = Variable( input_sequences_batch)#.cuda() )

        logits, _ = model(input_sequences_batch_var, sequences_lengths)

        loss = criterion_val(logits, output_sequences_batch_var)

        full_val_loss += loss.item()
        overall_sequence_length += sum(sequences_lengths)

    return full_val_loss / (overall_sequence_length * 88)


clip = 1.0
epochs_number = 10
sample_history = []
best_val_loss = float("inf")


def lrfinder(start, end, model, trainset_loader, epochs=2):
    model.train()  # into training mode
    lrs = np.linspace(start, end, epochs * len(trainset_loader))
    parameters = filter(lambda p: p.requires_grad, model.parameters())  # get all parameters which need grad
    optimizer = torch.optim.Adam(rnn.parameters(), start)
    loss_list = []
    ctr = 0

    for epoch_number in range(epochs):
        epoch_loss = []
        for batch in trainset_loader:
            optimizer.param_groups[0]['lr'] = lrs[ctr]
            ctr = ctr + 1

            post_processed_batch_tuple = post_process_sequence_batch(batch)

            input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

            output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1))#.cuda())

            input_sequences_batch_var = Variable(input_sequences_batch)#.cuda())

            optimizer.zero_grad()

            logits, _ = model(input_sequences_batch_var, sequences_lengths)

            loss = criterion(logits, output_sequences_batch_var)
            loss_list.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)

            optimizer.step()
        print('Epoch %d' % epoch_number)
    plt.plot(lrs, loss_list)
    return lrs, loss_list


if __name__ == "__main__":
    trainset = NotesGenerationDataset('./Nottingham/train/', longest_sequence_length=None)

    trainset_loader = data.DataLoader(trainset, batch_size=8, shuffle=True, drop_last=True)

    X = next(iter(trainset_loader))

    valset = NotesGenerationDataset('./Nottingham/valid/', longest_sequence_length=None)

    valset_loader = data.DataLoader(valset, batch_size=8, shuffle=False, drop_last=False)

    print(X[0].shape)

    X_val = next(iter(valset_loader))
    print(X_val[0].shape)

    model = RNN(input_size=88, hidden_size=512, num_classes=88)#.cuda()

    criterion = nn.CrossEntropyLoss()#.cuda()
    criterion_val = nn.CrossEntropyLoss()#.cuda()

    validate(model)

    rnn = RNN(input_size=88, hidden_size=512, num_classes=88)
    rnn = rnn#.cuda()
    lrs, losses = lrfinder(1e-4, 1e-1 * 5, rnn, trainset_loader)

    plt.plot(lrs[:15], losses[:15])
