"""
For each sentence in the data, encode the labels as integers.
Set a separate label as zero if it is not in the label index.

For the training data, count the frequency of each word and keep only the words that appear more than twice.
Use a max window size; try various values; 7, 10, 20, etc.
For each sentence, generate windows of this size, padding in front if necessary. Similarly, make a window for the corresponding labels.

Make a dataset and dataloader out of this data.

Create a RNN with the following architecture:
- Embedding layer
- RNN layer
- Fully connected layer
- Time distributed output layer (number of tags)
"""
import json

import conllu
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import argparse

def generate_dataset(file_path, vocabulary, word_index, label_index, window_size=5):
    with open(file_path) as file:
        f = file.read()

    sentences = conllu.parse(f)

    # Generate windows of a fixed size.
    word_windows = []
    label_windows = []
    for sentence in sentences:
        try:
            for i in range(len(sentence)):
                word_window = []
                label_window = []
                for j in range(1, window_size):
                    if i - j < 0:
                        word_window.insert(0, '<P>')
                        label_window.insert(0, 0)
                    else:
                        word_window.insert(0, sentence[i - j]['form'])
                        label_window.insert(0, label_index[sentence[i - j]['upos']])

                word_window.append(sentence[i]['form'])
                label_window.append(label_index[sentence[i]['upos']])

                for j in range(1, window_size):
                    if i + j >= len(sentence):
                        word_window.append('<P>')
                        label_window.append(0)
                    else:
                        word_window.append(sentence[i + j]['form'])
                        label_window.append(label_index[sentence[i + j]['upos']])

                word_windows.append(word_window)
                label_windows.append(label_window)
                # print(len(label_window))
        except KeyError:
            pass

    # Index words; drop words not in the vocabulary.
    word_embeddings = []
    for window in word_windows:
        row = []
        for word in window:
            if word not in vocabulary and word not in ('<S>', '</S>'):
                row.append(word_index['<UNK>'])
            else:
                row.append(word_index[word])
        word_embeddings.append(row)

    # print(label_windows)
    return word_embeddings, label_windows


class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            embedding = [self.embeddings[i] for i in idx]
            label = [self.labels[i] for i in idx]
        except TypeError:
            embedding = self.embeddings[idx]
            label = self.labels[idx]

        return torch.tensor(embedding), torch.tensor(label)


class RNN(nn.Module):
    def __init__(self, vocabulary, word_index, labels, label_index, window_size):
        super(RNN, self).__init__()

        self.vocabulary = vocabulary
        self.word_index = word_index
        self.labels = labels
        self.label_index = label_index
        self.window_size = window_size

        self.embedding = nn.Embedding(len(self.vocabulary) + 1, 256)
        self.rnn = nn.RNN(256, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, len(self.label_index))

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x)

        return x


class LSTM_Model(nn.Module):
    def __init__(self, vocabulary, word_index, labels, label_index, window_size):
        super(LSTM_Model, self).__init__()

        self.vocabulary = vocabulary
        self.word_index = word_index
        self.labels = labels
        self.label_index = label_index
        self.window_size = window_size

        self.embedding = nn.Embedding(len(self.vocabulary) + 1, 256)
        self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, len(self.label_index))

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)

        return x


class GRU_Model(nn.Module):
    def __init__(self, vocabulary, word_index, labels, label_index, window_size):
        super(GRU_Model, self).__init__()

        self.vocabulary = vocabulary
        self.word_index = word_index
        self.labels = labels
        self.label_index = label_index
        self.window_size = window_size

        self.embedding = nn.Embedding(len(self.vocabulary) + 1, 256)
        self.gru = nn.GRU(256, 256, 1, batch_first=True)
        self.fc = nn.Linear(256, len(self.label_index))

    def forward(self, x):
        # print(x)
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x)

        return x


parser = argparse.ArgumentParser()
parser.add_argument('--epoch_acc', action='store_true')
parser.add_argument('--file_name')

if __name__ == '__main__':
    args = parser.parse_args()

    with open('en_atis-ud-train.conllu') as f:
        txt = f.read()

    sentences = conllu.parse(txt)

    # Count the frequency of each word and keep only the words that appear more than twice.
    train_vocabulary = {}
    for sentence in sentences:
        for token in sentence:
            if token['form'] in train_vocabulary:
                train_vocabulary[token['form']] += 1
            else:
                train_vocabulary[token['form']] = 1

    train_vocabulary = {k: v for k, v in train_vocabulary.items() if v > 2}
    train_vocabulary['<UNK>'] = 48655
    train_vocabulary['<P>'] = 48655
    train_vocabulary['<P>'] = 48655

    # Create a word index.
    word_index = {}
    c = 1
    for key in train_vocabulary:
        word_index[key] = c
        c += 1

    # Encode the labels as integers.
    label_index = {}
    c = 1
    for sentence in sentences:
        for token in sentence:
            if token['upos'] not in label_index:
                label_index[token['upos']] = c
                c += 1

    label_index['<P>'] = 0
    labels = [k for k, v in sorted(label_index.items(), key=lambda x: x[1])]

    train_embeddings, train_labels = generate_dataset('en_atis-ud-train.conllu', train_vocabulary, word_index,
                                                      label_index)
    test_embeddings, test_labels = generate_dataset('en_atis-ud-test.conllu', train_vocabulary, word_index,
                                                    label_index)
    val_embeddings, val_labels = generate_dataset('en_atis-ud-dev.conllu', train_vocabulary, word_index,
                                                  label_index)

    train_dataset = CustomDataset(train_embeddings, train_labels)
    test_dataset = CustomDataset(test_embeddings, test_labels)
    val_dataset = CustomDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    md = LSTM_Model(train_vocabulary, word_index, labels, label_index, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(md.parameters(), lr=0.01)
    # print(len(label_index))

    # print(len(train_embeddings[0]), len(train_labels[0]))
    epochs = 5

    val_metrics = []

    for epoch in range(epochs):
        md.train()

        train_loss = 0
        training_accuracies = []
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1} (Train)') as t:
            for i, (X, y) in t:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                optimizer.zero_grad()
                y_pred = md(X)

                loss = criterion(y_pred.view(-1, len(label_index)), y.view(-1))

                y_classes = torch.argmax(y_pred, dim=2)
                # print(y_classes.shape)
                # print(y.shape)

                accuracy = sum(torch.eq(y_classes, y)) / len(y_classes)
                training_accuracies.append(accuracy)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        val_loss = 0
        md.eval()
        val_accuracies = []
        with tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch + 1} (Validation)') as t:
            for i, (X, y) in t:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                y_pred = md(X)
                loss = criterion(y_pred.view(-1, len(label_index)), y.view(-1))
                val_loss += loss.item()

                y_classes = torch.argmax(y_pred, dim=2)
                accuracy = sum(torch.eq(y_classes, y)) / len(y_classes)
                val_accuracies.append(accuracy)

        print(f"Train loss: {train_loss / len(train_loader)}")
        print("Train accuracy: ", (sum(training_accuracies) / len(training_accuracies)).mean().item())
        print(f"Validation loss: {val_loss / len(val_loader)}")
        print("Validation accuracy: ", (sum(val_accuracies) / len(val_accuracies)).mean().item())

        val_metrics.append((sum(val_accuracies) / len(val_accuracies)).mean().item())

    if args.epoch_acc:
        with open(f'epochs_{args.file_name}.json', 'w') as f:
            json.dump(val_metrics, f)

    torch.save(md, 'pretrained_models/lstm_model.pt')
