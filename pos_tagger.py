import string
import helpers
import torch
import argparse
from train_ffn import Leaky_FFN
from train_rnn import LSTM_Model


def generate_windows_rnn(sentence, window_size, vocabulary):
    windows = []
    for i in range(len(sentence)):
        window = []
        for j in range(1, window_size):
            if i - j < 0:
                window.insert(0, '<P>')
            else:
                word = sentence[i - j]
                if vocabulary.get(word) is not None:
                    window.insert(0, word)
                else:
                    window.insert(0, '<UNK>')

        if vocabulary.get(sentence[i]) is not None:
            window.append(sentence[i])
        else:
            window.append('<UNK>')

        for j in range(1, window_size):
            if i + j >= len(sentence):
                window.append('<P>')
            else:
                if vocabulary.get(sentence[i + j]) is not None:
                    window.append(sentence[i + j])
                else:
                    window.append('<UNK>')

        windows.append(window)
    return windows


def generate_windows_ffn(sentence, p, s, vocabulary):
    windows = []
    for i in range(len(sentence)):
        window = []
        for j in range(1, p + 1):
            if i - j < 0:
                window.insert(0, '<S>')
            else:
                word = sentence[i-j]
                if vocabulary.get(word) is not None:
                    window.insert(0, word)
                else:
                    window.insert(0, '<UNK>')

        if vocabulary.get(sentence[i]) is not None:
            window.append(sentence[i])
        else:
            window.append('<UNK>')

        for j in range(1, s + 1):
            if i + j >= len(sentence):
                window.append('</S>')
            else:
                if vocabulary.get(sentence[i + j]) is not None:
                    window.append(sentence[i + j])
                else:
                    window.append('<UNK>')

        windows.append(window)
    return windows


parser = argparse.ArgumentParser()
parser.add_argument('-f', action='store_true', help='Use FFN model')
parser.add_argument('-p', action='store_true', help='Use RNN model')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.f and args.p:
        print('Please choose only one model.')
        exit(1)

    elif args.f:
        md = torch.load('pretrained_models/best_ffn.pt')
        sentence = input('>')

        sentence = sentence.lower().strip().split()

        windows = generate_windows_ffn(sentence, md.prev, md.next, md.vocabulary)

        X = helpers.generate_embeddings(windows, md.word_index, md.vocabulary)

        X = torch.tensor(X)

        y = md(X)

        y_idx = torch.argmax(y, dim=1)
        tags = []
        for i in y_idx:
            tags.append(helpers.lookup_labelindex(i))

        for word, tag in zip(sentence, tags):
            print(word, tag)

    elif args.p:
        md = torch.load('pretrained_models/lstm_model.pt')
        sentence = input('>')

        sentence = sentence.lower().strip().split()

        windows = generate_windows_rnn(sentence, md.window_size, md.vocabulary)

        X = helpers.generate_embeddings(windows, md.word_index, md.vocabulary)

        X = torch.tensor(X)

        y = md(X)

        predictions = torch.argmax(y, dim=2)

        predicted_tags = [md.labels[p.item()] for p in predictions[0]]

        pred = [p.item() for p in predictions[0] if p.item() != 0]
        for row in predictions[1:]:
            if row[-1].item() != 0:
                pred.append(row[-1].item())
            else:
                break

        tags = [md.labels[p] for p in pred]

        for word, tag in zip(sentence, tags):
            print(word, tag)

    else:
        print('Please choose a model.')
        exit(1)
