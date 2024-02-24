# Compare various window sizes for the FFN.

from train_ffn import Leaky_FFN
import helpers
import conllu

from train_ffn import CustomDataset
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt

metrics = []
for window_size in range(5):
    print("Training with window size: ", window_size)
    PREV, NEXT = window_size, window_size

    train_windows, train_labels = helpers.process_file('en_atis-ud-train.conllu', PREV, NEXT)
    test_windows, test_labels = helpers.process_file('en_atis-ud-test.conllu', PREV, NEXT)
    val_windows, val_labels = helpers.process_file('en_atis-ud-test.conllu', PREV, NEXT)

    with open('en_atis-ud-train.conllu') as f:
        txt = f.read()
    sentences = conllu.parse(txt)

    word_freq = {}

    for sentence in sentences:
        for token in sentence:
            if token['form'] in word_freq:
                word_freq[token['form']] += 1
            else:
                word_freq[token['form']] = 1

    # print(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

    train_vocabulary = {k: v for k, v in word_freq.items() if v > 2}
    train_vocabulary['<UNK>'] = 48655
    train_vocabulary['<S>'] = 48655
    train_vocabulary['</S>'] = 48655

    c = 0
    word_index = {}
    for key in train_vocabulary:
        word_index[key] = c
        c += 1

    train_embeddings = helpers.generate_embeddings(train_windows, word_index, train_vocabulary)
    test_embeddings = helpers.generate_embeddings(test_windows, word_index, train_vocabulary)
    val_embeddings = helpers.generate_embeddings(val_windows, word_index, train_vocabulary)

    train_labels = helpers.encode_labels(train_labels)
    test_labels = helpers.encode_labels(test_labels)
    val_labels = helpers.encode_labels(val_labels)

    train_data = CustomDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_data, batch_size=32)

    test_data = CustomDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_data, batch_size=32)

    val_data = CustomDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_data, batch_size=32)

    md = Leaky_FFN(PREV, NEXT, train_vocabulary, word_index)
    if torch.cuda.is_available():
        md.cuda()

    val_metrics = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(md.parameters())

    epochs = 5

    for epoch in range(epochs):
        train_loss = 0
        md.train()
        training_accuracies = []
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1} (Train)') as t:
            for i, (X, y) in t:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                optimizer.zero_grad()
                y_pred = md(X)

                # print(y_pred)
                # print(y)
                loss = criterion(y_pred, y.long())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                y_classes = torch.argmax(y_pred, dim=1)
                accuracy = sum(torch.eq(y_classes, y)) / len(y_classes)
                training_accuracies.append(accuracy)

        val_loss = 0
        md.eval()
        val_accuracies = []
        with tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch + 1} (Validation)') as t:
            for i, (X, y) in t:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()

                y_pred = md(X)
                # print(y_pred)
                loss = criterion(y_pred, y.long())
                val_loss += loss.item()

                y_classes = torch.argmax(y_pred, dim=1)
                accuracy = sum(torch.eq(y_classes, y)) / len(y_classes)
                val_accuracies.append(accuracy)

        print(f"Training Loss: {train_loss / len(train_loader)}")
        print(f"Training Accuracy: {sum(training_accuracies) / len(training_accuracies)}")
        print(f"Validation Loss: {val_loss / len(val_loader)}")
        print(f"Validation Accuracy: {sum(val_accuracies) / len(val_accuracies)}")
        val_metrics.append((sum(val_accuracies) / len(val_accuracies)).item())

    metrics.append(val_metrics)

for i, row in enumerate(metrics):
    plt.plot(row, label=f'Window size {i}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(6))

plt.legend()

plt.savefig('comparing_windows.png')
