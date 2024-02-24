import argparse
import json

import helpers
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

PREV = 1
NEXT = 1


class CustomDataset(Dataset):
    def __init__(self, embedding, label):
        self.embedding = embedding
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            embedding = [self.embedding[i] for i in idx]
            label = [self.label[i] for i in idx]
        except TypeError:
            embedding = self.embedding[idx]
            label = self.label[idx]

        return torch.tensor(embedding), torch.tensor(label)


class FFN(nn.Module):
    def __init__(self, prev, next_, vocabulary, word_index):
        super(FFN, self).__init__()

        self.prev = prev
        self.next = next_
        self.vocabulary = vocabulary
        self.word_index = word_index

        self.embedding = nn.Embedding(len(self.vocabulary), 256)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(256 * (prev + next_ + 1), 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 13)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)

        return x


class Smaller_FFN(nn.Module):
    def __init__(self, prev, next_, vocabulary, word_index):
        super(Smaller_FFN, self).__init__()

        self.prev = prev
        self.next = next_
        self.vocabulary = vocabulary
        self.word_index = word_index

        self.embedding = nn.Embedding(len(self.vocabulary), 256)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(256 * (prev + next_ + 1), 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 13)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)

        return x


class Leaky_FFN(nn.Module):
    def __init__(self, prev, next_, vocabulary, word_index):
        super(Leaky_FFN, self).__init__()

        self.prev = prev
        self.next = next_
        self.vocabulary = vocabulary
        self.word_index = word_index

        self.embedding = nn.Embedding(len(self.vocabulary), 256)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(256 * (prev + next_ + 1), 256)
        self.relu1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(128, 13)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)

        return x


parser = argparse.ArgumentParser()
parser.add_argument('--epoch_acc', action='store_true')
parser.add_argument('--file_name', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

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

    if args.epoch_acc:
        with open(f'epochs_{args.file_name}.json', 'w') as f:
            json.dump(val_metrics, f)

    torch.save(md, 'pretrained_models/best_ffn.pt')
