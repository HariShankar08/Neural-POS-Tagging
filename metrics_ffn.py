from train_ffn import Leaky_FFN, CustomDataset
from helpers import process_file, generate_embeddings, encode_labels
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

md = torch.load('pretrained_models/best_ffn.pt')

dev_windows, dev_labels = process_file('en_atis-ud-dev.conllu', md.prev, md.next)
test_windows, test_labels = process_file('en_atis-ud-test.conllu', md.prev, md.next)

dev_embeddings = generate_embeddings(dev_windows, md.word_index, md.vocabulary)
test_embeddings = generate_embeddings(test_windows, md.word_index, md.vocabulary)

dev_labels = encode_labels(dev_labels)
test_labels = encode_labels(test_labels)

dev_data = CustomDataset(dev_embeddings, dev_labels)
test_data = CustomDataset(test_embeddings, test_labels)

dev_loader = DataLoader(dev_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

y_loader = []
y_preds = []

loader = dev_loader

for X, y in loader:
    if torch.cuda.is_available():
        X, y = X.cuda(), y.cuda()
    y_loader.extend(y.cpu().numpy())

    y_pred = md(X)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.cpu().numpy()
    y_preds.extend(y_pred)


print(classification_report(y_loader, y_preds))
# print(confusion_matrix(y_loader, y_preds))
