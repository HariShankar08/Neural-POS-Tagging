from train_rnn import LSTM_Model, generate_dataset, CustomDataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

md = torch.load('pretrained_models/lstm_model.pt')

dev_embeddings, dev_labels = generate_dataset('en_atis-ud-dev.conllu', md.vocabulary, md.word_index, md.label_index)
test_embeddings, test_labels = generate_dataset('en_atis-ud-test.conllu', md.vocabulary, md.word_index, md.label_index)

dev_data = CustomDataset(dev_embeddings, dev_labels)
test_data = CustomDataset(test_embeddings, test_labels)

dev_loader = DataLoader(dev_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

y_loader = []
y_preds = []

loader = test_loader

for X, y in loader:
    if torch.cuda.is_available():
        X, y = X.cuda(), y.cuda()

    y_pred = md(X)

    y_pred = torch.argmax(y_pred, dim=2)

    gold = [i.item() for i in y[0] if i.item() != 0]
    pred = [i.item() for i in y_pred[0] if i.item() != 0]
    for i in range(1, len(y)):
        if y[i][-1] != 0:
            gold.append(y[i][-1].item())
            pred.append(y_pred[i][-1].item())
        else:
            break

    y_loader.extend(gold)
    y_preds.extend(pred)


# print(classification_report(y_loader, y_preds))
print(confusion_matrix(y_loader, y_preds))
