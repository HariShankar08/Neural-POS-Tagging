from matplotlib import pyplot as plt
import json


with open('epochs_devset_accuracy/epochs_gru.json') as f:
    data = json.load(f)

plt.plot(data, label='GRU')

with open('epochs_devset_accuracy/epochs_lstm.json') as f:
    data = json.load(f)

plt.plot(data, label='LSTM')

with open('epochs_devset_accuracy/epochs_rnn.json') as f:
    data = json.load(f)

plt.plot(data, label='RNN')

plt.xlabel('Epochs')
plt.xticks(range(6))
plt.ylabel('Accuracy')

plt.legend()
plt.savefig('graphs/rnns_graph.png')
