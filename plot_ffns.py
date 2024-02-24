import json
from matplotlib import pyplot as plt

with open('epochs_devset_accuracy/epochs_ffn.json') as f:
    data = json.load(f)

plt.plot(data, label='FFN')

with open('epochs_devset_accuracy/epochs_leaky.json') as f:
    data = json.load(f)

plt.plot(data, label='Leaky ReLU FFN')

with open('epochs_devset_accuracy/epochs_smaller.json') as f:
    data = json.load(f)

plt.plot(data, label='Smaller FFN')


plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(6))

plt.legend()

plt.savefig('graphs/ffns_graph.png')
