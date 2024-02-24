import conllu


def process_sentence(sentence, prev, next_):
    windows = []
    targets = []
    for i in range(len(sentence)):
        window = []
        for j in range(1, prev + 1):
            if i - j < 0:
                window.insert(0, '<S>')
            else:
                window.insert(0, sentence[i-j]['form'])

        window.append(sentence[i]['form'])

        for j in range(1, next_ + 1):
            if i + j >= len(sentence):
                window.append('</S>')
            else:
                window.append(sentence[i+j]['form'])

        targets.append(sentence[i]['upos'])

        windows.append(window)
    return windows, targets


def process_file(file_path, prev, next_):
    with open(file_path) as file:
        train = file.read()

    sentences = conllu.parse(train)

    all_windows = []
    all_labels = []
    for sentence in sentences:
        windows, labels = process_sentence(sentence, prev, next_)

        all_windows.extend(windows)
        all_labels.extend(labels)

    return all_windows, all_labels


def generate_embeddings(windows, word_index, vocabulary):
    embeddings = []
    for window in windows:
        row = []
        for word in window:
            if word not in vocabulary and word not in ('<S>', '</S>'):
                row.append(word_index['<UNK>'])
            else:
                row.append(word_index[word])
        embeddings.append(row)

    return embeddings


def encode_labels(data):
    labels = ['NUM', 'VERB', 'CCONJ', 'DET', 'PROPN', 'NOUN', 'ADP', 'PART', 'AUX', 'INTJ', 'ADV', 'PRON', 'ADJ']
    return [labels.index(item) for item in data]

def lookup_labelindex(idx):
    labels = ['NUM', 'VERB', 'CCONJ', 'DET', 'PROPN', 'NOUN', 'ADP', 'PART', 'AUX', 'INTJ', 'ADV', 'PRON', 'ADJ']
    return labels[idx]