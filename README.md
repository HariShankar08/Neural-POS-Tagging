# INLP Assignment 2 - Neural POS Tagging

## Directory Structure

```
.
├── en_atis-ud-dev.conllu
├── en_atis-ud-test.conllu
├── en_atis-ud-train.conllu
├── epochs_devset_accuracy
│   ├── epochs_ffn.json
│   ├── epochs_gru.json
│   ├── epochs_leaky.json
│   ├── epochs_lstm.json
│   ├── epochs_rnn.json
│   └── epochs_smaller.json
├── ffn_analysis.py
├── graphs
│   ├── comparing_windows.png
│   ├── ffns_graph.png
│   └── rnns_graph.png
├── helpers.py
├── metrics_ffn_outputs
│   ├── classication_report_dev.png
│   ├── classification_report_ffn_dev.txt
│   ├── classification_report_ffn_test.txt
│   ├── classification_report_test.png
│   ├── confusion_matrix_dev.png
│   ├── confusion_matrix_ffn_dev.txt
│   ├── confusion_matrix_ffn_test.txt
│   └── confusion_matrix_test.png
├── metrics_ffn.py
├── metrics_rnn_outputs
│   ├── classification_report_dev.png
│   ├── classification_report_rnn_dev.txt
│   ├── classification_report_rnn_test.txt
│   ├── classification_report_test.png
│   ├── confusion_matrix_dev.png
│   ├── confusion_matrix_rnn_dev.txt
│   ├── confusion_matrix_rnn_test.txt
│   └── confusion_matrix_test.png
├── metrics_rnn.py
├── plot_ffns.py
├── plot_rnns.py
├── pos_tagger.py
├── pretrained_models
│   ├── best_ffn.pt
│   ├── ffn.pt
│   ├── gru_model.pt
│   ├── leaky_ffn.pt
│   ├── lstm_model.pt
│   ├── rnn.pt
│   └── smaller_ffn.pt
├── README.md
├── report.pdf
├── requirements.txt
├── train_ffn.py
└── train_rnn.py

5 directories, 47 files
```

## Setup

Install the dependencies as required:

```bash
pip install -r requirements.txt
```
## How to run the files:

```bash
python pos_tagger <-p/-f>
```

This runs the POS tagger using the best (as determined from the analysis) RNN/FFNN; following which the user is expected to input a sentence in the prompt.



