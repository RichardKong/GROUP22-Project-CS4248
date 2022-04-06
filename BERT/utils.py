import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score


class InputExample(object):
    """A single training/test example for simple sequence classification/regression."""
    def __init__(self, guid, text, label=None, val=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) int. The label of the example for the classification task.
            This should be specified for train and dev examples, but not for test examples.
            val: (Optional) float. The rating score of the exaple for the regression task.
            This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.val_feature = val


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label, segments_ids, val_feature=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label
        self.segment_ids = segments_ids
        self.numeric_features = val_feature


class DataProcessor(object):
    def __init__(self, dataset, task="classification"):
        """
          task: classfication or regression
        """
        self.task = task
        self.dataset = dataset

    def get_train_example(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, 'train.csv')), 'train')

    def get_dev_example(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, 'test.csv')), 'dev')

    def _create_examples(self, csv_data, set_type):
        csv_data['sentiment'] = csv_data['sentiment'].apply(int)
        if self.dataset == 'extra':
            csv_data["ner_count"] = (csv_data["ner_count"] - min(csv_data["ner_count"])) * 1.0 / max(csv_data["ner_count"])
        examples = []
        for i in range(len(csv_data)):
            row = csv_data.iloc[i]
            sentence = row['text']
            label = row['sentiment'] if self.task == "classification" else row["rating"]
            guid = "%s-%s-%s" % (set_type, self.task, i)
            val_feature = None

            if self.dataset == "stopwords_punctuations_#negStopwords":
                val_feature = row[["count_neg_stopwords"]]
            elif self.dataset == "representative words":
                val_feature = row[["count_neg_stopwords", "count_pos_representative_words",
                                   "count_neg_representative_words"]]
            elif self.dataset == "extra":
                val_feature = row[["count_neg_stopwords", "count_pos_representative_words",
                                   "count_neg_representative_words", "ner_count", "pronoun_count"]]

            examples.append(
                InputExample(guid=guid, text=sentence, label=label, val=val_feature)
            )
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, dataset="w-o preprocessing"):
    features = []
    for index, example in tqdm(enumerate(examples)):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # if index == 0:
        #     print("TEXT: ", tokens)
        #     print("IDS: ", input_ids)
        input_mask = [1] * len(input_ids)
        segments_ids = [0] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segments_ids += padding

        label = example.label
        numeric_features = None
        if dataset in ["stopwords_punctuations_#negStopwords", "representative words", "extra"]:
            numeric_features = example.val_feature
        new_feature = InputFeatures(input_ids, input_mask, label, segments_ids, numeric_features)
        features.append(new_feature)

    return features


def evaluation(output, labels, task="classification"):
    # print(output)
    if task == "classification":
        pred_labels = np.argmax(output, axis=1)
        f1 = f1_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels)
        return np.sum(pred_labels == labels) * 1.0 / len(pred_labels), precision, f1
    else:
        labels[labels < 5] = 0
        labels[labels >= 5] = 1
        output[output < 5] = 0
        output[output >= 5] = 1
        f1 = f1_score(labels, output)
        precision = precision_score(labels, output)
        return np.sum(output == labels) * 1.0 / len(output), precision, f1


class InputDataset(Dataset):
    def __init__(self, features, dataset="w-o preprocessing"):
        self.features = features
        self.dataset = dataset

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = torch.tensor(feature.input_ids)
        input_mask = torch.tensor(feature.input_mask)
        label = torch.tensor(feature.label)
        segment_ids = torch.tensor(feature.segment_ids)
        # print(input_ids.shape, label)

        if self.dataset == "w-o preprocessing" or self.dataset == "stopwords_punctuations":
            return input_ids, input_mask, label, segment_ids
        else:
            numeric_feature = torch.tensor(feature.numeric_features, requires_grad=False)
            return input_ids, input_mask, label, segment_ids, numeric_feature

    def __len__(self):
        return len(self.features)
