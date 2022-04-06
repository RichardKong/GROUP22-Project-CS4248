import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from tqdm import trange, tqdm
import math
import argparse
from utils import *


class BERTforClassification(nn.Module):
    def __init__(self, input_dim, args, output_dim=2, dropout_prob=0.1, numeric_dim=0):
        super(BERTforClassification, self).__init__()
        self.num_labels, self.numeric_dim = output_dim, numeric_dim
        self.bert = BertModel.from_pretrained(args.model)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_dim + numeric_dim, output_dim)
        self.loss_fct = nn.CrossEntropyLoss()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, numeric_features=None):
        # punct_ids = torch.tensor([3, 7, 15, 16, 17, 19, 20, 31, 32], dtype=torch.int64).cuda()
        # word_ids = torch.tensor([1, 2, 4, 5, 6, 8, 9, 10, 11],
        #                         dtype=torch.int64).cuda()
        # first_input = input_ids[[0]]
        # hidden_states = self.bert.embeddings(first_input, token_type_ids[[0]])
        # # print(hidden_states.shape)
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_mask_use = extended_attention_mask[[0]]
        # # print(extended_attention_mask.shape)
        # self_att_layer = self.bert.encoder.layer[0].attention.self
        #
        # mixed_query_layer = self_att_layer.query(hidden_states)
        # mixed_key_layer = self_att_layer.key(hidden_states)
        # mixed_value_layer = self_att_layer.value(hidden_states)
        # # print(mixed_query_layer.shape)
        #
        # query_layer = self_att_layer.transpose_for_scores(mixed_query_layer)
        # # print(query_layer.shape)
        # key_layer = self_att_layer.transpose_for_scores(mixed_key_layer)
        # value_layer = self_att_layer.transpose_for_scores(mixed_value_layer)
        #
        # # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self_att_layer.attention_head_size)
        # # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores + extended_mask_use

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print('ids: ', first_input)
        # for word_id in word_ids:
        #     print('att_prob_punct: ', attention_probs[0, 0, word_id, punct_ids])
        #     print('att_prob_words: ', attention_probs[0, 0, word_id, word_ids])
        #     print(torch.mean(attention_probs[0, 0, word_id, punct_ids]), torch.mean(attention_probs[0, 0, word_id, word_ids]))
        # exit()

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        if numeric_features is not None:
            pooled_output = torch.cat([pooled_output, numeric_features], dim=-1).to(torch.float32)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def eval_model(model, eval_loader, device="cpu", dataset="w-o preprocessing"):
    total_logits, labels = [], []
    model.eval()
    for data in eval_loader:
        data = [d.to(device) for d in data]

        numeric_features = None
        if dataset == "w-o preprocessing" or dataset == "stopwords_punctuations":
            input_ids, input_mask, label, segment_ids = data
        else:
            input_ids, input_mask, label, segment_ids, numeric_features = data

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, numeric_features=numeric_features)
            logits = logits.detach().cpu().numpy()
            total_logits.extend(logits)

        labels.extend(label.detach().cpu().numpy())
    acc, prec, f1 = evaluation(np.array(total_logits), np.array(labels))
    return acc, prec, f1


def train_model(train_loader, dev_loader, model, args, device="cpu", dataset="w-o preprocessing"):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.init_lr,
                         warmup=args.warmup)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, eps=1e-8)

    print("***** Running training *****")
    print("  Num examples = ", 25000)
    print("  Batch size = ", args.batch_size)

    best_acc = 0.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()
        for step, data in enumerate(tqdm(train_loader)):
            data = [d.to(device) for d in data]

            numeric_features = None
            if dataset == "w-o preprocessing" or dataset == "stopwords_punctuations":
                input_ids, input_mask, label, segment_ids = data
            else:
                input_ids, input_mask, label, segment_ids, numeric_features = data

            model.zero_grad()
            optimizer.zero_grad()
            loss = model(input_ids, segment_ids, input_mask, label, numeric_features)
            # print(loss)
            # exit()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print("Loss at Epoch{}: {}".format(epoch, total_loss * 1.0 / len(train_loader)))

        acc, p, f1 = eval_model(model, dev_loader, device, dataset)
        print("====================================================")
        print("| epoch: {} | acc: {} | precision: {} | f1: {}".format(epoch, acc, p, f1))
        print("====================================================")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "output/model_{}.pt".format(args.task_name))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, required=True,
                        help="The type of the task, including 'classification' and 'regression'")
    parser.add_argument("--data_dir", type=str, default="w-o preprocessing")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--model", type=str, default="bert-base-uncased")

    args = parser.parse_args()
    task = args.task_name
    numeric_dim = 0
    if args.data_dir == "stopwords_punctuations_#negStopwords":
        numeric_dim = 1
    elif args.data_dir == "representative words":
        numeric_dim = 3
    elif args.data_dir == "extra":
        numeric_dim = 5

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    # Process data
    print("Preprocessing data...")
    dataProcessor = DataProcessor(args.data_dir)
    train_examples = dataProcessor.get_train_example(os.path.join("../data", args.data_dir))
    dev_examples = dataProcessor.get_dev_example(os.path.join("../data", args.data_dir))
    train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, args.data_dir)
    dev_features = convert_examples_to_features(dev_examples, args.max_seq_length, tokenizer, args.data_dir)

    # generate the dataset and dataloaders
    if args.train:
        eval_features = train_features[int(len(train_features) * 0.75):]
        train_set, eval_set = InputDataset(train_features, args.data_dir), InputDataset(eval_features, args.data_dir)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(dataset=eval_set, batch_size=args.batch_size)
    dev_set = InputDataset(dev_features, args.data_dir)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)

    # load pretrained model
    if args.train:
        # model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)
        model = BERTforClassification(768, args, numeric_dim=numeric_dim)
        model = model.to(device)
        train_model(train_loader, eval_loader, model, args, device, args.data_dir)

    # load the best model and give predictions on the dev set
    model = BERTforClassification(768, args, numeric_dim=numeric_dim)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_{}.pt".format(args.task_name)),
                                     map_location='cuda'))
    # model = BertForSequenceClassification.from_pretrained(args.model, state_dict=model_state_dict, num_labels=2)
    model = model.to(device)
    acc = eval_model(model, dev_loader, device, args.data_dir)
    print("Final Accuracy of the Model: ", acc)


if __name__ == "__main__":
    main()
