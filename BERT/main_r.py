import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
from tqdm import trange, tqdm
import argparse
from utils import *
import numpy as np


class BERTforRegression(nn.Module):
    def __init__(self, input_dim, args, output_dim=1, dropout_prob=0.1, numeric_dim=0):
        super(BERTforRegression, self).__init__()
        self.num_labels, self.numeric_dim = output_dim, numeric_dim
        self.bert = BertModel.from_pretrained(args.model)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(input_dim + numeric_dim, output_dim)
        self.loss_fct = nn.MSELoss()
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        self.regressor.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, numeric_features=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        if numeric_features is not None:
            pooled_output = torch.cat([pooled_output, numeric_features], dim=-1)
        outputs = self.regressor(pooled_output)

        if labels is not None:
            loss = self.loss_fct(outputs.view(-1), labels.view(-1).float())
            return loss
        else:
            return outputs


def eval_model(model, eval_loader, device="cpu", dataset="w-o preprocessing",thres=5):
    total_outputs, labels = [], []
    model.eval()
    for data in eval_loader:
        data = [d.to(device) for d in data]

        numeric_features = None
        if dataset == "w-o preprocessing" or dataset == "stopwords_punctuations":
            input_ids, input_mask, label, segment_ids = data
        else:
            input_ids, input_mask, label, segment_ids, numeric_features = data

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, numeric_features=numeric_features)
            outputs = outputs.detach().cpu().numpy()
            total_outputs.extend(outputs)

        labels.extend(label.detach().cpu().numpy())
    acc, precision, f1  = evaluation(np.array(total_outputs), np.array(labels), task="regression", thres=thres)
    return acc, precision, f1


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

        acc, precision, f1 = eval_model(model, dev_loader, device, dataset, args.reg_threshold)
        print("====================================================")
        print("| epoch: {} | acc: {} | precision: {} | f1-score: {}".format(epoch, acc, precision, f1))
        print("====================================================")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "output/model_{}_{}.pt".format(args.task_name,args.type))


def main():
    parser = argparse.ArgumentParser()
    np.set_printoptions(threshold=np.inf)

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
    parser.add_argument("--type", type=int, default=0)
    parser.add_argument("--reg_threshold", type=float, default=5.0)

    args = parser.parse_args()
    task = args.task_name
    numeric_dim = 1 if args.data_dir == "stopwords_punctuations_#negStopwords" else (
        3 if args.data_dir == "representative words" else 5 if args.data_dir == "extra" else 0
    )
    print("numeric_dim: ",numeric_dim)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    # Process data
    print("Preprocessing data...")
    dataProcessor = DataProcessor(args.data_dir, task="regression")
    train_examples = dataProcessor.get_train_example(os.path.join("data", args.data_dir))
    dev_examples = dataProcessor.get_dev_example(os.path.join("data", args.data_dir))
    train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer, args.data_dir)
    dev_features = convert_examples_to_features(dev_examples, args.max_seq_length, tokenizer, args.data_dir)

    # generate the dataset and dataloaders
    if args.train:
        train_set = InputDataset(train_features, args.data_dir)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_set = InputDataset(dev_features, args.data_dir)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)

    # load pretrained model
    if args.train:
        # model = BertForSequenceClassification.from_pretrained(args.model, num_labels=2)
        model = BERTforRegression(768, args, numeric_dim=numeric_dim)
        model = model.to(device)
        train_model(train_loader, dev_loader, model, args, device, args.data_dir)

    # load the best model and give predictions on the dev set
    model = BERTforRegression(768, args, numeric_dim=numeric_dim)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_{}_{}.pt".format(args.task_name,args.type)),
                                     map_location='cuda'))
    # model = BertForSequenceClassification.from_pretrained(args.model, state_dict=model_state_dict, num_labels=2)
    model = model.to(device)
    acc, precision, f1 = eval_model(model, dev_loader, device, args.data_dir, args.reg_threshold)
    print("Final acc: {} | precision: {} | f1-score: {}".format(acc, precision, f1))


if __name__ == "__main__":
    main()
