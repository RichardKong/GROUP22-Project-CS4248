import functools
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


train=pd.read_csv('../data/representaive words/train.csv')
# s = (train['ner_count'] - train['ner_count'].min())/(train['ner_count'].max() - train['ner_count'].min())
# train=train.drop(['ner_count'],axis=1)
# train.insert(7,'ner_count',s)



test=pd.read_csv('../data/representaive words/test.csv')
# s2 = (test['ner_count'] - test['ner_count'].min())/(test['ner_count'].max() - test['ner_count'].min())
# test=test.drop(['ner_count'],axis=1)
# test.insert(7,'ner_count',s2)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def tokenize_data(data, tokenizer, max_length):
    tokens = tokenizer(data)[:max_length]
    length = len(tokens)
    return tokens,length

max_length = 128
train['tokens']='a'
train['length']=0
for i in range(len(train)):
    train['tokens'][i],train['length'][i]=tokenize_data(train['text'][i],tokenizer, max_length)

max_length = 128
test['tokens']='a'
test['length']=0
for i in range(len(test)):
    test['tokens'][i],test['length'][i]=tokenize_data(test['text'][i],tokenizer, max_length)

train_tokens=train['tokens']
train_tokens=train_tokens.tolist()

min_freq = 0
special_tokens = ['<unk>', '<pad>']

vocab = torchtext.vocab.build_vocab_from_iterator(train_tokens,
                                                  min_freq=min_freq,
                                                  specials=special_tokens)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

vocab.set_default_index(unk_index)

def numericalize_data(data, vocab):
    ids = [vocab[token] for token in data]
    return ids

train['ids']='a'
for i in range(len(train)):
    train['ids'][i]=numericalize_data(train['tokens'][i],vocab)

test['ids']='a'
for i in range(len(test)):
    test['ids'][i]=numericalize_data(test['tokens'][i],vocab)

train_use=train.drop(columns =['text','tokens'],inplace = False)

for i in range(len(train_use)):
    train_use['ids'][i]=np.array(train_use['ids'][i])
    train_use['ids'][i]=torch.from_numpy(train_use['ids'][i])

test_use=test.drop(columns = ['text','tokens'],inplace = False)

for i in range(len(test_use)):
    test_use['ids'][i]=np.array(test_use['ids'][i])
    test_use['ids'][i]=torch.from_numpy(test_use['ids'][i])

test_data=test_use

train_data, valid_data = train_test_split(train_use, test_size=0.25)

train_dic=train_data.to_dict(orient="record")
valid_dic=valid_data.to_dict(orient="record")
test_dic=test_data.to_dict(orient="record")

for i in range(len(train_dic)):
    train_dic[i]['sentiment']=np.array(train_dic[i]['sentiment'])
    train_dic[i]['sentiment']=torch.tensor(train_dic[i]['sentiment'])

for i in range(len(train_dic)):
    train_dic[i]['length']=np.array(train_dic[i]['length'])
    train_dic[i]['length']=torch.tensor(train_dic[i]['length'])

for i in range(len(valid_dic)):
    valid_dic[i]['sentiment']=np.array(valid_dic[i]['sentiment'])
    valid_dic[i]['sentiment']=torch.tensor(valid_dic[i]['sentiment'])

for i in range(len(valid_dic)):
    valid_dic[i]['length']=np.array(valid_dic[i]['length'])
    valid_dic[i]['length']=torch.tensor(valid_dic[i]['length'])

for i in range(len(test_dic)):
    test_dic[i]['sentiment']=np.array(test_dic[i]['sentiment'])
    test_dic[i]['sentiment']=torch.tensor(test_dic[i]['sentiment'])

for i in range(len(test_dic)):
    test_dic[i]['length']=np.array(test_dic[i]['length'])
    test_dic[i]['length']=torch.tensor(test_dic[i]['length'])

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, ids, length):
        ids=ids.long()
        length=length.long()
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction

vocab_size =len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(train_data['sentiment'].unique())
n_layers = 2
bidirectional = True
dropout_rate = 0.5

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
             pad_index)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

model.apply(initialize_weights)

vectors = torchtext.vocab.FastText()

pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())

model.embedding.weight.data = pretrained_embedding

lr = 5e-4

optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)

model = model.to(device)
criterion = criterion.to(device)

def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['sentiment'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'length': batch_length,
             'sentiment': batch_label}
    return batch

batch_size = 512

collate = functools.partial(collate, pad_index=pad_index)

train_dataloader = torch.utils.data.DataLoader(train_dic, 
                                               batch_size=batch_size, 
                                               collate_fn=collate, 
                                               shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(valid_dic, batch_size=batch_size, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(test_dic, batch_size=batch_size, collate_fn=collate)

def train(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_losses = []
    epoch_accs = []
    epoch_pre=[]
    epoch_f1=[]
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']  
        label = batch['sentiment'].to(device)
        prediction = model(ids, length)
        label=label.long()
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        precision=get_precision(prediction, label)
        f1 = get_f1(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        epoch_pre.append(precision.item())
        epoch_f1.append(f1.item())

    return epoch_losses, epoch_accs,epoch_pre,epoch_f1

def evaluate(dataloader, model, criterion, device):
    
    model.eval()
    epoch_losses = []
    epoch_accs = []
    epoch_pre = []
    epoch_f1 = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            
            length = batch['length']
            label = batch['sentiment'].to(device)
            
            prediction = model(ids, length)
            label=label.long()
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            precision = get_precision(prediction, label)
            f1 = get_f1(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            epoch_pre.append(precision.item())
            epoch_f1.append(f1.item())

    return epoch_losses, epoch_accs,epoch_pre,epoch_f1

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def get_precision(prediction,label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    predicted_classes= predicted_classes.cpu().numpy()
    label=label.cpu().numpy()
    precision=precision_score(predicted_classes,label)
    return precision

def get_f1(prediction,label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    predicted_classes= predicted_classes.cpu().numpy()
    label=label.cpu().numpy()
    f1=f1_score(predicted_classes,label)
    return f1

n_epochs = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
train_pre=[]
train_f1=[]
valid_losses = []
valid_accs = []
valid_pre=[]
valid_f1=[]

for epoch in range(n_epochs):

    train_loss, train_acc,train_pre,train_f1 = train(train_dataloader, model, criterion, optimizer, device)

    valid_loss, valid_acc, valid_pre, valid_f1 = evaluate(valid_dataloader, model, criterion, device)
    train_losses.extend(train_loss)
    train_accs.extend(train_acc)
    train_pre.extend(train_pre)
    train_f1.extend(train_f1)

    valid_losses.extend(valid_loss)
    valid_accs.extend(valid_acc)
    valid_pre.extend(valid_pre)
    valid_f1.extend(valid_f1)
    
    epoch_train_loss = np.mean(train_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_train_pre = np.mean(train_pre)
    epoch_train_f1 = np.mean(train_f1)
    epoch_valid_loss = np.mean(valid_loss)
    epoch_valid_acc = np.mean(valid_acc)
    epoch_valid_pre = np.mean(valid_pre)
    epoch_valid_f1 = np.mean(valid_f1)
    
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        torch.save(model.state_dict(), 'lstm.pt')
    
    print(f'epoch: {epoch+1}')
    print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}, train_pre: {epoch_train_pre: .3f}, train_f1: {epoch_train_f1: .3f}')
    print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f},valid_pre: {epoch_valid_pre:.3f},valid_f1: {epoch_valid_f1:.3f}')

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(1,1,1)
# ax.plot(train_losses, label='train loss')
# ax.plot(valid_losses, label='valid loss')
# plt.legend()
# ax.set_xlabel('updates')
# ax.set_ylabel('loss');
#
#
#
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(1,1,1)
# ax.plot(train_accs, label='train accuracy')
# ax.plot(valid_accs, label='valid accuracy')
# plt.legend()
# ax.set_xlabel('updates')
# ax.set_ylabel('accuracy');

model.load_state_dict(torch.load('lstm.pt'))

test_loss, test_acc,test_pre,test_f1= evaluate(test_dataloader, model, criterion, device)

epoch_test_loss = np.mean(test_loss)
epoch_test_acc = np.mean(test_acc)
epoch_test_pre = np.mean(test_pre)
epoch_test_f1 = np.mean(test_f1)

print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f},test_pre: {epoch_test_pre:.3f},test_f1: {epoch_test_f1:.3f}')
