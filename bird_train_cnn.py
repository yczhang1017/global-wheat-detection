import os
#import sys
import random
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
#from torch.nn.utils.rnn import pad_sequence
from time import time
from sklearn.model_selection import train_test_split
from ranger import Ranger
import ast
#import librosa
#import librosa.display
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Train ResNeSt for bird call')
parser.add_argument('-d','--data', default='.', help='data directory')
parser.add_argument('-e','--epoch', default=42, help='number of epoch')
parser.add_argument('-l','--length', default=1293, help='length of sequence')
parser.add_argument('--lr', default=1e-6, help='learnig rate')
parser.add_argument('-r','--restart', default=None, help='restart epoch:dict_file')
parser.add_argument('-m','--milestones', default="3,6,9,12,15,18,21,24,27,30,33,36,39" ,help='number of epoch')
parser.add_argument('-g','--gamma', default=0.3 ,help='number of epoch')


args = parser.parse_args()

ndim = 128
batch_size = 32
root = Path(args.data)
cols = ['ebird_code', 'filename', 'duration', 'secondary_labels', 'primary_label']
df_train1 = pd.read_csv(root/'train.csv', usecols=cols)
ext_start = len(df_train1)
df_train2 = pd.read_csv(root/'train_extended.csv', usecols=cols)
df_train = pd.concat([df_train1,df_train2], ignore_index=True)

df_test = pd.read_csv(root/'test.csv')
df_example = pd.read_csv(root/'example_test_audio_summary.csv')
df_example["birds"].fillna(".", inplace = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
else:
    device = torch.device("cpu")

set_seed(42)    
label2code = {}
code2label = {}
code2tag = {}
label2tag = {}
tag2code = []

tag = 0
for idx, row in df_train.iterrows():
    label = row['primary_label'] 
    code = row['ebird_code']
    if code not in code2tag.keys():
        code2label[code] = label
        code2tag[code] = tag
        tag2code += [code]
        tag += 1
    if label not in label2code.keys():
        label2code[label] = code
        label2tag[label] = code2tag[code]
    
    
ntag = len(tag2code)
df_train['tag'] = df_train['ebird_code'].map(code2tag)
df_train['secondary_tags'] = df_train['secondary_labels'].map(lambda x: [label2tag[l] for l in ast.literal_eval(x) if l in label2tag])
df_example['tags']=df_example["birds"].map(lambda x: [code2tag[b] for b in x.split(' ') if b in code2tag.keys() ])

ids1 = df_train.index[df_train['duration']<15]
ids2 = df_train.index[(df_train['duration']>=15) & (df_train['duration']<45)]
ids3 = df_train.index[df_train['duration']>45]
ndist = df_train.groupby('tag').count()['filename'].values
weight = np.exp(((100/ndist)-1)/10)
#weight = (ndist.max()/ndist)**0.2
print(f'ndist range: [{ndist.min()},{ndist.max()}]')
print(f'weight range: [{weight.min()},{weight.max()}]')
with np.printoptions(precision=3, suppress=True):
    print(f'weight:{weight}')
weight = torch.tensor(weight, dtype=torch.float).to(device)


class TrainData(Dataset):
    def __init__(self, df, indices, mosaic=(1,3), l = args.length):
        self.df = df
        self.indices = indices
        self.mosaic = mosaic
        self.l = l
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        ids = [self.indices[idx]]
        mosaic = random.randint(*self.mosaic)
        for i in range(mosaic-1):
            ids += [random.choice(self.indices)]
        
        cur = 0
        x = -4*torch.ones((self.l,ndim))
        t = torch.zeros((ntag))
        #target_weight = torch.ones((ntag))
        for i, idx in enumerate(ids):
            row = self.df.iloc[idx]
            pre = root/'tensors' if idx < ext_start else root/'extended_tensors'
            filename = pre/(row['filename'][:-3]+'pt')
            Sdb = torch.load(filename)
            Sdb = (Sdb+20)/12
            l = Sdb.shape[0]
            c = self.l-cur if i==mosaic-1 else int(self.l // mosaic * random.uniform(0.8,1.2)) 
            if l > c:
                s = random.randrange(0,l-c)
                x[cur:cur+c,:] = Sdb[s:s+c,:]
            else:
                x[cur:cur+l,:] = Sdb
            cur = c 
            #if self.mosaic==(1,1): return x.view((1,-1,ndim)), torch.tensor(row['tag'])
            t[row['tag']] = 1    
            #target_weight[row['tag']] = 3
            t[row['secondary_tags']] = 1
        x = x.view((1,-1,ndim))
        #valid_len = (x>-4).sum().item()/ndim
        return x,t

class ExampleData(Dataset):
    def __init__(self, df, mosaic=(3,3), l = args.length):
        self.df = df
        self.mosaic = mosaic
        self.l = l
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        ids = [idx]
        mosaic = random.randint(*self.mosaic)
        for i in range(2):
            ids += [random.randrange(len(self.df))]
        cur = 0
        x = -4*torch.ones((self.l,ndim))
        t = torch.zeros((ntag))
        #target_weight = torch.ones((ntag))
        for i, idx in enumerate(ids):
            row = self.df.iloc[idx]
            filename = root/'example_tensors'/(row['filename_seconds']+'.pt')
            Sdb = torch.load(filename)
            Sdb = (Sdb+20)/12
            l = Sdb.shape[0]
            c = self.l-cur if i==mosaic-1 else int(self.l // mosaic * random.uniform(0.8,1.2)) 
            if l > c:
                s = random.randrange(0,l-c)
                x[cur:cur+c,:] = Sdb[s:s+c,:]
            else:
                x[cur:cur+l,:] = Sdb
            cur = c 
            t[row['tags']] = 1
        x = x.view((1,-1,ndim))
        #valid_len = (x>-4).sum().item()/ndim
        return x, t


class BertModel(nn.Module):
    def __init__(self,ndim=ndim,ntag=ntag,npos=8):
        super(BertModel, self).__init__()
        self.dm = ndim + npos
        self.npos = npos
        encoder_layer = nn.TransformerEncoderLayer(self.dm, 8, 256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(self.dm,ntag)
    def forward(self,x):
        b,l,d = x.shape
        s = torch.linspace(0,l-1,l).to(device)
        s = s.view((1,l,1)).expand((b,l,1))
        ps = [torch.sin(2*np.pi*s/(4*1.6*i)) for i in range(self.npos)]
        x = torch.cat([x]+ps, dim = 2)
        x = self.encoder(x)
        y = self.fc(x).mean(1)
        return y

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduce=True, weight=None ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce
        self.weight = weight

    def forward(self, inputs, targets):
        BCE_loss = torch.binary_cross_entropy_with_logits(inputs,targets,pos_weight=self.weight)
        pt = torch.exp(-BCE_loss).detach()
        F_loss = (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
#skf = StratifiedKFold(n_splits=5)
ids2t, ids2v = train_test_split(ids2,test_size=0.1, random_state=42)
save = root/f'ResNeSt-v5'
if not save.exists(): save.mkdir()
trainset1 = TrainData(df_train, ids1, mosaic=(1,3), l = args.length)
trainset2 = TrainData(df_train, ids2t, mosaic=(1,1), l = args.length)
trainset3 = ExampleData(df_example,  mosaic=(3,3), l = args.length)

trainset4 = TrainData(df_train, ids2t, mosaic=(1,2), l = args.length*4+1)
trainset5 = TrainData(df_train, ids3, mosaic=(1,1), l = args.length*4+1)
valset = TrainData(df_train, ids2v, mosaic=(1,1), l = args.length)

print(f'trainsets:{len(trainset1)},{len(trainset2)},{len(trainset3)},{len(trainset4)},{len(trainset5)}')

dataset = {'train1': ConcatDataset((trainset1, trainset2, trainset3)),
           'train2': ConcatDataset((trainset4, trainset5)),
            'val':valset}
data_loader = {k: DataLoader(v,
        batch_size=batch_size//4 if k=='train2' else batch_size, 
        shuffle = k.startswith('train'),
        num_workers=4, pin_memory=True)
        for k,v in dataset.items()}
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.conv1[0] = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1), bias=False)
model.fc = nn.Linear(2048,ntag)
model.to(device)
criterion = FocalLoss(weight=weight).cuda()
#optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
optimizer = Ranger(model.parameters(),lr=args.lr,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones.split(","), gamma=args.gamma)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch)
best_acc = 0
start = -1
if args.restart:
    restart = args.restart.split(':')
    if len(restart) == 1: 
        start = int(restart[0])
        checkpoint = save/f'weight_{start}.pt'
    elif len(restart) == 2:
        start, checkpoint = int(restart[0]), restart[1]
    else:
        exit(1)
    model.load_state(torch.load(checkpoint, map_location=device))
    for i in range(start+1): scheduler.step()

for e in range(start+1,args.epoch):
    """
    if e == stage:
        dataset['train'] = TrainData(df_train, train_indices, mosaic=(1,3), l = 821)
        data_loader['train'] = DataLoader(dataset['train'], batch_size=batch_size, shuffle = True, num_workers=4,pin_memory=True)
        criterion = torch.nn.BCELoss(weight=weight).cuda()
        print('Start using Mosaic and BCELoss:')
    """
    for phase,loader in data_loader.items():
        if phase.startswith('train'):
            model.train()
        else:
            model.eval()
        sum_loss = 0
        sum_tot = 0
        sum_tp = 0
        sum_fn = 0
        sum_fp = 0
        t0 = time()
        print(f'{e}/{args.epoch}:{phase}')
        for i,(x,t) in enumerate(loader):
            x = x.to(device)
            t = t.to(device)
            #target_weight = target_weight.to(device)
            y = model(x)-2
            loss = criterion(y, t)
            with torch.set_grad_enabled(phase == 'train'):
                loss.backward()
                optimizer.step()
            sum_loss += loss.item()*x.shape[0] 
            sum_tot += x.shape[0] 
            if False:
                pred = y.argmax(1)
                sum_tp += (pred==t).sum().item()
                sum_fp += len(t)                    
                sum_fn += len(pred)
            else:
                y = torch.sigmoid(y)
                top, _ = y.topk(3,1)
                thresh = top[:,-1].mean().item()
                pred = y > thresh
                sum_tp += ((pred==1) & (t==1)).sum().item()
                sum_fp += t.sum().item()                    
                sum_fn += pred.sum().item()
            recall = sum_tp/sum_fp*100 if sum_tp else 0
            prec = sum_tp/sum_fn*100 if sum_tp else 0
            if i%10==0: 
                print(f'{i}\t{(time()-t0)/(i+1):1.2f}s\t{sum_loss/sum_tot:1.4e}\t{recall:1.4f}\t{prec:1.4f}\t{thresh:1.4f}')
        print(f'{phase}({e})\t{(time()-t0):1.2f}s\t{sum_loss/sum_tot:1.4e}\t{recall:1.4f}\t{prec:1.4f}')
    torch.save(model.state_dict(), save/f'weight_{e}.pt')
    scheduler.step()

#break #ifold
