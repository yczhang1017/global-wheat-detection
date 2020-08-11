import os
import sys
import random
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence
from time import time
from sklearn.model_selection import StratifiedKFold
#import ast
#import librosa
#import librosa.display
 
root = sys.argv[1] if len(sys.argv)>1 else './'
os.chdir(root)
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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
    if label not in label2code.keys():
        label2code[label] = code
        code2label[code] = label
        code2tag[code] = tag
        label2tag[label] = tag
        tag2code += [code]
        tag += 1
    
ntag = len(tag2code)
ndim = 128
mlen = 441
batch_size = 32
epoch = 20
df_train['tag'] = df_train['ebird_code'].map(code2tag)
ndist = df_train.groupby('tag').count()['rating'].values
weight = torch.tensor(np.exp(((100/ndist)-1)/10), dtype=torch.float).to(device)

class TrainData(Dataset):
    def __init__(self,df,indices):
        self.df = df
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx, mlen = mlen, encode_tag = True, mosaic = 2, image = True):
        rows = [self.df.loc[self.indices[idx]]]
        for i in range(mosaic-1):
            rows += [self.df.loc[random.choice(self.indices)]]
        cur = 0
        x = -4*torch.ones((mlen,ndim))
        t = torch.zeros((ntag))
        for i, row in enumerate(rows):
            filename = os.path.join('tensors', row['filename'][:-3]+'pt')
            Sdb = torch.load(filename)
            Sdb = (Sdb+20)/12
            l = Sdb.shape[0]
            c = int(mlen // mosaic * random.uniform(0.8,1.2)) 
            if l > c:
                s = random.randrange(0,l-c)
                x[cur:cur+c,:] = Sdb[s:s+c,:]
            elif l < mlen:
                x[cur:cur+l,:] = Sdb
            t[row['tag']] = 1    
        if image: x = x.view((3,-1,ndim))
        return x,t


def adjust_learning_rate(optimizer, e, warmup=1, Tmax=epoch-1):
    if e <= warmup:
        lr = 1e-6
    else:
        lr = 1e-6/2*(1+np.cos((e-warmup)*np.pi/Tmax))
    print(f'learnig rate={lr:1.3e}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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


skf = StratifiedKFold(n_splits=5)
for ifold, (train_indices, val_indices) in enumerate(skf.split(df_train.index, df_train['tag'])):
    save =f'ResNeSt{ifold}'
    if not os.path.exists(save):
        os.mkdir(save)
    
    dataset = {'train':TrainData(df_train, train_indices),
                'val':TrainData(df_train, val_indices)}
    data_loader = {x: DataLoader(dataset[x],
            batch_size=batch_size, shuffle = (x=='train'),
            num_workers=4,pin_memory=True)
            for x in ['train', 'val']}
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    model.fc = nn.Linear(2048,ntag)
    #model = BertModel()
    model.to(device)
    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-6,weight_decay=1e-3)
    best_acc = 0
    for e in range(epoch):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
                adjust_learning_rate(optimizer,e)
            else:
                model.eval()
            sum_loss = 0
            sum_tot = 0
            sum_correct = 0
            t0 = time()
            print(f'fold{ifold}|{e}/{epoch}:{phase}')
            for i,(x,t) in enumerate(data_loader[phase]):
                x = x.to(device)
                t = t.to(device)
                y = model(x)
                loss = criterion(y, t)
                with torch.set_grad_enabled(phase == 'train'):
                    loss.backward()
                    optimizer.step()
                pred = y.argmax(1)
                sum_loss += loss.item()
                sum_tot += len(t)
                sum_correct += (pred == t).sum().item()
                if i%10==0: 
                    print(f'{i}\t{(time()-t0)/(i+1):1.2f}s\t{sum_loss/sum_tot:1.4f}\t{sum_correct/sum_tot*100:1.4f}')
            print(f'{phase}({e})\t{(time()-t0)}s\t{sum_loss/sum_tot:1.4f}\t{sum_correct/sum_tot*100:1.4f}')
        torch.save(model.state_dict(), os.path.join(save,'weight_{}.pt'.format(e)))

    break #ifold
