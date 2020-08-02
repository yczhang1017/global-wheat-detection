import os
import sys
import random
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
mlen = 444
batch_size = 32
epoch = 30
class TrainData1(Dataset):
    def __init__(self,df,indices):
        self.df = df
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx, mlen = mlen):
        row = self.df.loc[self.indices[idx]]
        code = row['ebird_code']
        filename = os.path.join('tensors', row['filename'][:-3]+'pt')
        tag = code2tag[code]
        Sdb = torch.load(filename)
        l = Sdb.shape[0]
        Sdb = (Sdb+20)/12
        if l > mlen:
            s = random.randrange(0,l-mlen)
            Sdb = Sdb[s:s+mlen,:]
        elif l < mlen:
            Sdb = torch.cat((Sdb, -4*torch.ones((mlen-l,ndim))),dim=0)
        a,b = Sdb.shape
        Sdb = Sdb.view((3,-1,b))
        return Sdb, torch.tensor(tag)


def adjust_learning_rate(optimizer, e, lr0=1e-3, warmup=5, Tmax=epoch-5):
    if e <= warmup:
        lr = 1e-3 - (1e-3 - 1e-5)*(warmup-e)/warmup
    else:
        lr = 1e-3/2*(1+np.cos((e-warmup)*np.pi/Tmax))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class BertModel(nn.Module):
    def __init__(self,ndim=ndim,ntag=ntag,npos=8):
        super(myModel, self).__init__()
        self.dm = ndim + npos
        encoder_layer = nn.TransformerEncoderLayer(self.dm, 16, 512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ext = nn.Linear(dm,ntag)
    def forward(self,x):
        h = self.encoder(x)
        h0,h1 = h.split([1,x.shape[0]-1])
        y = self.ext(h0.squeeze(0))
        return y,h1        


skf = StratifiedKFold(n_splits=5)
df_train['tag'] = df_train['ebird_code'].map(code2tag)
for ifold, (train_indices, val_indices) in enumerate(skf.split(df_train.index, df_train['tag'])):
    save =f'ResNeSt{ifold}'
    if not os.path.exists(save):
        os.mkdir(save)
    
    dataset = {'train':TrainData1(df_train, train_indices),
                'val':TrainData1(df_train, val_indices)}
    data_loader = {x: DataLoader(dataset[x],
            batch_size=batch_size, shuffle = (x=='train'),
            num_workers=4,pin_memory=True)
            for x in ['train', 'val']}
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    model.fc = nn.Linear(2048,ntag)
    model.to(device)
    celoss = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
    for e in range(epoch):
        for phase in ['train','val']:
            sum_loss = 0
            sum_tot = 0
            sum_correct = 0
            t0 = time()
            print(f'fold{ifold}|{e}/{epoch}:{phase}')
            for i,(x,t) in enumerate(data_loader[phase]):
                if phase == 'train':
                    model.train()
                    adjust_learning_rate(optimizer,e)
                else:
                    model.eval()
                x = x.to(device)
                t = t.to(device)
                y = model(x)
                loss = celoss(y, t)
                with torch.set_grad_enabled(phase == 'train'):
                    loss.backward()
                    optimizer.step()
                pred = y.argmax(1)
                sum_loss += loss.item()
                sum_tot += len(t)
                sum_correct += (pred == t).sum().item()
                if i%10==0: 
                    print(f'{i}\t{(time()-t0)/(i+1):1.2f}s\t{sum_loss/sum_tot:1.4f}\t{sum_correct/sum_tot*100:1.4f}')
        if phase == 'train':
            torch.save(model.state_dict(), os.path.join(save,'weight_{}.pt'.format(e)))

    break #ifold
