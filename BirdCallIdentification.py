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
npos = 8
mlen = 800
batch_size = 4
dm = ndim+npos


mel_parameters = {
    "fmin": 100,
    "fmax": 10000
}    

def mask(x, mask_value):
    mask = (torch.rand((x.size(0),1))>0.9)
    mask[0,0] = False
    x[mask.expand(x.shape)] = mask_value
    return x
    
class TrainData1(Dataset):
    def __init__(self,df,mask=True):
        self.df = df
        self.mask = mask
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx, mlen = mlen):
        row = self.df.loc[idx]
        code = row['ebird_code']
        filename = os.path.join('tensors', row['filename'][:-3]+'pt')
        code = row['ebird_code']
        tag = code2tag[code]
        Sdb = torch.load(filename)
        #labels = ast.literal_eval(row['secondary_labels'])
        #tags = [label2tag[l] for l in labels]
        l = Sdb.shape[0]
        if l > mlen:
            s = random.randrange(0,l-mlen)
            Sdb = Sdb[s:s+mlen,:]
        Sdb = (Sdb+20)/12
        if 1==1:
            START = torch.ones((1,ndim))*4
            Sdb = torch.cat((START,Sdb), dim=0)
            Sm = Sdb.clone().detach()
            Sm = mask(Sm,-4)
            l = Sdb.shape[0]
            x = torch.linspace(0,l-1,l).view((l,1))
            ps = [torch.sin(2*np.pi*x/2*1.8*i) for i in range(npos)]
            Sdb = torch.cat([Sdb]+ps, dim = 1)
            Sm = torch.cat([Sm]+ps, dim = 1)
        return Sdb, Sm, tag


def pad_collate(batch, pad = -5):
    x,xm,t = zip(*batch)
    x_pad = pad_sequence(x, padding_value=pad)
    xm_pad = pad_sequence(x, padding_value=pad)
    return x_pad,xm_pad, t

class myModel(nn.Module):
    def __init__(self,dm,ntag):
        super(myModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(dm, 8, 512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ext = nn.Linear(dm,ntag)
    def forward(self,x):
        h = self.encoder(x)
        h0,h1 = h.split([1,x.shape[0]-1])
        y = self.ext(h0.squeeze(0))
        return y,h1
        
dataset = TrainData1(df_train)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
model = myModel(dm, ntag)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-3)
celoss = torch.nn.CrossEntropyLoss().cuda()
mseloss = torch.nn.MSELoss().cuda()
epoch = 10
save ='bert'
if not os.path.exists(save):
    os.mkdir(save)

for e in range(epoch):
    sum_loss1 = 0
    sum_loss2 = 0
    sum_tot = 0
    sum_correct = 0
    t0 = time()
    print(f'{e}/{epoch}')
    for i,(x, xm, t) in enumerate(data_loader):
        model.train()
        x = x.to(device)
        xm = xm.to(device)
        t = torch.tensor(t).to(device)
        optimizer.zero_grad()
        y,h1 = model(xm)        
        loss1 = celoss(y, t)
        loss2 = mseloss(h1, x[1:,:,:])
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        sum_loss1 += loss1
        sum_loss2 += loss2
        pred = y.argmax(1)
        sum_tot += len(t)
        sum_correct += (pred == t).sum().item()
        if i%10==0: 
            print(f'{i}\t{(time()-t0)/(i+1):1.2f}s\t{sum_loss1/sum_tot:1.4f}\t{sum_loss2/sum_tot:1.4f}\t{sum_correct/sum_tot*100:1.4f}')
    torch.save(model.state_dict(), os.join(save,'weight_{}.pt'.format(e)))