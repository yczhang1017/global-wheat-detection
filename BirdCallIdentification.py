import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

#import ast
#import librosa
#import librosa.display

root = sys.argv[1] if len(sys.argv)>1 else './'
os.chdir(root)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.columns)


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
dm = ndim+npos
batch_size = 4

mel_parameters = {
    "fmin": 100,
    "fmax": 10000
}    


def mask(x, mask_value):
    mask = (torch.rand((x.size(0),1,1))>0.9).expand(x.shape)
    x[mask] = mask_value
    return x


    
class TrainData1(Dataset):
    def __init__(self,df,mask=True):
        self.df = df
        self.mask = mask
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx, mlen = 512):
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
            #START = torch.ones((1,ndim))
            #Sdb = torch.cat((START,Sdb), dim=0)
            l = Sdb.shape[0]
            x = torch.linspace(0,l-1,l).view((l,1))
            ps = [torch.sin(2*np.pi*x/2*1.8*i) for i in range(npos)]
            Sdb = torch.cat([Sdb]+ps, dim = 1)
        return Sdb, tag


def pad_collate(batch, pad = -4):
    x,t = zip(*batch)
    x_pad = pad_sequence(x, padding_value=pad)
    return x_pad, t

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

optimizer = torch.optim.Adam(model.parameters(),lr=1e-2,weight_decay=1e-3)
celoss = torch.nn.CrossEntropyLoss().cuda()
mseloss = torch.nn.MSELoss().cuda()
epoch = 10

for e in range(epoch):
    sum_loss1 = 0
    sum_loss2 = 0
    sum_tot = 0
    sum_correct = 0
    print(f'{e}/{epoch}')
    for i,(x, t) in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.train()
        x = x.to(device)
        xo = x.clone().detach()
        x = mask(x,-5)
        xo = xo[1:,:,:]
        t = torch.tensor(t).to(device)
        optimizer.zero_grad()
        y,h1 = model(x)        
        loss1 = celoss(y, t)
        loss2 = mseloss(h1, xo)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        sum_loss1 += loss1
        sum_loss2 += loss2
        pred = y.argmax(1)
        sum_tot += len(t)
        sum_correct += (pred == t).sum().item()
        if i%1==0: print(f'{i},{sum_loss1/sum_tot},{sum_loss2/sum_tot},{sum_correct/sum_tot*100}')

   