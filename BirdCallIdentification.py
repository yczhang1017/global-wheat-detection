
import random
import numpy as np
import pandas as pd
import torch 
from tqdm.auto import tqdm
import warnings
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import ast
import librosa.display

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/birdsong-recognition/train.csv')
df_test = pd.read_csv('../input/birdsong-recognition/test.csv')
print(df_train.columns)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed = 42):
    #random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    
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
    
ndim = 128
mel_parameters = {
    "fmin": 100,
    "fmax": 10000
}    


def gen_mask(x):
    mask = torch.zeros(x.shape,dtype=torch.bool)
    l = x.shape[0]
    num = random.randrange(5,10)
    for i in range(num):
        s = random.randrange(l)
        k = random.randrange(max(min(l//100,20),1))
        mask[s:s+k+1] = True
    x_masked = x.clone().detach()
    x_masked[mask] = 0
    return x_masked
def shift_left(x,pad=0):
    x = torch.roll(x.clone().detach(), -1, 0)
    x[-1,:, :] = pad
    return x


class TrainData1(Dataset):
    def __init__(self,df,mask=True, root='/kaggle/input/birdsong-recognition/train_audio/'):
        self.df = df
        self.root = root
        self.mask = mask
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx, maxlen= 2048):
        row = self.df.loc[idx]
        code = row['ebird_code']
        filename = os.path.join(self.root, code, row['filename'])
        code = row['ebird_code']
        tag = code2tag[code]
        #labels = ast.literal_eval(row['secondary_labels'])
        #tags = [label2tag[l] for l in labels]
        x, sr = librosa.load(filename)
        S = librosa.feature.melspectrogram(x,sr=sr,n_mels=ndim,**mel_parameters)
        Sdb = librosa.power_to_db(S).astype(np.float32).transpose()
        l = Sdb.shape[0]
        if l > maxlen:
            s = random.randrange(0,l-maxlen)
            Sdb = Sdb[s:s+maxlen,:]
        Sdb = (Sdb+34.2973)/9.9774
        Sdb = torch.tensor(Sdb)     
        START = torch.ones((1,ndim))
        Sdb = torch.cat((START,Sdb), dim=0)
        l = Sdb.shape[0]
        x = torch.linspace(0,l-1,l).view((l,1))
        p1 = torch.cos(x/3*2*np.pi)
        p2 = torch.cos(x/5*2*np.pi)
        p3 = torch.cos(x/7*2*np.pi)
        p4 = torch.cos(x/11*2*np.pi)
        p5 = torch.cos(x/23*2*np.pi)
        p6 = torch.cos(x/34*2*np.pi)
        p7 = torch.cos(x/45*2*np.pi)
        p8 = torch.cos(x/56*2*np.pi)
        Sdb = torch.cat((Sdb,p1,p2,p3,p4,p5,p6,p7,p8), dim = 1)
        return Sdb, tag
    
from torch.nn.utils.rnn import pad_sequence

pad = -2.3752
def pad_collate(batch):
    x,t = zip(*batch)
    x_pad = pad_sequence(x, padding_value=pad)
    return x_pad, t

dataset = TrainData1(df_train)
#x,mask,tag = dataset[0]
#librosa.display.specshow(x.numpy().transpose(),y_axis='mel', x_axis='time', **mel_parameters)
data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)
model = torch.nn.Transformer(d_model=136, nhead=8, num_encoder_layers=4, num_decoder_layers=4, 
                             dim_feedforward=512, dropout=0.1, activation='gelu')

model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2, 
            momentum=0.9, weight_decay=1e-3)

criterion = torch.nn.MSELoss()
epoch = 10
sum_loss = 0
for e in range(epoch):
    print(f'{e}/{epoch}')
    for i,(x, t) in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.train()
        x = x.to(device)
        xm =gen_mask(x)
        y = shift_left(x, pad=pad)
        optimizer.zero_grad()
        output = model(xm, x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        sum_loss += loss
        if i%10: print(f'loss:{sum_loss/i}')

    