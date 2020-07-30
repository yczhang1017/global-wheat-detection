import librosa
import os 
import sys
import numpy as np
import torch
from tqdm import tqdm
mel_parameters = {
    "fmin": 40,
    "fmax": 14000
}    

def sound2tensor(filename,ndim=128):
    x, sr = librosa.load(filename)
    S = librosa.feature.melspectrogram(x,sr=sr,n_mels=ndim,**mel_parameters)
    Sdb = librosa.power_to_db(S).astype(np.float).transpose()
    return torch.tensor(Sdb)

root = sys.argv[1] if len(sys.argv)>1 else './'
os.chdir(root)
save = 'tensors'
os.mkdir(save)
for t in tqdm(os.listdir('train_audio')):
    for f in os.listdir(os.path.join('train_audio',t)):
        x = sound2tensor(os.path.join('train_audio',t,f))
        torch.save(x, os.path.join(save, f[:-3]+'pt'))

for f in os.listdir('example_test_audio'):
    x = sound2tensor(os.path.join('example_test_audio',f))
    torch.save(x, os.path.join(save, f[:-3]+'pt'))


    