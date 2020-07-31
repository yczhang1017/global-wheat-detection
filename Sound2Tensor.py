import librosa
from pydub import AudioSegment
import os 
import sys
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
mel_parameters = {
    "fmin": 40,
    "fmax": 14000
}    


def sound2tensor(filename, form='mp3', ndim=128, sr = 44100):
    if form== 'mp3':
        audio = AudioSegment.from_wp3(filename)
    elif form == 'wav': 
        audio = AudioSegment.from_wav(filename)
    sample = audio.set_channels(1).set_frame_rate(sr).get_array_of_samples()
    
    x = np.array(sample).astype(np.float32)/2**14
    S = librosa.feature.melspectrogram(x,sr,n_mels=ndim,**mel_parameters)
    Sdb = librosa.power_to_db(S).astype(np.float32).transpose()
    return torch.tensor(Sdb)

def df2tensor(df,form='mp3'):
    for r in tqdm(df.iterrows(), total = len(df1)):
        t = r['ebird_code']
        f = r['filename'] 
        x = sound2tensor(os.path.join('train_audio',t,f), form)
        torch.save(x, os.path.join(save, f[:-3]+'pt'))
            

root = sys.argv[1] if len(sys.argv)>1 else './'
os.chdir(root)
save = 'tensors'
df = pd.read_csv('train.csv')
df1 = [df['file_type'] == 'wav']
df2 = [df['file_type'] != 'wav']
df2tensor(df1,'wav')
#df2tensor(df2,'mp3')

"""    
for f in os.listdir('example_test_audio'):
    x = sound2tensor(os.path.join('example_test_audio',f))
    torch.save(x, os.path.join(save, f[:-3]+'pt'))

"""
    