import librosa
from pydub import AudioSegment
import os 
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
mel_parameters = {
    "fmin": 40,
    "fmax": 14000
}    

import argparse
parser = argparse.ArgumentParser(description='Convert mp3 to tensor')
parser.add_argument('-d','--data', default='.', help='data directory')
parser.add_argument('-j','--job', default='example,train', help='jobs to convert')
parser.add_argument('-s','--save', default='tensors', help='output folder name')
parser.add_argument('--sr', default=44100, help='sample rate')
args = parser.parse_args()

root = Path(args.data)
save = root/args.save
if not save.exist(): save.mkdir()
def sound2tensor(filename, form='mp3', ndim=128, sr = args.sr):
    if form== 'mp3':
        audio = AudioSegment.from_mp3(filename)
    else: 
        audio = AudioSegment.from_file(filename, form)
    sample = audio.set_channels(1).set_frame_rate(sr).get_array_of_samples()
    x = np.array(sample).astype(np.float32)/2**14
    S = librosa.feature.melspectrogram(x,sr,n_mels=ndim,**mel_parameters)
    Sdb = librosa.power_to_db(S).astype(np.float32).transpose()
    return torch.tensor(Sdb)

def df2tensor(df,form='mp3'):
    for i,r in tqdm(df.iterrows(), total = len(df)):
        t = r['ebird_code']
        f = r['filename'] 
        try:
            x = sound2tensor(root/'train_audio'/t/f, form)
        except:
            print(f'{f} cannot be decoded')
        torch.save(x, save/(f[:-3]+'pt'))
            
def convertExample(df=df_example,form='mp3'):
    folder = root/'example_test_audio'
    for f in folder.iterdir():
        audio = AudioSegment.from_mp3(f).set_channels(1)
        duration = math.ceil(len(audio)/1000)
        for s in range(5,duration,5):
            sample = audio[(s-5)*1000:s*1000].set_frame_rate(args.sr).get_array_of_samples()
            torch.save(sound2tensor(sample), save/f.stem+'_'+str(s)+'.pt')

jobs= args.job.split(',')
if "train" in jobs:
    df = pd.read_csv(root/'train.csv')
    df1 = df[df['file_type'] == 'wav']
    df2 = df[df['file_type'] == 'aac']
    df3 = df[np.logical_or(df['file_type'] == 'mp3', df['file_type'] == 'mp4')]
    df2tensor(df1,'wav')
    df2tensor(df2,'aac')
if "example" in job.split(','):
    df = pd.read_csv(root/'example_test_audio_summary.csv')

#df2tensor(df2,'mp3')

"""    
for f in os.listdir('example_test_audio'):
    x = sound2tensor(os.path.join('example_test_audio',f))
    torch.save(x, os.path.join(save, f[:-3]+'pt'))

"""
    