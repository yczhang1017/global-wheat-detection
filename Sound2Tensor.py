import librosa
from pydub import AudioSegment
import os 
import sys
import numpy as np
import torch
from tqdm import tqdm
mel_parameters = {
    "fmin": 40,
    "fmax": 14000
}    


def sound2tensor(filename,ndim=128, sr = 44100):
    audio = AudioSegment.from_mp3(filename)
    sample = audio.set_channels(1).set_frame_rate(sr).get_array_of_samples()
    x = np.array(sample).astype(np.float32)/2**14
    S = librosa.feature.melspectrogram(x,sr,n_mels=ndim,**mel_parameters)
    Sdb = librosa.power_to_db(S).astype(np.float32).transpose()
    return torch.tensor(Sdb)

root = sys.argv[1] if len(sys.argv)>1 else './'
os.chdir(root)
save = 'tensors'
if not os.path.exists(save):
    os.mkdir(save)
for t in tqdm(os.listdir('train_audio')):
    for f in os.listdir(os.path.join('train_audio',t)):
        try:
            x = sound2tensor(os.path.join('train_audio',t,f))
        except: 
            print(f'{f} cannot be decoded')
        torch.save(x, os.path.join(save, f[:-3]+'pt'))

for f in os.listdir('example_test_audio'):
    try:
        x = sound2tensor(os.path.join('example_test_audio',f))
    except: 
        print(f'{f} cannot be decoded')
    torch.save(x, os.path.join(save, f[:-3]+'pt'))


    