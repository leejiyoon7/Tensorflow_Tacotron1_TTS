import pandas as pd
import numpy as np
import os, librosa, re, glob
from tqdm import tqdm
from util.hparams import *
from util.text import text_to_sequence


# 코드가 있는 경로의 /kss폴더 지정후 그 안에 모든 txt파일을 찾습니다.
text_dir = glob.glob(os.path.join('./kss', '*.txt'))
# 문장에서 걸러내고 싶은 기호들을 적습니다.
filters = '([.,!?])'

# txt파일을 불러와서 |로 구분한 후 경로와 문장을 구분해 저장합니다.
metadata = pd.read_csv(text_dir[0], dtype='object', sep='|', header=None)
wav_dir = metadata[0].values
text = metadata[3].values

# data폴더를 만든 후 그안에 각각 text, mel, dec, spec폴더를 만들어 줍니다.
# exist_ok=True 인수를 주면 이미 디렉토리가 존재하더라도 오류가 발생하지 않습니다.
out_dir = './data'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir + '/text', exist_ok=True)
os.makedirs(out_dir + '/mel', exist_ok=True)
os.makedirs(out_dir + '/dec', exist_ok=True)
os.makedirs(out_dir + '/spec', exist_ok=True)

# text 전처리
print('Load Text')
text_len = []
# 위에서 저장한 문장의 개수만큼 for문 실행
for idx, s in enumerate(tqdm(text)):
    # 문장에서 filters애 등록했던 문자들을 제거
    sentence = re.sub(re.compile(filters), '', s)
    # text_to_sequence를 이용하여 문장을 분리
    sentence = text_to_sequence(sentence)
    # 분리된 sequence의 길이를 배열에 저장
    text_len.append(len(sentence))
    # 파일의 이름 지정 후 text폴더에 저장
    text_name = 'kss-text-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/text', text_name), sentence, allow_pickle=False)
# 모든 저장이 끝난 후 sequence의 길이를 저장해둔 배열도 저장
np.save(os.path.join(out_dir + '/text_len.npy'), np.array(text_len))
print('Text Done')

# audio
print('Load Audio')
mel_len_list = []
for idx, fn in enumerate(tqdm(wav_dir)):
    file_dir = './kss/'+ fn
    wav, _ = librosa.load(file_dir, sr=sample_rate)
    wav, _ = librosa.effects.trim(wav)
    wav = np.append(wav[0], wav[1:] - preemphasis * wav[:-1])
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft = np.abs(stft)
    mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_dim)
    mel_spec = np.dot(mel_filter, stft)

    mel_spec = 20 * np.log10(np.maximum(1e-5, mel_spec))
    stft = 20 * np.log10(np.maximum(1e-5, stft))

    mel_spec = np.clip((mel_spec - ref_db + max_db) / max_db, 1e-8, 1)
    stft = np.clip((stft - ref_db + max_db) / max_db, 1e-8, 1)

    mel_spec = mel_spec.T.astype(np.float32)
    stft = stft.T.astype(np.float32)
    mel_len_list.append([mel_spec.shape[0], idx])

    # padding
    remainder = mel_spec.shape[0] % reduction
    if remainder != 0:
        mel_spec = np.pad(mel_spec, [[0, reduction - remainder], [0, 0]], mode='constant')
        stft = np.pad(stft, [[0, reduction - remainder], [0, 0]], mode='constant')

    mel_name = 'kss-mel-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

    stft_name = 'kss-spec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/spec', stft_name), stft, allow_pickle=False)

    # Decoder Input
    mel_spec = mel_spec.reshape((-1, mel_dim * reduction))
    dec_input = np.concatenate((np.zeros_like(mel_spec[:1, :]), mel_spec[:-1, :]), axis=0)
    dec_input = dec_input[:, -mel_dim:]
    dec_name = 'kss-dec-%05d.npy' % idx
    np.save(os.path.join(out_dir + '/dec', dec_name), dec_input, allow_pickle=False)

mel_len = sorted(mel_len_list)
np.save(os.path.join(out_dir + '/mel_len.npy'), np.array(mel_len))
print('Audio Done')
