import os, librosa, glob, scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from jamo import hangul_to_jamo
from models.tacotron import Tacotron
from util.hparams import *
from util.plot_alignment import plot_alignment
from util.text import sequence_to_text, text_to_sequence
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from models.tacotron import post_CBHG
from models.modules import griffin_lim

# 만들고 싶은 문장 입력 (기호쓰지않고 한글만)
sentences = ['안녕하세요 반갑습니다 이지윤입니다']

# test1,2에서 학습했던 모델을 불러올 경로 지정
checkpoint_dir1 = './checkpoint/1'
checkpoint_dir2 = './checkpoint/2'

# test결과를 저장할 폴더를 지정 후 생성
# exist_ok=True 인수를 주면 이미 디렉토리가 존재하더라도 오류가 발생하지 않습니다.
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)

# 문장을 가지고 sequence로 분리한 후 mel-spectrogram생성 및 npy로 저장해주는 함수
# 원래는 mel-spectrogram길이가 고정되어 있어서 문장뒤에 공백이 길게 나오고 png가 이상하게 생성되어서 임의적으로 적절히 잘리도록 수정함

# 원본코드
# def test_step(text, idx):
#     seq = text_to_sequence(text)
#     enc_input = np.asarray([seq], dtype=np.int32)
#     sequence_length = np.asarray([len(seq)], dtype=np.int32)
#     dec_input = np.zeros((1, max_iter, mel_dim), dtype=np.float32)

#     pred = []
#     for i in range(1, max_iter+1):
#         mel_out, alignment = model(enc_input, sequence_length, dec_input, is_training=False)
#         if i < max_iter:
#             dec_input[:, i, :] = mel_out[:, reduction * i - 1, :]
#         pred.extend(mel_out[:, reduction * (i-1) : reduction * i, :])

#     pred = np.reshape(np.asarray(pred), [-1, mel_dim])
#     alignment = np.squeeze(alignment, axis=0)

#     np.save(os.path.join(save_dir, 'mel-{}'.format(idx)), pred, allow_pickle=False)

#     input_seq = sequence_to_text(seq)
#     alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
#     plot_alignment(alignment, alignment_dir, input_seq)

# 수정 코드
def test_step1(text, idx):
    seq = text_to_sequence(text)
    enc_input = np.asarray([seq], dtype=np.int32)
    sequence_length = np.asarray([len(seq)], dtype=np.int32)
    dec_input = np.zeros((1, sequence_length[0]+8, mel_dim), dtype=np.float32)

    pred21 = []
    for i in range(1, sequence_length[0]+9):
        mel_out, alignment = model(enc_input, sequence_length, dec_input, is_training=False)
        if i < sequence_length[0]+8:
            dec_input[:, i, :] = mel_out[:, reduction * i - 1, :]
        pred21.extend(mel_out[:, reduction * (i-1) : reduction * i, :])

    pred21 = np.reshape(np.asarray(pred21), [-1, mel_dim])
    alignment = np.squeeze(alignment, axis=0)

    np.save(os.path.join(save_dir, 'mel-{}'.format(idx)), pred21, allow_pickle=False)

    input_seq = sequence_to_text(seq)
    alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
    plot_alignment(alignment, alignment_dir, input_seq)


model = Tacotron(K=16, conv_dim=[128, 128])
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir1)).expect_partial()

# 문장을 자모로 바꿔준다음 test_step1에 넣고 mel-spectrogram 및 npy배열 생성
# 문장이 길다면 배열로 적당히 나눠서 생성하기
for i, text in enumerate(sentences):
    jamo = ''.join(list(hangul_to_jamo(text)))
    test_step1(jamo, i)


# 위에서 생성된 npy파일을 불러옴
mel_list = glob.glob(os.path.join(save_dir, '*.npy'))


# 불러온 npy를 가지고 wav파일로 만들어주는 함수
def test_step2(mel, idx):
    mel = np.expand_dims(mel, axis=0)
    pred2 = model(mel, is_training=False)

    pred2 = np.squeeze(pred2, axis=0)
    pred2 = np.transpose(pred2)

    pred2 = (np.clip(pred2, 0, 1) * max_db) - max_db + ref_db
    pred2 = np.power(10.0, pred2 * 0.05)
    wav = griffin_lim(pred2 ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)

    # 아래 두줄은 문장을 다읽고 뒤에 공백이 있으면 제거해주는 함수이나 가끔 띄워쓰기에서도 잘라내는 경우가 있어서 주석처리
    # 특히 위에서 이미 길이를 적절히 잘라주었기 때문에도 필요없어진 코드
    # endpoint = librosa.effects.split(wav, frame_length=win_length, hop_length=hop_length)[0, 1]
    # wav = wav[:endpoint]
    wav = wav.astype(np.float32)
    scipy.io.wavfile.write(os.path.join(save_dir, '{}.wav'.format(idx)), sample_rate, wav)

model = post_CBHG(K=8, conv_dim=[256, mel_dim])
optimizer = Adam()
step = tf.Variable(0)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir2)).expect_partial()

# mel_list를 test_step2에 넣고 wav파일 생성
for i, fn in enumerate(mel_list):
    mel = np.load(fn)
    test_step2(mel, i)
