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


sentences = ['안녕하세요 반갑습니다 이지윤입니다']

checkpoint_dir1 = './checkpoint/1'
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)


def test_step(text, idx):
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

for i, text in enumerate(sentences):
    jamo = ''.join(list(hangul_to_jamo(text)))
    test_step(jamo, i)


checkpoint_dir2 = './checkpoint/2'
mel_list = glob.glob(os.path.join(save_dir, '*.npy'))


def test_step(mel, idx):
    mel = np.expand_dims(mel, axis=0)
    pred2 = model(mel, is_training=False)

    pred2 = np.squeeze(pred2, axis=0)
    pred2 = np.transpose(pred2)

    pred2 = (np.clip(pred2, 0, 1) * max_db) - max_db + ref_db
    pred2 = np.power(10.0, pred2 * 0.05)
    wav = griffin_lim(pred2 ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    #endpoint = librosa.effects.split(wav, frame_length=win_length, hop_length=hop_length)[0, 1]
    #wav = wav[:endpoint]
    wav = wav.astype(np.float32)
    scipy.io.wavfile.write(os.path.join(save_dir, '{}.wav'.format(idx)), sample_rate, wav)

model = post_CBHG(K=8, conv_dim=[256, mel_dim])
optimizer = Adam()
step = tf.Variable(0)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir2)).expect_partial()

for i, fn in enumerate(mel_list):
    mel = np.load(fn)
    test_step(mel, i)
