from distributed import apply_gradient_allreduce
import time

import IPython.display as ipd

from numpy import finfo
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from text import hangul_to_sequence
from waveglow.denoiser import Denoiser
from scipy.io import wavfile

hparams=create_hparams()


def load_model(hparams):
	model = Tacotron2(hparams).cuda()
	if hparams.fp16_run:
		model.decoder.attention_layer.score_mask_value = finfo('float16').min
	if hparams.distributed_run:
		model = apply_gradient_allreduce(model)
	return model



def plot_data(data, figsize=(16, 4)):
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1, len(data), figsize=figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect='auto', origin='bottom',
					   interpolation='none')
	return plt

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	# proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))

if __name__ =='__main__':
	hparams = create_hparams()
	checkpoint_path = "outdir/male/jeju/checkpoint_70000"
	model = load_model(hparams)
	model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
	_ = model.cuda().eval().half()

	waveglow_path = 'waveglow/checkpoints/waveglow_146000'
	taco = checkpoint_path.split('_')[-1]
	wave = waveglow_path.split('_')[-1]


	waveglow = torch.load(waveglow_path)['model']
	waveglow.cuda().eval().half()
	for k in waveglow.convinv:
		k.float()
	denoiser = Denoiser(waveglow)

	# text = "야. 도로모깡도 왜정시대나 낫주. 도로모깡도 엇일 땐양 허벅에."
	text = '해산물 뭐 바다에 잇는 건 다 심어오고. 전복이고 해삼이고 문어고 원 그자 봐진 건 다 심어와, 먹는 건.'
	sequence = np.array(hangul_to_sequence(text))[None, :]
	sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

	start = time.time()
	mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

	print(text)
	print('mel output shape: {}'.format(mel_outputs.shape))
	print('Tacotron synthesize time: {}'.format(time.time() - start))
	start = time.time()
	with torch.no_grad():
		audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

	print('Wavenet synthesize time: {}'.format(time.time() - start))
	audio = denoiser(audio, strength=0.01)[:, 0]
	start = time.time()
	save_wav(audio[0].data.cpu().numpy(), 'output_{}_{}.wav'.format(taco, wave), sr=hparams.sampling_rate)
	print('Audio --> .wav file saving time: {}'.format(time.time() - start))
	# 	audio = audio[0].data.cpu().numpy()
	#
	# start = time.time()
	# save_wav(audio, 'output.wav', sr=hparams.sampling_rate)
	# print('audio saving time: {}'.format(time.time() - start))

	# start = time.time()
	# audio_denoised = denoiser(audio, strength=0.01)[:, 0]
	# ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
	# print('audio denoising time: {}'.format(time.time() - start))

	plt=plot_data((mel_outputs.float().data.cpu().numpy()[0],
			   mel_outputs_postnet.float().data.cpu().numpy()[0],
			   alignments.float().data.cpu().numpy()[0].T))
	plt.savefig('output_{}_{}.png'.format(taco, wave))

#
# def mel_to_audio(hparams, mel, sigma=0.1):
#
# 	upsample = torch.nn.ConvTranspose1d(hparams.n_mel_channels,hparams.n_mel_channels,1024, stride=256)
# 	spect = upsample(mel)