# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from source.translation.tranliteration.data_helper \
    import create_or_get_voca, sentence_to_token_ids
from source.translation.tranliteration.model import Encoder, Decoder, Seq2Seq


class Transliteration(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, x_path=None,  beam_search=False, k=1):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = self.checkpoint['seq_len']
        self.batch_size = 20
        self.x_path = x_path
        self.beam_search = beam_search
        self.k = k
        self.en_voc = create_or_get_voca(vocabulary_path=dictionary_path + "vocab40.en")
        self.ko_voc = create_or_get_voca(vocabulary_path=dictionary_path + "vocab1000.ko")
        self.model = self.model_load()

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Seq2Seq(encoder, decoder, self.seq_len, beam_search=self.beam_search, k=self.k)
        model = nn.DataParallel(model)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        return model

    def src_input(self, sentence):
        idx_list = sentence_to_token_ids(sentence, self.en_voc)
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list])

    def tar_input(self):
        idx_list = [self.en_voc['<s>']]
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list])

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def tensor2sentence(self, indices: torch.Tensor) -> list:
        translation_sentence = []
        vocab = {v: k for k, v in self.ko_voc.items()}
        for idx in indices:
            if idx != 1:
                translation_sentence.append(vocab[idx])
            else:
                break
        translation_sentence = ''.join(translation_sentence).strip()
        return translation_sentence

    def transform(self, sentence: str) -> (str, torch.Tensor):
        src_input = self.src_input(sentence)
        tar_input = self.tar_input()
        output = self.model(src_input, tar_input, teacher_forcing_rate=0)
        if isinstance(output, tuple):  # attention이 같이 출력되는 경우 output만
            output = output[0]
        _, indices = output.view(-1, output.size(-1)).max(-1)
        pred = self.tensor2sentence(indices.tolist())

        print('Korean: ' + sentence)
        print('Predict: ' + pred)
        return pred
