import os
import torch
import shutil
from abc import *
import sentencepiece as spm
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_or_get_voca(save_path, ko_corpus_path=None, di_corpus_path=None, ko_vocab_size=4000, di_vocab_size=4000,
                       region='gy'):
    ko_corpus_prefix = 'ko_corpus_{}'.format(ko_vocab_size)  # vocab_size를 바꾸면 embedding_size도 변경
    di_corpus_prefix = '{}_corpus_{}'.format(region, di_vocab_size)

    if ko_corpus_path and di_corpus_path:
        templates = '--input={} --model_prefix={} --vocab_size={} ' \
                    '--bos_id=0 --eos_id=1 --unk_id=2 --pad_id=3'
        # input : 학습시킬 텍스트의 위치
        # model_prefix : 만들어질 모델 이름
        # vocab_size : 사전의 크기
        ko_model_train_cmd = templates.format(ko_corpus_path, ko_corpus_prefix, ko_vocab_size)
        di_model_train_cmd = templates.format(di_corpus_path, di_corpus_prefix, di_vocab_size)

        spm.SentencePieceTrainer.Train(ko_model_train_cmd)  # Korean 텍스트를 가지고 학습
        spm.SentencePieceTrainer.Train(di_model_train_cmd)  # English 텍스트를 가지고 학습

        # 파일을 저장위치로 이동
        shutil.move(ko_corpus_prefix + '.model', save_path)
        shutil.move(ko_corpus_prefix + '.vocab', save_path)
        shutil.move(di_corpus_prefix + '.model', save_path)
        shutil.move(di_corpus_prefix + '.vocab', save_path)

    ko_sp = spm.SentencePieceProcessor()
    di_sp = spm.SentencePieceProcessor()
    ko_sp.load(os.path.join(save_path, ko_corpus_prefix + '.model'))  # model load
    di_sp.load(os.path.join(save_path, di_corpus_prefix + '.model'))  # model load
    return ko_sp, di_sp


class TranslationDataset(Dataset, metaclass=ABCMeta):
    # Translation Dataset abstract class
    # download, read data 등등을 하는 파트.
    def __init__(self, x_path, y_path, ko_voc, di_voc, sequence_size):
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.y = open(y_path, 'r', encoding='utf-8').readlines()  # English data 위치
        self.ko_voc = ko_voc  # Korean 사전
        self.en_voc = di_voc  # English 사전
        self.sequence_size = sequence_size  # sequence 최대길이

    def __len__(self):  # data size를 넘겨주는 파트
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    @abstractmethod
    def encoder_input_to_vector(self, sentence: str): pass

    @abstractmethod
    def decoder_input_to_vector(self, sentence: str): pass

    @abstractmethod
    def decoder_output_to_vector(self, sentence: str): pass


class LSTMSeq2SeqDataset(TranslationDataset):
    # for Seq2Seq model & Seq2Seq attention model
    # using google sentencepiece (https://github.com/google/sentencepiece.git)
    def __init__(self, x_path, y_path, ko_voc, en_voc, sequence_size):
        super().__init__(x_path, y_path, ko_voc, en_voc, sequence_size)
        self.KO_PAD_ID = ko_voc['<pad>']  # 3 Padding
        self.EN_PAD_ID = en_voc['<pad>']  # 3 Padding
        self.EN_BOS_ID = en_voc['<s>']  # 0 Start Token
        self.EN_EOS_ID = en_voc['</s>']  # 1 End Token

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.ko_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.insert(0, self.EN_BOS_ID)  # Start Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str):
        idx_list = self.en_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.EN_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list


class LSTMTestDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, x_path, ko_voc=None, sequence_size=80):
        self.KO_PAD_ID = ko_voc['<pad>']  # 3 Padding
        self.EN_EOS_ID = ko_voc['</s>']  # 1 End Token
        self.x = open(x_path, 'r', encoding='utf-8').readlines()  # korean data 위치
        self.ko_voc = ko_voc  # Korean 사전
        self.sequence_size = sequence_size  # sequence 최대길이

    def __len__(self):  # data size를 넘겨주는 파트
        return len(self.x)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        return encoder_input

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.ko_voc.EncodeAsIds(sentence)  # str -> idx
        idx_list.append(self.EN_EOS_ID)  # End Token 삽입
        idx_list = self.padding(idx_list, self.KO_PAD_ID)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        return None

    def decoder_output_to_vector(self, sentence: str):
        return None

    def padding(self, idx_list, padding_id):
        length = len(idx_list)  # 리스트의 길이
        if length < self.sequence_size:
            # sentence가 sequence_size가 작으면 나머지를 padding token으로 채움
            idx_list = idx_list + [padding_id for _ in range(self.sequence_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.sequence_size]
        return idx_list
