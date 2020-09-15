# -*- coding: cp949 -*-
import json
import sys
import time
from docutils.nodes import header
from numpy import finfo
import pandas as pd
from source.distributed import apply_gradient_allreduce
from speech_synthesis import Text2Speech
from source.db.app import TTS
sys.path.append('source/waveglow/')
import numpy as np
from scipy.io import wavfile
from flask import Flask, render_template, request
from source.hparams import create_hparams
hparams=create_hparams()
from source.model import Tacotron2
from translation import Korean2Dialect




def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wav = wav.astype(np.int16)
    wavfile.write(path, sr, wav)
    return wav


def uri_mapping(speaker: str, model_type: str, attitude_style: str) -> str:
    uri = 'source/outdir'
    if speaker.strip() == '남성':
        uri += '/male'
    elif speaker.strip() == '여성':
        uri += '/female'
    else:
        print('speaker {}'.format(speaker))
        print('uri {}'.format(uri))
        raise NotImplementedError("speaker is not implemented")

    if model_type.strip() == '표준어':
        uri += '/standard'
    elif model_type.strip() == '제주도':
        uri += '/jeju'
    elif model_type.strip() == '대구':
        uri += '/daegu'
    elif model_type.strip() == '경상북도':
        uri += '/gyeonsangbuk'
    elif model_type.strip() == '부산':
        uri += '/busan'
    else:
        print('uri {}'.format(uri))
        print('model_type {}'.format(model_type))
        raise NotImplementedError("model type is not implemented")




def clean_text(txt:str)->list:
    ### transform english char to korean text
    transform_dict = {'a':'에이','b':'비','c':'시','d':'디','e':'이','f':'에프','g':'지','h':'에이치','i':'아이','j':'제이','k':'케이','l':'엘','m':'엠',
                      'n':'엔','o':'오','p':'피', 'q':'큐','r':'아르','s':'에스','t':'티','u':'유','v':'브이','w':'더블유','x':'엑스','y':'와이','z':'제트',
                      u"'":u'"', '(':', ', ')':', ', '#':'샵', '%':'프로', '@':'고팽이', '+':'더하기', '-':'빼기', ':':'나누기', '*':'별'}
    ### remove not allowed chars
    not_allowed_characters = list('^~')
    txt = ''.join(i for i in txt if not i in not_allowed_characters)
    txt = txt.lower().strip()
    ### transform special char to hangul
    for k,v in transform_dict.items():
        txt=txt.replace(k, v).replace(' .', '.').replace(' ?', '?').strip()
    from koalanlp import API
    from koalanlp.proc import SentenceSplitter
    from koalanlp.Util import initialize, finalize
    #### split paragraph to list of sentences
    initialize(hnn="2.1.3")
    splitter = SentenceSplitter(api=API.HNN)
    paragraph = splitter(txt)
    finalize()
    # return paragraph
    txt_list=[]
    import string
    max_len=60
    for s in paragraph:
        txt_ = s.translate(str.maketrans('', '', string.punctuation.replace(',','')))
        txt_=txt_.strip()

        while True:
            if ',,' in txt_:
                txt_=txt_.replace(',,',',')
            else:
                break

        if len(txt_.replace(',','').replace(' ','').strip())>0:
            txt_ = txt_.replace(' ,', ',').replace(',', ', ')
            if len(txt_) >= max_len:
                start = 0
                while True:
                    if start>=len(txt_):
                        break
                    else:
                        sub_txt = txt_[start:start+max_len]
                        start += max_len
                        if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
                            sub_txt = sub_txt + '.'
                        txt_list.append(sub_txt.strip())
            else:
                if not (txt_.endswith('.') or txt_.endswith('?') or txt_.endswith('!')):
                    txt_ = txt_ + '.'
                txt_list.append(txt_.strip())
    return txt_list

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model


## =============================== load pretrained model =======================================
### tacotron
# =============================== define web app =======================================
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI']= DB_URL
# db = SQLAlchemy(app)

jeju_translation = Korean2Dialect('jeju', beam_search=False, k=0)           # 제주 번역 클래스 선언
gyeong_translation = Korean2Dialect('gyeong', beam_search=False, k=0)       # 경상 번역 클래스 선언
jeon_translation = Korean2Dialect('jeon', beam_search=False, k=0)           # 전라 번역 클래스 선언

jeju_speech = Text2Speech('제주')  # 제주 음성합성 클래스 선언
gyeong_speech = Text2Speech('경상')  # 경상 음성합성 클래스 선언
jeon_speech = Text2Speech('전라')                                            # 전라 음성합성 클래스 선언


@app.route('/ml-inference', methods=['POST'])
def ml_inference():
    print('====== Synthesizing ======')
    print(request.form)
    total_time = time.time()
    gender = str(request.form['gender-options'])         # [0, 1] == ['남자', '여자']
    model_type = str(request.form['model-type-options'])      # [0, 1, 2] = ['제주도', '경상도', '전라도]
    # 사투리 결정
    # ##### select model
    if model_type == '표준':
        ###표준
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type == '제주':
        ## with jeju voice, need  translation (standard --> jeju language)
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type == '경상':
        ##제주
        korean2dialect = gyeong_translation
        text2speech = gyeong_speech
    elif model_type == '전라':
        korean2dialect = jeon_translation
        text2speech = jeon_speech
    else:
        return app.response_class(response=None, status=404, mimetype='application/json')

    korean = request.form['input-text']  # 표준어 Input
    # dialect = korean2dialect.transform(korean)  # 번역
    dialect = korean  # 번역
    txt_list = clean_text(txt=dialect)  # 번역된 텍스트 클리닝
    wav_file, error_log = text2speech.forward(txt_list)  # 텍스트 -> wav file
    error_sentences = []
    for k, v in error_log.items():
        if v==True:
            error_sentences.append(k)
    error_sentences = '|'.join(error_sentences)
    return_data = {'translated_text': dialect, 'audio_stream': wav_file}
    res = app.response_class(response=json.dumps(return_data), status=200, mimetype='application/json')
    ip = request.remote_addr
    print('Total time(translation + synthesize): {}'.format(time.time() - total_time))
    tts = TTS(dialect_type=model_type, korean=korean, dialect=dialect,ip=ip, error=error_sentences)
    # db.session.add(tts)
    # db.session.commit()
    return res


def test():
    test_txts = read_ejn(file_path=r'source/filelists/donate_comment_500000.csv')
    print(request.form)
    gender = "male"
    model_type = "경상"

    # ##### select model
    if model_type=='표준':
        ###표준
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type=='제주':
        ## with jeju voice, need  translation (standard --> jeju language)
        korean2dialect = jeju_translation
        text2speech = jeju_speech
    elif model_type=='경상':
        ##제주
        korean2dialect = gyeong_translation
        text2speech = gyeong_speech
    elif model_type=='전라':
        korean2dialect = jeon_translation
        text2speech = jeon_speech
    else:
        return app.response_class(response=None,status=404,mimetype='application/json')

    test_result = []
    save_flag=0
    for row in test_txts.itertuples():
        korean = row[1]
        dialect = korean2dialect.transform(korean)  # 번역
        txt_list = clean_text(txt=dialect)  # 번역된 텍스트 클리닝
        print(txt_list)
        try:
            base_64_wav_file, audio = text2speech.forward(txt_list)  # 텍스트 -> wav file
            test_result.append([korean]+ txt_list + [len(txt_list[0]), audio.shape[0]])
        except:
            test_result.append([korean]+ txt_list +[len(txt_list[0]), None])

        save_flag +=1
        if save_flag%100==0:
            test_result_df = pd.DataFrame(test_result, columns=['ejn_text', 'trans_text','text_length', 'audio_length'])
            test_result_df.to_csv('test_result.csv')
    test_result_df = pd.DataFrame(test_result, columns=['ejn_text', 'text', 'text_length', 'audio_length'])
    test_result_df.to_csv('test_result.csv')
        # return_data = {'translated_text': dialect, 'audio_stream': wav_file}
        # res = app.response_class(response=json.dumps(return_data), status=200, mimetype='application/json')
    return True


@app.route('/')
def index():
    return render_template('index.html')

def read_ejn(file_path):
    # file_path = r'source/filelists/donate_comment_500000.csv'
    ejn_df = pd.read_csv(file_path, encoding='utf-8', header=None)
    ejn_df=ejn_df.sample(frac=1)
    ejn_df=ejn_df.iloc[:5000,:]
    text = ejn_df.iloc[:,0]
    ejn_df['length'] = text.str.len()
    ejn_df.sort_values('length', ascending=False, inplace=True)
    return ejn_df