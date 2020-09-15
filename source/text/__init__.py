#-*- coding: utf-8 -*-
import numpy as np
import re
from source.text import cleaners
from jamo import h2j
from itertools import chain

import re
from source.text import cleaners
from source.text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
hangul_symbol =     u'''␀␃%"ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆞᆢᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆪᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀ'''
hangul_symbol_hcj = u'''␀␃%"ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎᅌㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣᆞᆢㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌ'''

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def number_to_hangul(text):
    import re
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    for number in numbers:
        number_text = digit2txt(number)
        text = text.replace(number, number_text, 1)
    return text


def digit2txt(strNum):
    # 만 단위 자릿수
    tenThousandPos = 4
    # 억 단위 자릿수
    hundredMillionPos = 9
    txtDigit = ['', '십', '백', '천', '만', '억']
    txtNumber = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    txtPoint = '쩜 '
    resultStr = ''
    digitCount = 0
    #자릿수 카운트
    for ch in strNum:
        # ',' 무시
        if ch == ',':
            continue
        #소숫점 까지
        elif ch == '.':
            break
        digitCount = digitCount + 1
    digitCount = digitCount-1
    index = 0
    while True:
        notShowDigit = False
        ch = strNum[index]
        #print(str(index) + ' ' + ch + ' ' +str(digitCount))
        # ',' 무시
        if ch == ',':
            index = index + 1
            if index >= len(strNum):
                break
            continue
        if ch == '.':
            resultStr = resultStr + txtPoint
        else:
            #자릿수가 2자리이고 1이면 '일'은 표시 안함.
            # 단 '만' '억'에서는 표시 함
            if(digitCount > 1) and (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos) and int(ch) == 1:
                resultStr = resultStr + ''
            elif int(ch) == 0:
                resultStr = resultStr + ''
                # 단 '만' '억'에서는 표시 함
                if (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos):
                    notShowDigit = True
            else:
                resultStr = resultStr + txtNumber[int(ch)]
        # 1억 이상
        if digitCount > hundredMillionPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-hundredMillionPos]
        # 1만 이상
        elif digitCount > tenThousandPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-tenThousandPos]
        else:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount]
        if digitCount <= 0:
            digitCount = 0
        else:
            digitCount = digitCount - 1
        index = index + 1
        if index >= len(strNum):
            break
    return resultStr

#
# #
# def get_hangul_to_ids():
#     hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol)}
#     ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol)}
#     return hangul_to_ids, ids_to_hangul
#

def clean_text(txt):
    # txt = '''문재인^ ""^^대통령은" 오늘(16일)" 과학기술정보통신부와 방송통신위원회의 업무보고를 받는 것을 시작으로 새해 부처별 업무보고 일정을 시작합니다.
    # 			집권 4년 ^차를 맞아 부처별 국정성과를 독려하고 이를 통해 '확실한 변화'를 끌어낼 발판을 마련하겠다는 것이 이번 업무보고의 목표라고 청와대는 설명했습니다.
    # 			과기부와 방통위는 '과학기술?AI(인공지능)'를 주제로 업무 보고를 진행할 예정입니다. 4차 산업혁명 시대를 맞아 첨단기술 분야 경쟁력을 키우기 위한 방안이 집중논의될 전망입니다.
    # 			문 대통령은 오늘 업무 보고를 시작으로 '확실한 변화, 대한민국 2020'이라는 슬로건 아래 내달까지 모든 부처의 업무보고를 주재할 계획입니다. 과기부와 방통위 이후에는 강한 국방, 체감 복지, 공정 정의, 일자리, 문화 관광, 혁신 성장, 안전 안심, 외교 통일을 주제로 보고가 진행됩니다.
    # 			오늘 보고에는 정세균 국무총리도 배석합니다.
    # 			한정우 청와대 부대변인은 "2020년 확실한 변화를 만들기 위해 국민이 체감하는 성과를 다짐하는 자리가 될 것"이라고 설명했습니다.'''
    # txt='저는 송당육우다우다.'
    ### transform english char to korean text
    transform_dict = {'#':'샵', '@':'고팽이', '-':'빼기', ':':'나누기', '*':'별', '~':',', '·':' ', '’':'\"', '$':'달러', '%':'퍼센트', '&':'앤', '+':'플러스', '‘':'"', '\'':'"', '`':'"', '“':'"', '”':'"'}
    ### remove not allowed chars
    not_allowed_characters = list('^[]<>')
    txt = ''.join(i for i in txt if not i in not_allowed_characters)
    txt = txt.lower()

    ### transform special char to hangul
    for k,v in transform_dict.items():
        txt=txt.replace(k, v).replace(' .', '.').replace(' ?', '?')
    return txt


def hangul_to_sequence(hangul_text):
    hangul_text = clean_text(hangul_text)
    # load conversion dictionaries
    ### clean number
    hangul_text_ = number_to_hangul(hangul_text)
    ### add end of sentence symbol
    hangul_text_ = hangul_text_ + u"␃"  # ␃: EOS
    ### get dictionary of chars
    hangul_to_ids= _symbol_to_id
    ### process jamos
    text = [h2j(char) for char in hangul_text_]
    text = chain.from_iterable(text)
    hangul_text_ = [h2j(char) for char in text]
    hangul_text_ = chain.from_iterable(hangul_text_)
    sequence = []
    try:
        ### convert jamos to ids using dictionary
        for char in hangul_text_:
            if char in symbols:
                sequence.append(hangul_to_ids[char])
            else:
                sequence.append(hangul_to_ids[symbols[hangul_symbol_hcj.index(char)]])
    except KeyError as e:
        raise KeyError('KeyError (at key: {}) when processing: {}'.format(e,hangul_text))
    return sequence

def text_to_sequence_(text, cleaner_names):
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'


def group_words(s):
    return re.findall(u'[a-z]+', s)



def text_to_sequence(txt, cleaner):
    txt=txt.lower().strip() + u"␃"
    txt_list = re.split(r'([a-z]+)', txt)
    sequence=[]
    for txt_ in txt_list:
        try:
            sequence+=hangul_to_sequence(txt_)[:-1]
        except:
            sequence+=text_to_sequence_(txt_,cleaner)
    return sequence

#
# for line in open(r'F:\Nvidia-Tacotron\tacotron2_en_gyeongsang\filelists\ljs_audio_text_train_filelist.txt'):
#     txt = line.split('|')[-1]
#     print(sequence_to_text(text_to_sequence(txt, ['english_cleaners'])))
# txt='완성작의 저작권은 슈가캣 & 캔디도기에 있어예.'
# sequence_to_text(text_to_sequence(txt, ['english_cleaners']))
# print(_symbol_to_id)
#Unfortunately he took to drink
# txt = 'Converts a sequence of IDs back to a string.'


# sequence_to_text(text_to_sequence(txt, ['english_cleaners']))
