from source.translation.tools import Translation


class Korean2Dialect(object):
    def __init__(self, region, beam_search=False, k=0):
        self.translation = Translation(checkpoint='source/translation/Model/' + str(region) + '/best_transformer.pth',
                                       dictionary_path='source/translation/Dictionary/' + str(region),
                                       beam_search=beam_search, k=k, region=region)
        self.model = self.translation.model_load()

    def transform(self, sentence):
        token = sentence.split(' ')
        tmp = token[0]
        lst_gy = []
        if len(token) ==1:
            lst_gy.append(self.translation.korean2dialect(self.model, tmp))
        else:
            for j in token[1:]:
                tmp = ' '.join([tmp, j])
                if len(tmp) >= 35:
                    lst_gy.append(self.translation.korean2dialect(self.model, tmp))
                    tmp = ''
                elif j == token[-1]:
                    lst_gy.append(self.translation.korean2dialect(self.model, tmp))
        output_sentence = ' '.join(lst_gy)
        return output_sentence


if __name__ == '__main__':
    gyeong_translation = Korean2Dialect('gyeong', beam_search=False, k=0)
    sen = gyeong_translation.transform("록스의 ‘표준어 사투리 번역 TTS’는 딥러닝 기반의 TTS(Text to Speech) 서비스 입니다.")
    print(sen)