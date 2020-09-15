class PostProcessing:
    def __init__(self):
        self.src = None
        self.tar = None

    def post_processing(self, src, tar):
        """
            output 문장의 unk 발생시 표준어를 살펴보고 unk을 대체
            :param src: input 문장(표준어)
            :param tar: output 문장(표준어)
            :return: <unk>제거한 output 문장
        """
        if tar.find('<unk>') == -1:  # unk이 없을 경우
            return tar
        elif tar == '<unk>':
            return src

        self.src = src
        self.tar = tar
        while tar.find('<unk>') != -1:
            unk_idx = tar.find('<unk>')
            if unk_idx == 0:  # unk가 맨 앞에 있을 때
                voca, start = self.first_unk(src, tar, unk_idx)
                tar = voca + tar[start:]
                if voca == "<unk>":
                    print("First Unknown Error")

            elif unk_idx == len(tar) - 5:  # unk가 맨뒤에 있을 때
                voca, end = self.last_unk(src, tar, unk_idx)
                tar = tar[:end + 1] + voca
                if voca == "<unk>":
                    print("Last Unknown Error")
            else:  # 맨앞, 맨뒤 모두 아닐 때
                voca, start, end = self.middle_unk(src, tar, unk_idx)
                tar = tar[:end+1] + voca + tar[start:]
                if voca == "<unk>":
                    print("Unknown Error")
            if voca == "<unk>":
                break
        return tar

    def first_unk(self, src, tar, unk_idx):
        start = unk_idx + 5
        token = tar[start:start + 1]
        if token.find('<') != -1:
            pre_voca, _ = self.first_unk(src, tar, start + token.find('<'))
            return pre_voca, start
        src_idx = src.find(token)
        if src_idx == -1:
            return "<unk>", 0
        else:
            voca = src[:src_idx]
            return voca, start

    def last_unk(self, src, tar, unk_idx):
        end = unk_idx - 1
        token = tar[end - 4:end + 1]
        src_idx = src.find(token)
        if src_idx == -1:
            return "<unk>", end - 1
        else:
            voca = src[src_idx + len(token):]
            return voca, end

    def middle_unk(self, src, tar, unk_idx):
        end = unk_idx - 1  # 앞 단어
        if unk_idx < 4:
            pre_token = tar[:end + 1]
        else:
            pre_token = tar[end - 4:end+1]

        if pre_token.find(">") != -1:
            bow_idx = pre_token.index(">")
            pre_token = pre_token[bow_idx + 1:]
        pre_idx = src.find(pre_token)

        start = unk_idx + 5  # 뒤 단어
        if unk_idx >= len(tar) - 5:
            post_token = tar[start:]
        else:
            post_token = tar[start:start + 4]
        if post_token.find("<") != -1:
            bow_idx = post_token.index('<')
            post_token = post_token[:bow_idx]
        post_idx = src.find(post_token)

        if pre_idx == -1 or post_idx == -1:  # 해당 token이 source 문장에 없을 경우
            voca = "<unk>"
            return voca, start, end
        voca = src[pre_idx + len(pre_token):post_idx]
        return voca, start, end