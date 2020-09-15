import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_sinusoid_encoding_table(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, 2 * (i // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])  # => [pos, d_model]
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k, padding_id):
    # seq_q, seq_k => [batch_size, seq_len] 입력문장

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(padding_id).unsqueeze(1)  # => [batch_size, 1, seq_q(=seq_k)]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # => [batch_size, len_q, len_k]


def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** 0.5

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_k, d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        # => [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn_prob = nn.Softmax(dim=-1)(scores)  # => [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)  # => [batch_size, n_heads, len_k, d_v]
        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, head_dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

        self.W_Q = nn.Linear(hid_dim, n_heads * head_dim)
        self.W_K = nn.Linear(hid_dim, n_heads * head_dim)
        self.W_V = nn.Linear(hid_dim, n_heads * head_dim)

        self.Attention = ScaledDotProductAttention(head_dim, dropout)
        self.linear = nn.Linear(n_heads * head_dim, hid_dim)

    def forward(self, q, k, v, attn_mask):
        # q => [batch_size, len_q, d_model]
        # k => [batch_size, len_k, d_model]
        # v => [batch_size, len_k, d_model]
        batch_size = q.size(0)

        # q_s => [batch_size, n_heads, len_q, d_k]
        q_s = self.W_Q(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # k_s => [batch_size, n_heads, len_k, d_k]
        k_s = self.W_K(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # v_s => [batch_size, n_heads, len_k x d_v]
        v_s = self.W_V(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # attn_mask => [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(device)

        # context => [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        context, attn = self.Attention(q_s, k_s, v_s, attn_mask)

        # context => [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)

        output = self.linear(context)  # => [batch_size, len_q, d_model]
        output = self.dropout(output)
        top_attn = attn.view(batch_size, self.n_heads, q.size(1), k.size(1))[:, 0, :, :].contiguous()
        return output, top_attn


class PoswiseFeedForwardNet(nn.Module):
    # 포지션-와이즈 피드 포워드 신경망
    # FFNN(x) = MAX(0, xW1 + b1)W2 + b2
    def __init__(self, hid_dim, pf_dim, dropout=0):
        super(PoswiseFeedForwardNet, self).__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(in_channels=hid_dim, out_channels=pf_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=pf_dim, out_channels=hid_dim, kernel_size=1)
        self.active = nn.functional.gelu

    def forward(self, inputs):
        # inputs => [batch_size, seq_len, d_model]
        output = self.dropout(self.active(self.conv1(inputs.transpose(1, 2))))  # => [batch_size, pf_dim, seq_len]
        output = self.conv2(output).transpose(1, 2)  # => [batch_size, seq_len, d_model]
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, head_dim, pf_dim, dropout, layer_norm_epsilon=1e-12):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(hid_dim, n_heads, head_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(hid_dim, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(hid_dim, pf_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hid_dim, eps=layer_norm_epsilon)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs to same Q, K, V
        # enc_inputs => [batch_size, seq_len]
        # enc_self_attn_mask => [batch_size, seq_len, seq_len ]
        attn_outputs, attn_prob = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # attn_outputs => [batch_size, seq_len, d_model]
        # attn_prob => [batch_size, n_heads, len_q, len_k]
        attn_outputs = self.layer_norm1(enc_inputs + attn_outputs)  # => [batch_size, seq_len, d_model]

        ffn_outputs = self.pos_ffn(attn_outputs)  # => [batch_size, len_q, d_model]
        ffn_outputs = self.layer_norm2(ffn_outputs + attn_outputs)  # => [batch_size, len_q, d_model]
        return ffn_outputs, attn_prob


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, head_dim, pf_dim, dropout, layer_norm_epsilon=1e-12):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(hid_dim, n_heads, head_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(hid_dim, eps=layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(hid_dim, n_heads, head_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hid_dim, eps=layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(hid_dim, pf_dim, dropout)
        self.layer_norm3 = nn.LayerNorm(hid_dim, eps=layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        self_att_outputs, self_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        dec_att_outputs, dec_enc_attn = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_att_outputs = self.layer_norm2(self_att_outputs + dec_att_outputs)

        ffn_outputs = self.pos_ffn(dec_att_outputs)
        ffn_outputs = self.layer_norm3(dec_att_outputs + ffn_outputs)
        return ffn_outputs, self_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, head_dim, pf_dim, dropout=0, max_length=50, padding_id=3):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(input_dim, hid_dim)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(max_length + 1, hid_dim))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, head_dim, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.padding_id = padding_id

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        # enc_inputs => [batch_size, sequence_len]
        # positions => [batch_size, sequence_len]
        positions = torch.arange(enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype) \
                        .expand(enc_inputs.size(0), enc_inputs.size(1)).contiguous() + 1

        pos_mask = enc_inputs.eq(self.padding_id)  # padding을 masking
        positions.masked_fill_(pos_mask, 0).to(device)  # True는 0으로 masking
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(positions)  # Embedding + pos_enbeding
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.padding_id)
        # => [batch_size, seq_len, seq_len ]

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # enc_outputs => [batch_size, len_q, d_model]
            # enc_self_attn => [batch_size, n_heads, len_q, len_k]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, head_dim, pf_dim, dropout, max_length=50, padding_id=3):
        super(Decoder, self).__init__()
        self.tar_emb = nn.Embedding(input_dim, hid_dim)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(max_length + 1, hid_dim))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, head_dim, pf_dim, dropout)
                                     for _ in range(n_layers)])
        self.padding_id = padding_id
        self.classifier = nn.Linear(hid_dim, input_dim)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):  # dec_inputs : [batch_size x target_len]
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype) \
                        .expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.padding_id)
        positions.masked_fill_(pos_mask, 0)
        dec_outputs = self.tar_emb(dec_inputs) + self.pos_emb(positions)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.padding_id)

        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.padding_id)

        # self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                   dec_enc_attn_mask)
            # self_attn_probs.append(self_attn_prob)
            # dec_enc_attn_probs.append(dec_enc_attn_prob)
        # (bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]
        dec_outputs = self.classifier(dec_outputs)
        dec_outputs = nn.functional.log_softmax(dec_outputs, dim=-1)
        return dec_outputs, dec_enc_attn_prob  # self_attn_probs, dec_enc_attn_probs


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs, dec_inputs => [batch_size, seq_len]
        enc_outputs, _ = self.encoder(enc_inputs)
        dec_outputs, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        return dec_outputs, dec_enc_attns


def greedy_decoder(model, enc_input, seq_len=50, start_symbol=0):
    """
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 0
    :return: The target input
    """
    batch_size = enc_input.size(0)
    enc_outputs, enc_self_attns = model.module.encoder(enc_input)
    dec_input = torch.LongTensor(batch_size, seq_len).fill_(4).to(device)
    next_symbol = start_symbol
    for i in range(0, seq_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _ = model.module.decoder(dec_input, enc_input, enc_outputs)
        prob = dec_outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
        if next_word.item() == 1:
            break
    return dec_input


class Greedy:
    def __init__(self, model, seq_len=50, start_symbol=0):
        self.encoder = model.module.encoder
        # 미리 한번 불러서 선언해두면 이후 속도가 매우 빨라짐
        lst = [4 for _ in range(seq_len)]
        _ = self.encoder(torch.tensor([lst]).to(device))
        self.decoder = model.module.decoder
        self.seq_len = seq_len
        self.start_symbol = start_symbol

    def greedy_decoder(self, enc_input):
        batch_size = enc_input.size(0)
        enc_outputs, enc_self_attns = self.encoder(enc_input)
        dec_input = torch.LongTensor(batch_size, self.seq_len).fill_(4).to(device)
        next_symbol = self.start_symbol
        for i in range(0, self.seq_len):
            dec_input[0][i] = next_symbol
            dec_outputs, _ = self.decoder(dec_input, enc_input, enc_outputs)
            prob = dec_outputs.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[i]
            next_symbol = next_word.item()
            if next_symbol == 1:
                break
        return dec_input


class Beam:
    def __init__(self, model, beam_size, start_token_id=0, end_token_id=1, padding_token_id=4, seq_len=50):
        self.k = beam_size
        self.start_token = start_token_id
        self.end_token = end_token_id
        self.padding_token = padding_token_id
        self.seq_len = seq_len
        self.prev_ks = []  # 후보 idx
        self.prev_ks_score = []  # 후보 score
        self.finished = []  # 후보 idx
        self.finished_score = []  # 후보 score
        self.model = model
        self.encoder = model.module.encoder
        # 미리 한번 불러서 선언해두면 이후 속도가 매우 빨라짐
        lst = [4 for _ in range(seq_len)]
        _ = self.encoder(torch.tensor([lst]).to(device))

    def beam_initialize(self):
        self.prev_ks = []
        self.prev_ks_score = []  # 후보 score
        self.finished = []  # 후보 idx
        self.finished_score = []  # 후보 score

    def beam_search_decoder(self, enc_input):
        # enc_input = [batch_size(=1), seq_len)
        batch_size = enc_input.size(0)
        enc_outputs, _ = self.encoder(enc_input)
        for i in range(self.k):  # k개 후보 빈문장 생성
            dec_input = torch.LongTensor(batch_size, self.seq_len).fill_(self.padding_token).to(device)
            dec_input[0][0] = self.start_token  # start token 추가
            self.prev_ks.append(dec_input)  # 후보 추가
            self.prev_ks_score.append(1)  # 기본 score 1 추가

        for i in range(0, self.seq_len - 1):  # seq_len -1 만큼  반복
            self.advance(enc_input, enc_outputs, i)  # 전개 실시
            if len(self.finished) == self.k:  # 최종후보가 k개 모이면 break
                break

        if len(self.finished) != self.k:  # advance 종료시에도 최종후보가 충분치 못하면
            for idx in range(len(self.prev_ks)):  # 후보의 개수만큼 반복해서 순서대로 최종후보 채우기
                self.finished.append(self.prev_ks[idx][0])
                self.finished_score.append(self.prev_ks_score[idx])
                if len(self.finished) == self.k:  # 최종후보가 k개 되면 종료
                    break

        max_idx = torch.FloatTensor(self.finished_score).topk(1)[1]  # 최종 후보중 가장좋은 값 선택
        return self.finished[max_idx]

    def advance(self, enc_input, enc_outputs, i):
        all_scores = []
        all_scores_id = []
        attentions = []
        if i == 0:  # 첫번째 전개
            dec_outputs, attention = self.model.module.decoder(self.prev_ks[0], enc_input, enc_outputs)
            top_score, top_score_id = dec_outputs.squeeze(0).topk(self.k + 1, dim=-1)
            all_scores += top_score.data[i]
            all_scores_id += top_score_id.data[i]
            for _ in range(self.k + 1):
                attentions.append(attention)
            top_scores, temp_ids = torch.tensor(all_scores).topk(self.k + 1, sorted=True)
        else:  # 두번째 이후 전개
            for prev in self.prev_ks:
                dec_outputs, attention = self.model.module.decoder(prev, enc_input, enc_outputs)
                top_score, top_score_id = dec_outputs.squeeze(0).topk(self.k, dim=-1)
                all_scores += top_score.data[i]  # K^2개의 자식노드의 score
                all_scores_id += top_score_id.data[i]  # K^2개의 자식노드의 id
                for _ in range(self.k):
                    attentions.append(attention)
            top_scores, temp_ids = torch.tensor(all_scores).topk(self.k * 2, sorted=True)  # 2k개의 후보노드 저장
        top_score_ids = [all_scores_id[j].item() for j in temp_ids]  # 2k개의 실제 index 저장
        top_attentions = [attentions[j] for j in temp_ids]
        prev_status_idx, prev_status_score = self.prev_top(temp_ids)  # 이전 경로를 2k개 순서대로 저장

        count = 0
        j = 0

        while count < self.k:  # k개의 후보경로가  생성되면 종료
            if i == 1 and top_score_ids[j] == 1:  # 첫번째 branch때 end token이 뜨는 경우
                j += 1
                continue

            if top_score_ids[j] == 1:  # end token이 나왔는지 확인
                # 나왔다면 최종 후보지로 등록
                self.finished.append(prev_status_idx[j])
                # 최종 후보지 score등록
                length_norm = self._get_length_penalty(i)  # length normalization
                coverage_norm = self._get_coverage_penalty(top_attentions[j], i)  # coverage normalization
                prev_status_score[j] /= length_norm + coverage_norm
                self.finished_score.append(prev_status_score[j])
                j += 1
                if len(self.finished) == self.k:
                    break
            else:
                prev_status_idx[j][i + 1] = top_score_ids[j]  # 후보노드에 해당 노드를 추가해서 저장
                prev_status_score[j] += top_scores[j].item()  # 누적확률 저장

                self.prev_ks[count][0] = prev_status_idx[j]  # 다음 후보경로 선정
                self.prev_ks_score[count] = prev_status_score[j]  # 후보경로의 누적확률 저장
                j += 1
                count += 1

    # top index가 나오면 그 개수의 2배만큼 이전 road를 순서대로 생성
    def prev_top(self, temp_idx):
        result = []
        result_score = []
        for idx in temp_idx:
            creterion = self.k
            for j in range(self.k * 2):
                if creterion - self.k <= idx <= creterion - 1:  # temp_idx 에 나온 idx에 따라 해당 이전 road를 추가
                    result.append(self.prev_ks[j][0].clone())
                    result_score.append(self.prev_ks_score[j])
                    break
                creterion += self.k
        return result, result_score

    # beam의 길이에 따른 penalty
    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """ 확률은 0~1 사이이므로 길이가 길어질 수록 더 적아진다. 이를 보완하기 위해 길이에 따른 패널티를 부여하고 계산하며,
        일반적으로 alpha = 1.2, min_length = 5를 사용하며, 이는 수정가능하다."""
        return ((min_length + length) / (min_length + 1)) ** alpha

    def _get_coverage_penalty(self, attention, x_lenth, beta=0.2, cp=0):
        attention = attention.squeeze(0)
        for i in range(x_lenth):
            sum_ = 0
            for j in range(x_lenth + 1):
                sum_ += attention[i, j].item()
            min_ = min(sum_, 1.0)
            log_ = np.log(min_)
            cp = cp + log_
        return cp * beta
