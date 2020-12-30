# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: /content/gdrive/My Drive/colab_/trade/enc_dec_noDME_relative_2_2_FF_GRU_gate.py
# Compiled at: 2020-05-14 20:51:09
# Size of source mod 2**32: 17940 bytes
import torch, torch.nn as nn, torch.nn.functional as F, json
from masked_cross_entropy import *
from config import *
from relative_selfattention import *
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import random

# torch.nn.Module.dump_patches = True
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("20 heads")


def create_mask(src: torch.Tensor,
                trg: torch.Tensor,
                src_pad_idx: int,
                trg_pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src_mask = _create_padding_mask(src, src_pad_idx)
    trg_mask = None
    if trg is not None:
        trg_mask = _create_padding_mask(trg, trg_pad_idx)  # (256, 1, 33)
        nopeak_mask = _create_nopeak_mask(trg)  # (1, 33, 33)
        trg_mask = trg_mask & nopeak_mask  # (256, 33, 33)
    return src_mask, trg_mask

def _create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    seq 형태를  (256, 33) -> (256, 1, 31) 이렇게 변경합니다.

    아래와 같이 padding index부분을 False로 변경합니다. (리턴 tensor)
    아래의 vector 하나당 sentence라고 보면 되고, True로 되어 있는건 단어가 있다는 뜻.
    tensor([[[ True,  True,  True,  True, False, False, False]],
            [[ True,  True, False, False, False, False, False]],
            [[ True,  True,  True,  True,  True,  True, False]]])
    """
    return (seq != pad_idx).unsqueeze(-2)

def _create_nopeak_mask(trg) -> torch.Tensor:
    """
    NO PEAK MASK
    Target의 경우 그 다음 단어를 못보게 가린다
    """
    batch_size, seq_len = trg.size()
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=trg.device), diagonal=1)).bool()
    return nopeak_mask

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFF(nn.Module):

    def __init__(self, d_input, d_inner, dropout):
        super().__init__()
        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = nn.Sequential(nn.Linear(d_input, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_inner, d_input), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, input_: torch.FloatTensor) -> torch.FloatTensor:
        ff_out = self.ff(input_)
        output = self.layer_norm(input_ + ff_out)
        return output


class EncoderRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, dropout, lang, n_layers=int(args['layer'])):
        print('20 head')
        super(EncoderRNN, self).__init__()
        hidden_size = 400
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.d_model = 400
        self.num_heads = 8
        # self.positional_enc = PositionalEncoding(self.d_model, self.dropout)
        self.multihead_attn = MultiHeadedAttention_RPR(self.d_model, self.num_heads, self.vocab_size, dropout=dropout)
        self.selfnorm1 = NormLayer(self.hidden_size)
        self.ff = PositionWiseFeedForward(embed_dim=self.hidden_size)
        self.selfnorm2 = NormLayer(self.hidden_size)

        self.norm1 = nn.LayerNorm(self.hidden_size*2)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.update_layer = nn.Linear((2 * hidden_size), hidden_size, bias=False)
        self.gate = nn.Linear((2 * hidden_size), hidden_size, bias=False)


        if args['load_embedding']:
            print('glove loaded. ')
            with open(os.path.join('data/', 'emb{}.json'.format(vocab_size))) as (f):
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            del new
            torch.cuda.empty_cache()
        if args['dataset'] == 'kvr':
            initial_arr = self.embedding.weight.data.cpu().numpy()
            embedding_arr = torch.from_numpy(get_glove_matrix(lang, initial_arr))
            self.embedding.weight.data.copy_(embedding_arr)
            self.embedding.weight.requires_grad = True
        if args['fix_embedding']:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):

        embedded = self.embedding(input_seqs) # # torch.Size([329, batch])

        embedded__ = embedded
        embedded_ = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded_, input_lengths, batch_first=False)

        outputs, hidden_ = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden_ = hidden_[0] + hidden_[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # orch.Size([98, 6, 400])

        embedded__ = embedded__.permute(1,0,2) # (batch, len, emb)
        attn_output, attn_weights = self.multihead_attn(embedded__, embedded__, embedded__)  # 400 dim
        attn_output = self.selfnorm1(attn_output)

        #첫 sub-layer는 multi-head self-attention mechanism이고 두번쨰는 간단한 point-wise fc-layer이다. 그리고 모델 전체적으로 각 sub-layer에 residual connection을 사용했다. 그리고 residual 값을 더한 뒤에 layer 값을 Nomalize한다.
        # 즉 각 sub-layer는 결과에 대해 residual 값을 더하고 그 값을 nomalize한 값이 output으로 나오게 된다.
        ff_out = self.ff(attn_output)
        ff_out = self.selfnorm2(ff_out)

        inputs = torch.cat([outputs, ff_out.permute(1,0,2)], dim=2)
        inputs = self.norm1(inputs)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * outputs  # orch.Size([420, 6, 400])
        updated_output = self.norm2(updated_output)

        temp = attn_weights.clone()
        new_attn_weight = []
        for batch in temp:
            i = 0
            for for_head_weight in batch:
                if i == 0:
                    weight = for_head_weight
                else:
                    weight += for_head_weight
                i += 1

            new_temp = weight.cpu().detach().numpy() / i
            new_attn_weight.append(new_temp)

        new_attn_weight = np.array(new_attn_weight)
        return (updated_output.transpose(0, 1), hidden_.unsqueeze(0), new_attn_weight)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Generator(nn.Module):

    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        hidden_size = 400
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # self.norm(self.hidden_size)


        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split('-')[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[0]] = len(self.slot_w2i)
            if slot.split('-')[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[1]] = len(self.slot_w2i)

        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)
        self.count = 0

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slot_temp):
        """
        encoded_hidden = ([1, 16, 400])
        'generate_y': ['none', 'guest house', 'yes', '5', 'wednesday', '1', 'north', '0', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']}
        """
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA:
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            if slot.split('-')[0] in self.slot_w2i.keys():
                domain_w2idx = [
                 self.slot_w2i[slot.split('-')[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA:
                    domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            if slot.split('-')[1] in self.slot_w2i.keys():
                slot_w2idx = [
                 self.slot_w2i[slot.split('-')[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA:
                    slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size)
        del slot_emb_arr
        torch.cuda.empty_cache()
        encoded_hidden = encoded_hidden.squeeze(0)
        hidden = encoded_hidden.repeat(1, len(slot_temp), 1)  # torch.Size([1, 480, 400])
        del encoded_hidden
        torch.cuda.empty_cache()
        words_point_out = [[] for i in range(len(slot_temp))]

        # print(target_batches.shape, max_res_len) # orch.Size([16, 30, 4]) 4

        for wi in range(max_res_len):
            dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
            enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
            # print(hidden.shape, enc_out.shape) # orch.Size([1, 480, 400]) torch.Size([480, 309, 400])
            enc_len = encoded_lens * len(slot_temp)
            context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)  # orch.Size([480, 400]) torch.Size([480, 371]) torch.Size([480, 371])

            if self.training:
                if 1000 <= self.count <= 1002:
                    self.visualize_attend(prob, slot_temp, story)

                    if self.count == 1002:
                        self.count = 0
                self.count += 1


            del enc_out
            torch.cuda.empty_cache()
            if wi == 0:
                all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())
            p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
            p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
            del context_vec
            del dec_state
            torch.cuda.empty_cache()
            vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
            del p_gen_vec
            torch.cuda.empty_cache()
            p_context_ptr = torch.zeros(p_vocab.size())
            if USE_CUDA:
                p_context_ptr = p_context_ptr.cuda()
            p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

            # torch.Size([480, 18311])
            # 복사할 확률 + gen할 확률
            final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                            vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab

            pred_word = torch.argmax(final_p_vocab, dim=1) # orch.Size([480])

            words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]
            del p_context_ptr
            torch.cuda.empty_cache()
            for si in range(len(slot_temp)):
                words_point_out[si].append(words[si * batch_size:(si + 1) * batch_size])


            all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, (len(slot_temp), batch_size, self.vocab_size))

            if use_teacher_forcing:
                # print("target_batches : ", target_batches.shape)  # torch.Size([16, 30, 7])

                decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1, 0))) # [480, 400]
            else:
                decoder_input = self.embedding(pred_word)
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

        return (all_point_outputs, all_gate_outputs, words_point_out, [])

    def attend(self, seq, cond, lens): # torch.Size([480, 355, 400]) torch.Size([480, 400])

        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        del cond
        torch.cuda.empty_cache()
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf

        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return (
         context, scores_, scores)

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores

    def visualize_attend(self, prob, slot_temp, story):

        directory = './save/' + str(args['addName'] + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        story = story[0]
        prob = prob.reshape((args['batch'], int(prob.size(0)/args['batch']), prob.size(1)))
        prob = prob[0] # (30, seq_len)

        with torch.no_grad():
            story = story.cpu().detach().numpy()
            prob = prob.cpu().detach().numpy()

            input_words = [self.lang.index2word[w_idx.item()] for w_idx in story]
            sns.set()
            plt.clf()
            input_words = input_words[:50]
            plt.figure(figsize=(len(input_words), len(slot_temp)))
            # prob = prob.cpu().numpy()
            # ax = sns.heatmap(prob[13:20, :len(input_words)],
            #                  xticklabels=input_words,
            #                  yticklabels=slot_temp[13:20])
            ax = sns.heatmap(prob[:, :len(input_words)],
                             xticklabels=input_words,
                             yticklabels=slot_temp)

            ax.invert_yaxis()
            plt.savefig(directory+'attention' + str(random.random()) + '.png')
            plt.cla()
            plt.close()


class LayerNormalization(nn.Module):
    """ Layer normalization module """
    __module__ = __name__
    __qualname__ = 'LayerNormalization'

    def __init__(self, d_hid, eps=0.001):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter((torch.ones(d_hid)), requires_grad=True)
        self.b_2 = nn.Parameter((torch.zeros(d_hid)), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        else:
            mu = torch.mean(z, keepdim=True, dim=(-1))
            sigma = torch.std(z, keepdim=True, dim=(-1))
            ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
            return ln_out


def get_glove_matrix(vocab, initial_embedding_np):
    if os.path.exists('./data/vocab/emb-kvret.npy'):
        vec_array = np.load('./data/vocab/emb-kvret.npy')
        old_avg = np.average(vec_array)
        old_std = np.std(vec_array)
        logging.info('embedding.  mean: %f  std %f' % (old_avg, old_std))
        return vec_array
    else:
        ef = open('./data/glove.6B.200d.txt', 'r', encoding='utf-8')
        cnt = 0
        vec_array = initial_embedding_np
        old_avg = np.average(vec_array)
        old_std = np.std(vec_array)
        vec_array = vec_array.astype(np.float32)
        new_avg, new_std = (0, 0)
        for line in ef.readlines():
            line = line.strip().split(' ')
            word, vec = line[0], line[1:]
            vec = np.array(vec, np.float32)
            word_idx = vocab.index_word(word)
            if word.lower() in ('unk', '<unk>') or word_idx != UNK_token:
                cnt += 1
                vec_array[word_idx] = vec
                new_avg += np.average(vec)
                new_std += np.std(vec)

        new_avg /= cnt
        new_std /= cnt
        ef.close()
        logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (
         cnt, old_avg, new_avg, old_std, new_std))
        np.save('./data/vocab/emb-kvret.npy', vec_array)
        return vec_array
# okay decompiling enc_dec_original.pyc
