# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: /content/gdrive/My Drive/colab_/trade/relative_selfattention.py
# Compiled at: 2020-05-14 20:51:09
# Size of source mod 2**32: 4615 bytes
import torch, torch.nn as nn, torch.utils as utils, torch.nn.functional as F
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
import numbers
print('rel long ver')
torch.nn.Module.dump_patches = True

class MultiHeadedAttention_RPR(nn.Module):

    def __init__(self, d_model, h, max_relative_position, dropout=0.1):
        """
        multi-head attention
        :param h: nhead
        :param d_model: d_model
        :param dropout: float
        """
        super(MultiHeadedAttention_RPR, self).__init__()
        assert d_model % h == 0
        dropout = 0.1
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = 1000
        self.vocab_size = max_relative_position
        self.embed_K = nn.Embedding(self.vocab_size, self.d_k).to(device=device)
        self.embed_V = nn.Embedding(self.vocab_size, self.d_k).to(device=device)

    def forward(self, query, key, value, mask=None):
        """
        ---------------------------
        L : target sequence length
        S : source sequence length:
        N : batch size
        E : embedding dim
        ---------------------------
        :param query: (N,L,E)
        :param key: (N,S,E)
        :param value: (N,S,E)
        :param mask:
        """
        nbatches = query.size(0)
        seq_len = query.size(1)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        relation_keys = self.generate_relative_positions_embeddings(seq_len, seq_len, self.embed_K)
        relation_values = self.generate_relative_positions_embeddings(seq_len, seq_len, self.embed_V)
        logits = self._relative_attn_inner(query, key, relation_keys, True)
        weights = self.dropout(F.softmax(logits, -1))
        x = self._relative_attn_inner(weights, value, relation_values, False)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return (self.linears[(-1)](x), weights)

    def _generate_relative_positions_matrix(self, len_q, len_k):
        """
        genetate rpr matrix
        ---------------------------
        :param len_q: seq_len
        :param len_k: seq_len
        :return: rpr matrix, dim: (len_q, len_q)
        """
        assert len_q == len_k
        range_vec_q = range_vec_k = torch.arange(len_q).to(device=device)
        distance_mat = range_vec_k.unsqueeze(0) - range_vec_q.unsqueeze(-1)
        disntance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position).to(device=device)
        return disntance_mat_clipped + self.max_relative_position

    def generate_relative_positions_embeddings(self, len_q, len_k, embedding_table):
        """
        generate relative position embedding
        ----------------------
        :param len_q:
        :param len_k:
        :return: rpr embedding, dim: (len_q, len_q, d_k)
        """
        relative_position_matrix = self._generate_relative_positions_matrix(len_q, len_k)
        return embedding_table(relative_position_matrix)

    def _relative_attn_inner(self, x, y, z, transpose):
        """
        efficient implementation
        ------------------------
        :param x:
        :param y:
        :param z:
        :param transpose:
        :return:
        """
        # x : orch.Size([6, 8, 342, 50])
        nbatches = x.size(0)
        heads = x.size(1)
        seq_len = x.size(2)
        xy_matmul = torch.matmul(x, y.transpose(-1, -2) if transpose else y)
        x_t_v = x.permute(2, 0, 1, 3).contiguous().view(seq_len, nbatches * heads, -1)
        x_tz_matmul = torch.matmul(x_t_v, z.transpose(-1, -2) if transpose else z)
        x_tz_matmul_v_t = x_tz_matmul.view(seq_len, nbatches, heads, -1).permute(1, 2, 0, 3)
        return xy_matmul + x_tz_matmul_v_t
# okay decompiling relative_selfattention_original.pyc


class PositionWiseFeedForward(nn.Module):

    def __init__(self, embed_dim: int, d_ff: int = 1600, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x + residual



class NormLayer(nn.Module):
    __constants__ = ['norm_shape', 'weight', 'bias', 'eps']

    def __init__(self, norm_shape, eps=1e-6):
        super(NormLayer, self).__init__()
        if isinstance(norm_shape, numbers.Integral):
            norm_shape = (norm_shape,)
        self.norm_shape = norm_shape

        # create two trainable parameters to do affine tunening
        self.weight = nn.Parameter(torch.ones(*self.norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(*self.norm_shape), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.weight * (x - x.mean(dim=-1, keepdim=True))
        norm /= (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm