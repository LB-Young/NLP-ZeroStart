"""


"""


import torch
import torch.nn as nn
import torch.nn.functional as f


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tag_vocab_size,
                 position_num,
                 src_padding_index,
                 tag_padding_index,
                 embedding_size,
                 block_num,
                 head_num,
                 model_size,
                 forward_hidden_size,
                 kq_d,
                 v_d,
                 dropout,
                 enc_dec_embedding_share=True,
                 dec_out_embedding_share=True
                 ):
        """

        接受参数；
        scale_emb和scale_prj记录是否使用了两种权重共享，用于传递参数和forward中调用
        定义encoder和decoder、以及最后的分类层
        模型参数均匀分布处理
        根据权重共享的参数，对需要共享参数的层进行参数共享

        :param src_vocab_size:  输入语言的词汇表大小
        :param tag_vocab_size:  输出语言的词汇表大小
        :param position_num:    位置编码个数
        :param src_padding_index:  输入语言的padding索引
        :param tag_padding_index: 输出语言的padding索引
        :param embedding_size: 词向量大小
        :param block_num: block数目（原文encoder和decoder的block都设置为6）
        :param head_num: 头数（原文8）
        :param model_size: 模型的size
        :param forward_hidden_size: 前馈部分有两个线性层，第一个线性层输出维度为forward_hidden_size，第二个输出维度为model_size
        :param kq_d: k、q维度是相同的
        :param v_d:
        :param dropout:
        :param enc_dec_embedding_share: encoder和decoder的embedding词表是否共享权重（输入和输出为一种语言则可以共享，否则不能共享）
        :param dec_out_embedding_share: decoder和最后输出的linear层共享参数，一般decoder的输入语言就是decoder上一时刻的输出语言，所以一般设置为True
        """
        super(Transformer, self).__init__()

        self.src_padding_index, self.tag_padding_index = src_padding_index, tag_padding_index
        self.model_size = model_size

        self.encoder = Encoder(src_vocab_size=src_vocab_size,
                               position_num=position_num,
                               src_padding_index=src_padding_index,
                               embedding_size=embedding_size,
                               block_num=block_num,
                               head_num=head_num,
                               model_size=model_size,
                               forward_hidden_size=forward_hidden_size,
                               kq_d=kq_d,
                               v_d=v_d,
                               dropout=dropout
                               )

        self.decoder = Decoder(tag_vocab_size=tag_vocab_size,
                               position_num=position_num,
                               tag_padding_index=tag_padding_index,
                               block_num=block_num,
                               head_num=head_num,
                               embedding_size=embedding_size,
                               model_size=model_size,
                               forward_hidden_size=forward_hidden_size,
                               kq_d=kq_d,
                               v_d=v_d,
                               dropout=dropout
                               )
        self.classifier = nn.Linear(model_size, tag_vocab_size, bias=False)

        # 参数均匀分布处理
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert model_size == embedding_size

        if enc_dec_embedding_share:
            self.encoder.src_embedding.weight = self.decoder.tag_embedding.weight

        if dec_out_embedding_share:
            self.classifier.weight = self.decoder.tag_embedding.weight

    def forward(self, src_seq, src_position_seq, tag_seq, tag_position_seq):
        """

        :param src_seq:  [batch, input_seq_len] [32 * 120]
        :param tag_seq:  [batch, output_seq_len] [32 * 30]
        :return:
        """

        src_mask = get_pad_mask(src_seq, self.src_padding_index)
        # src_mask: [batch, input_seq_len] => [batch, 1, input_seq_len]
        tag_mask = get_pad_mask(tag_seq, self.tag_padding_index) & get_subsequence_mask(tag_seq)
        # src_mask: [batch, output_seq_len] => [batch, 1, out_seq_len] => [batch, out_seq_len, out_seq_len]

        # print("==============", src_seq.shape, src_position_seq.shape, src_mask.shape, "==========")
        enc_out = self.encoder(src_seq, src_position_seq, src_mask)
        # print("111111111", enc_out.shape)
        dec_out = self.decoder(tag_seq, tag_position_seq, tag_mask, enc_out, src_mask)
        seq_predict = self.classifier(dec_out)

        # seq_predict: [batch, seq_len, vocab_size] => [batch*seq_len, vocab_size]
        return seq_predict.view(-1, seq_predict.size(2))


def get_pad_mask(seq, padding_index):
    # return seq != padding_index
    return (seq != padding_index).unsqueeze(-2)
    # 在将mask作用到attn上面的时候，attn是torch.Size([32, 2, 120, 120])，需要把mask，reshape成torch.Size([32, 1, 1, 120])
    # 此处先增加了一维，后面会继续再增加一维


def get_subsequence_mask(seq):
    batch_size, seq_len = seq.size()
    subsequence_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequence_mask


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 position_num,
                 src_padding_index,
                 block_num,
                 head_num,
                 embedding_size,
                 model_size,
                 forward_hidden_size,
                 kq_d,
                 v_d,
                 dropout
                 ):
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx=src_padding_index)
        self.position_embedding = nn.Embedding(position_num, embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(model_size, forward_hidden_size, kq_d, v_d, head_num, dropout=dropout) for _ in range(block_num)]
        )

        self.model_size = model_size

    def forward(self, src_seq, src_position_seq, src_mask=None):
        # print("encoder部分:", src_seq.shape, src_position_seq.shape, src_mask.shape)
        src_token_embedding = self.src_embedding(src_seq)
        src_position_embedding = self.position_embedding(src_position_seq)
        src_embedding = self.dropout(src_token_embedding + src_position_embedding)
        src_embedding = self.layer_norm(src_embedding)

        for encoder_block in self.encoder_blocks:
            output = encoder_block(src_embedding, src_mask)
            src_embedding = output
        # print("src_embedding.shape:", src_embedding.shape)
        return src_embedding


class Decoder(nn.Module):
    def __init__(self,
                 tag_vocab_size,
                 position_num,
                 tag_padding_index,
                 block_num,
                 head_num,
                 embedding_size,
                 model_size,
                 forward_hidden_size,
                 kq_d,
                 v_d,
                 dropout
                 ):
        super(Decoder, self).__init__()
        self.tag_embedding = nn.Embedding(tag_vocab_size, embedding_size, padding_idx=tag_padding_index)
        self.position_embedding = nn.Embedding(position_num, embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(model_size, forward_hidden_size, kq_d, v_d, head_num, dropout=dropout) for _ in range(block_num)]
        )

        self.model_size = model_size

    def forward(self, tag_seq, tag_position_seq, tag_mask, enc_out, src_mask=None):

        tag_token_embedding = self.tag_embedding(tag_seq)
        tag_position_embedding = self.position_embedding(tag_position_seq)
        tag_embedding = self.dropout(tag_token_embedding + tag_position_embedding)
        tag_embedding = self.layer_norm(tag_embedding)

        for decoderblock in self.decoder_blocks:
            output = decoderblock(tag_embedding, enc_out, src_mask, tag_mask)
            tag_embedding = output

        return tag_embedding


class EncoderBlock(nn.Module):
    def __init__(self,
                 model_size,
                 forward_hidden_size,
                 kq_d,
                 v_d,
                 head_num,
                 dropout
                 ):
        super().__init__()
        self.EncSelfAttn = MultiHeadAttention(model_size, kq_d, v_d, head_num, dropout=dropout)
        self.Forward = FeedForward(model_size, forward_hidden_size, dropout=dropout)

    def forward(self, src_embedding, src_mask):
        # print("encoderblock部分:", src_embedding.shape, src_mask.shape)
        output = self.EncSelfAttn(src_embedding, src_embedding, src_embedding, src_mask)
        output = self.Forward(output)
        # print("encoderblock部分output.shape:", output.shape)
        return output


class DecoderBlock(nn.Module):
    def __init__(self,
                 model_size,
                 forward_hidden_size,
                 kq_d,
                 v_d,
                 head_num,
                 dropout
                 ):
        super().__init__()
        self.DecSelfAttn = MultiHeadAttention(model_size, kq_d, v_d, head_num, dropout=dropout)
        self.EncDecAttn = MultiHeadAttention(model_size, kq_d, v_d, head_num, dropout=dropout)
        self.Forward = FeedForward(model_size, forward_hidden_size, dropout=dropout)

    def forward(self, tag_embedding, enc_out, src_mask, tag_mask):
        output = self.DecSelfAttn(tag_embedding, tag_embedding, tag_embedding, tag_mask)
        output = self.EncDecAttn(output, enc_out, enc_out, src_mask)
        output = self.Forward(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 model_size,
                 kq_d,
                 v_d,
                 head_num,
                 dropout):
        super(MultiHeadAttention, self).__init__()
        self.model_size = model_size
        self.kq_d = kq_d
        self.v_d = v_d
        self.head_num = head_num

        self.q_w = nn.Linear(model_size, head_num*kq_d, bias=False)
        self.k_w = nn.Linear(model_size, head_num*kq_d, bias=False)
        self.v_w = nn.Linear(model_size, head_num*v_d, bias=False)

        self.attention = ScaledDotProductAttention(kq_d)

        self.linear = nn.Linear(head_num*v_d, model_size)
        self.layer_norm = nn.LayerNorm(model_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        """

        输入： batch * sentence_len * model_size；
        经过线性层得到QKV矩阵： batch * sentence_len * head_num*kq_d；
        按头数将最后一维分成多份，然后将sentence_len换到倒数第二维，最后一维是qkv的size
        将mask补齐维度，与矩阵一起送入Scaled_Dot_Product_Attention，并接受返回值，q和attn
                    q: batch * head_num * sentence_len(q) * v_d
                    attn: batch * head_num * sentence_len(q) * sentence_len(k)
        将形状reshape成初始形状，经过线性层，dropout与residual相加返回

        :param q: batch * sentence_len * model_size => batch * sentence_len * head_num*kq_d
        :param k: batch * sentence_len * model_size => batch * sentence_len * head_num*kq_d
        :param v: batch * sentence_len * model_size => batch * sentence_len * head_num*v_d
                  after Scaled_Dot_Product_Attention: batch * sentence_len * head_num*v_d
        :param mask:
        :return:
        """
        q_d, k_d, v_d, head_num = self.kq_d, self.kq_d, self.v_d, self.head_num
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # print("++++++", q.shape, k.shape, v.shape, "_______")
        residual = q

        q = self.q_w(q).view(sz_b, len_q, head_num, q_d)
        k = self.k_w(k).view(sz_b, len_k, head_num, k_d)
        v = self.v_w(v).view(sz_b, len_v, head_num, v_d)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask)
        # q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = q.transpose(1, 2).reshape(sz_b, len_q, -1)
        q = self.dropout(self.linear(q))
        q += residual
        q = self.layer_norm(q)
        # print("q.shape", q.shape)
        return q


class FeedForward(nn.Module):
    def __init__(self,
                 model_size,
                 forward_hidden_size,
                 dropout):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(model_size, forward_hidden_size, bias=False)
        self.layer2 = nn.Linear(forward_hidden_size, model_size, bias=False)
        self.activate = mygelu
        self.layer_norm = nn.LayerNorm(model_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # print("forward部分:", x.shape)
        residual = x

        x = self.layer2(self.activate(self.layer1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        # print("x.shape", x.shape)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 kq_d,
                 attn_dropout=0.1):
        """
        计算缩放点积只有 （k_d ** 0.5）需要用到k_d这个参数，只需要传递一个参数。
        :param kq_d:
        :param attn_dropout:
        """
        super(ScaledDotProductAttention, self).__init__()
        self.kq_d = kq_d
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        k和q相乘，乘积加mask（根据k选mask，k是src就用src_mask,k是tag就用tag_mask），attn经过softmax和dropout与v相乘即得缩放点积结果
        :param q: batch * head_num * sentence_len * kq_d
        :param k: batch * head_num * sentence_len * kq_d
        :param v: batch * head_num * sentence_len * v_d
        :param mask: encoder和enc_dec的attn部分是src_mask, decoder的attn部分是tag_mask
        :return:batch * head_num * sentence_len * v_d (the sentence_len is q's sentence_len)
        """
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.kq_d ** 0.5)
        # print(mask)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(f.softmax(attn, dim=-1))

        q = torch.matmul(attn, v)

        return q


def mygelu(x):
    return 0.5 * x * (1 + torch.tanh(((2 / torch.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))
