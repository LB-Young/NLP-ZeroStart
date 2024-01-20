import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        """

        :param temperature: 论文中的 根号dk
        :param attn_dropout:
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """

        :param q:  b x n x lq x dk
        :param k:  b x n x lq x dk
        :param v:  b x n x lq x dv
        :param mask:
        :return:
        """

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # b x n x lq x lq
        """
        mask_shape: torch.Size([32, 1, 1, 120])
        attn_shape: torch.Size([32, 2, 120, 120])
        mask_shape: torch.Size([32, 1, 30, 30])
        attn_shape: torch.Size([32, 2, 30, 30])
        """
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        # b x n x lq x dv

        return output, attn
