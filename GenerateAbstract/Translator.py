''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import Transformer, get_subsequence_mask


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        """

        :param model: 模型
        :param beam_size:
        :param max_seq_len:  与输出最大长度相等
        :param src_pad_idx:
        :param trg_pad_idx:
        :param trg_bos_idx:  bos开始的index
        :param trg_eos_idx:  eos的index
        """

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.tag_pad_idx = trg_pad_idx
        self.trg_bos_idx = trg_bos_idx      # [CLS]
        self.trg_eos_idx = trg_eos_idx      # [SEP]

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self, trg_seq, trg_position_seq, tag_mask, enc_output, src_mask):

        # print("trg_seq.shape:", trg_seq.shape)
        # print("trg_position_seq.shape:", trg_position_seq.shape)
        # print("enc_output.shape:", enc_output.shape)
        # print("src_mask.shape:", src_mask.shape)
        '''
        trg_seq.shape: torch.Size([1, 1])
        trg_position_seq.shape: torch.Size([1, 30])
        enc_output.shape: torch.Size([1, 120, 128])
        src_mask.shape: torch.Size([1, 1, 120])

        trg_seq.shape: torch.Size([5, 2])
        trg_position_seq.shape: torch.Size([1, 30])
        enc_output.shape: torch.Size([5, 120, 128])
        src_mask.shape: torch.Size([1, 1, 120])
        '''
        # trg_mask = get_subsequent_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq, trg_position_seq, tag_mask, enc_output, src_mask)
        # print(dec_output.shape)
        return F.softmax(self.model.classifier(dec_output), dim=-1)

    def _get_init_state(self, src_seq, src_mask, src_position_seq, tag_seq, tag_mask, trg_position_seq):
        """

        :param src_seq:    torch.Size([1, 120])
        :param src_mask:    torch.Size([1, 1, 120])
        :param src_position_seq:    torch.Size([1, 120])
        :param tag_seq:    torch.Size([1, 1])
        :param tag_mask:    torch.Size([1, 1, 1])
        :param trg_position_seq:    torch.Size([1, 30])
        :return:
        """
        # print(src_seq.shape, src_mask.shape, src_position_seq.shape, tag_seq.shape, tag_mask.shape, trg_position_seq.shape)
        beam_size = self.beam_size

        # print("==============", src_seq.shape, src_position_seq.shape, src_mask.shape, "==========")
        enc_output = self.model.encoder(src_seq, src_position_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, trg_position_seq, tag_mask, enc_output, src_mask)
        
        best_k_probs, best_k_idx = dec_output[:, len(tag_seq), :].topk(beam_size)
        # print(best_k_probs, best_k_idx)
        # tensor([[0.0004, 0.0004, 0.0004, 0.0004, 0.0004]])
        # tensor([[2560, 3790, 2579, 3355, 1]])
        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq, input_position_seq, tag_seq, tag_position_seq):
        # print("src_seq:", src_seq.shape)
        # print("input_position_seq:", input_position_seq.shape)
        # print("tag_position_seq:", tag_position_seq.shape)
        """

        :param src_seq: [1, 120]
        :param input_position_seq: [1, 120]
        :param tag_position_seq: [1, 30]
        :return:
        """
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == input_position_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            """
            src_seq: torch.Size([1, 120]) => src_mask: torch.Size([1, 1, 120]);
            src_mask是True or False 的tensor，padding的位置为False
            input_position_mask = get_pad_mask(input_position_seq, src_pad_idx)
            不需要input_position_mask,因为mask与src_mask完全相同
            print("self.init_seq:", self.init_seq)
            """
            tag_mask = get_pad_mask(self.init_seq, self.tag_pad_idx) & get_subsequence_mask(self.init_seq)
            # print(src_mask.shape, tag_mask.shape)  torch.Size([1, 1, 120]) torch.Size([1, 1, 1])
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask, input_position_seq, self.init_seq, tag_mask, tag_position_seq)
            # 此处的Decoder是从self.init_seq生成第一个字，第一个字以后的部分在下面的循环中生成
            """
            print("enc_output.shape:", enc_output.shape) => torch.Size([5, 120, 128])
            print("gen_seq.shape:", gen_seq.shape) => torch.Size([5, 30])
            print("scores.shape:", scores.shape) => torch.Size([5])
            """

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:1, :step], tag_position_seq[:, :step], tag_mask, enc_output[:1, :, :], src_mask)
                """
                dec_output = self._model_decode(gen_seq[:1, :step], tag_position_seq[:, :step], tag_mask, enc_output[:1, :, :], src_mask)
                """
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
