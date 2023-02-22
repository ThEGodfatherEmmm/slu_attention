# coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F


class slu_attention(nn.Module):
    def __init__(self, config):
        super(slu_attention, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=0
        )
        self.rnn = getattr(nn, self.cell)(
            config.embed_size,
            config.hidden_size // 2,
            num_layers=config.num_layer,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(
            config.hidden_size, config.num_tags, config.tag_pad_idx
        )

        self._e_attention = SelfAttention(
            config.embed_size, config.attention_size, config.dropout
        )
        self._lstm_layer = nn.LSTM(
            input_size=config.embed_size + config.attention_size,
            hidden_size=config.hidden_size // 2,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            dropout=config.dropout,
        )
        self._d_attention = SelfAttention(
            config.hidden_size, config.attention_size, config.dropout
        )
        self._intent_pred_linear = nn.Linear(config.hidden_size, config.vocab_size)
        self._intent_gate_linear = nn.Linear(
            config.attention_size + config.vocab_size, config.hidden_size
        )
        self._slot_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        # embed = self.word_embed(input_ids)
        # packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True)
        # packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        # rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        # hiddens = self.dropout_layer(rnn_out)
        # tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        # return tag_output
        # embed = self._embedding_layer(input_w, input_l)
        
        embed = self.word_embed(input_ids)
        attention_x = self._e_attention(embed)
        emb_attn_x = torch.cat([embed, attention_x], dim=-1)
        lstm_hidden, _ = self._lstm_layer(emb_attn_x)

        pool_hidden = torch.mean(lstm_hidden, dim=1, keepdim=True)
        linear_intent = self._intent_pred_linear(pool_hidden)

        # 预测一个 batch 的 intent 负对数分布.
        # pred_intent = F.log_softmax(linear_intent.squeeze(1), dim=-1)

        rep_intent = torch.cat([linear_intent] * max(lengths), dim=1)
        attn_hidden = self._d_attention(lstm_hidden)
        com_hidden = torch.cat([rep_intent, attn_hidden], dim=-1)
        lin_hidden = self._intent_gate_linear(com_hidden)
        gated_hidden = lin_hidden * lstm_hidden

        linear_slot = self._slot_linear(gated_hidden)
        # expand_slot = [
        #     linear_slot[i][: lengths[i], :] for i in range(0, len(lengths))
        # ]  # 去掉padding部分
        
        # pred_slot = F.log_softmax(torch.cat(expand_slot, dim=0), dim=-1)
        # pred_slot = torch.cat(attention_out, dim=-1)
        # print(linear_slot.shape)
        hiddens = self.dropout_layer(linear_slot)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

        # return pred_slot, pred_intent

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()
    def decode_(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels


class TaggingFNNDecoder(nn.Module):
    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        # logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob


class SelfAttention(nn.Module):
    """
    基于 KVQ 计算模式的自注意力机制.
    """

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        self._k_matrix = nn.Linear(input_dim, output_dim)
        self._v_matrix = nn.Linear(input_dim, output_dim)
        self._q_matrix = nn.Linear(input_dim, output_dim)
        self._dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        k_x = self._k_matrix(input_x)
        v_x = self._v_matrix(input_x)
        q_x = self._q_matrix(input_x)

        drop_kx = self._dropout_layer(k_x)
        drop_vx = self._dropout_layer(v_x)
        drop_qx = self._dropout_layer(q_x)

        alpha = F.softmax(torch.matmul(drop_qx.transpose(-2, -1), drop_kx), dim=-1)
        return torch.matmul(drop_vx, alpha)
