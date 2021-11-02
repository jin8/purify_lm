from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2Config
import time
import numpy as np
import random
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs[0]
        hidden_states = outputs[-1]

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(-1).unsqueeze(-1) + 1e-8))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / (
                        attention_mask.sum(-1).unsqueeze(-1) + 1e-8)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / (
                        attention_mask.sum(-1).unsqueeze(-1) + 1e-8)
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MappingGPT2(GPT2LMHeadModel):
    """Mapping based GPT2"""

    def __init__(self, config):
        super().__init__(config)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # n_gpu = torch.cuda.device_count()
        self.ablation = config.ablation

        self.pooler = Pooler('avg')
        self.pooler2 = Pooler('avg')
        self.mlp = MLPLayer(config)
        self.mlp2 = MLPLayer(config)

        self.sim = Similarity(temp=0.07)
        self.attr_embedding = nn.Embedding(2, config.hidden_size)

        self.block = transformers.models.gpt2.modeling_gpt2.Block(config.n_ctx, config, scale=True)

        self.discrim = nn.Linear(config.hidden_size, 2)
        if 'freeze' in self.ablation:
            self.freeze_gpt()
        self.alpha = config.alpha
        self.mse_loss = config.mse_loss
        self.mixup_loss = config.mixup_loss
        self.supervised = config.supervised

    def freeze_gpt(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def mixup_hidden(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        if 'rand' in self.ablation:
            index = torch.randperm(batch_size).cuda()
        else:
            cnt, cnt_zero = 0, 0
            index = torch.zeros(batch_size).cuda()
            one_index = (y == 1).nonzero(as_tuple=True)[0]
            zero_index = (y == 0).nonzero(as_tuple=True)[0]
            for k in range(batch_size):
                if y[k] == 0:
                    if cnt < len(one_index):
                        index[k] = one_index[cnt].long()
                        cnt += 1
                    else:
                        index[k] = zero_index[cnt_zero].long()
                        cnt_zero += 1
                elif y[k] == 1:
                    if cnt_zero < len(zero_index):
                        index[k] = zero_index[cnt_zero].long()
                        cnt_zero += 1
                    else:
                        index[k] = one_index[cnt].long()
                        cnt += 1
            index = index.long()

        mixed_hidden = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_hidden, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):

        criterion = nn.CrossEntropyLoss()

        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def gather(self, z1, z2, labels=None):
        # Gather all embeddings if using distributed training
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]

        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        if labels is not None:
            labels_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=labels_list, tensor=labels.contiguous())
            labels_list[dist.get_rank()] = labels
            labels = torch.cat(labels_list, 0)
            return z1, z2, labels

        else:
            return z1, z2

    def sup_contrastive_loss(self, z1, z2, labels, temperature=0.07, contrast_mode='all',
                             base_temperature=0.07):

        # z1 : BxD
        # z2 : BxD
        features = torch.stack([z1, z2], dim=1)
        batch_size = len(features)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(labels.device)
        contrast_count = features.shape[1]  # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 2*B x D
        if contrast_mode == 'one':  # n_views = 2
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits BxD 2BxD -> Bx2B
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask BxB -> BxB
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(anchor_feature.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def self_contrastive_loss(self, z1, z2, diff=False):
        loss_fct = nn.CrossEntropyLoss()
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        if diff:
            cos_loss = loss_fct(cos_sim, cos_labels.unsqueeze(1))
        else:
            cos_loss = loss_fct(cos_sim, cos_labels)
        return cos_loss

    def consistency_loss(self, prev_t1, prev_t2, aft_t1, aft_t2, prev_n1, prev_n2, aft_n1, aft_n2):

        consistency_t, consistency_n, total_mse, mse_distance = 0, 0, 10, 0
        l1_loss = nn.L1Loss()

        if not (aft_t1.size(0) == 0):
            prevt_sim = self.sim(prev_t1, prev_t2)
            aftt_sim = self.sim(aft_t1, aft_t2)
            consistency_t = l1_loss(aftt_sim, prevt_sim.detach()).mean()
        if not (aft_n1.size(0) == 0):
            prevn_sim = self.sim(prev_n1, prev_n2)
            aftn_sim = self.sim(aft_n1, aft_n2)
            consistency_n = l1_loss(aftn_sim, prevn_sim.detach()).mean()

        mse_loss = nn.MSELoss()
        cnt = 0

        if aft_t1.size(0) > 0 and aft_n1.size(0) > 0:

            for nb in range((aft_n1.size(0))):
                for tb in range((aft_t1.size(0))):
                    # print(mse_loss(aft_n1[nb].view(-1), aft_t1[tb].view(-1)))
                    total_mse -= mse_loss(aft_n1[nb], aft_t1[tb])
                    cnt += 1

            mse_distance = float(float(total_mse) / cnt)

        return consistency_t, consistency_n, mse_distance

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                labels=None,
                attr_ids=None,
                attr_labels=None,
                use_cache=False,
                gen_approach=None):

        if not (gen_approach is None):
            self.ablation = gen_approach

        batch_size = len(attr_labels)

        if len(attr_labels.size()) == 1:
            attr_labels = attr_labels.unsqueeze(-1)
        if labels is not None:
            labels = labels.view((-1, labels.size(-1)))
            if 'cont' in self.ablation:
                input_ids = torch.stack([input_ids, input_ids], dim=1)
                input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)

                attention_mask = torch.stack([attention_mask, attention_mask], dim=1)
                attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)

                labels = torch.stack([labels, labels], dim=1)
                labels = labels.view((-1, labels.size(-1)))  # (bs * num_sent len)

                batch_attr_label = attr_labels

                attr_labels = torch.stack([attr_labels, attr_labels], dim=1)
                attr_labels = attr_labels.view((-1, attr_labels.size(-1)))  # (bs * num_sent len)

        attr_embed = self.attr_embedding(attr_labels)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if 'attention' in self.ablation:
            hidden_states = torch.cat((attr_embed, transformer_outputs[0]), dim=1)
        else:
            hidden_states = transformer_outputs[0] + attr_embed

        if labels is not None:
            if 'cont' in self.ablation:
                pooler_output = self.pooler(attention_mask, transformer_outputs)
                mlp_output = F.normalize(self.mlp(pooler_output), dim=-1)
                mlp_output = mlp_output.view((batch_size, -1, mlp_output.size(-1)))
                z1 = mlp_output[:, 0]
                z2 = mlp_output[:, 1]
                if dist.is_initialized() and self.training:
                    z1, z2, _ = self.gather(z1, z2, batch_attr_label)
                cos_loss = self.self_contrastive_loss(z1, z2, diff=False)

        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            if 'cont' in self.ablation:
                b_size = batch_size * 2
            else:
                b_size = batch_size
            if 'attention' in self.ablation:
                block_attention_mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1)
                block_attention_mask = block_attention_mask.view(b_size, -1)
            else:
                block_attention_mask = attention_mask.view(b_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            block_attention_mask = block_attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            block_attention_mask = block_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            block_attention_mask = (1.0 - block_attention_mask) * -10000.0

        embed_hidden = self.block(hidden_states, attention_mask=block_attention_mask)
        if labels is not None:
            if 'consistency' in self.ablation:
                pooler_output2 = self.pooler2(block_attention_mask, embed_hidden)
                mlp_output2 = F.normalize(self.mlp2(pooler_output2), dim=-1)
                mlp_output2 = mlp_output2.view((batch_size, -1, mlp_output2.size(-1)))
                z11 = mlp_output2[:, 0]
                z22 = mlp_output2[:, 1]
                if dist.is_initialized() and self.training:
                    z11, z22, gather_attr_labels = self.gather(z11, z22, batch_attr_label)

                toxic_index, non_index = (gather_attr_labels.squeeze() == 1), (gather_attr_labels.squeeze() == 0)
                prev_toxic_z1, prev_non_z1 = z1[toxic_index], z1[non_index]
                aft_toxic_z1, aft_non_z1 = z11[toxic_index], z11[non_index]
                prev_toxic_z2, prev_non_z2 = z2[toxic_index], z2[non_index]
                aft_toxic_z2, aft_non_z2 = z22[toxic_index], z22[non_index]
                # prev_t1, prev_t2, aft_t1, aft_t2, prev_n1, prev_n2, aft_n1, aft_n2
                t, n, mse = self.consistency_loss(prev_toxic_z1, prev_toxic_z2, aft_toxic_z1, aft_toxic_z2, prev_non_z1,
                                                  prev_non_z2, aft_non_z1, aft_non_z2)
                consistency_loss = t + n + mse
        embed_hidden = embed_hidden[0]

        if labels is not None:
            if self.mixup_loss:
                mixed_hidden, y_a, y_b, lam = self.mixup_hidden(embed_hidden.sum(1).squeeze(1), attr_labels, 1.0)
                fc_output = self.discrim(mixed_hidden)
            else:
                fc_output = self.discrim(embed_hidden.sum(1).squeeze(1))

        lm_logits = self.lm_head(embed_hidden)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:

            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            if 'attention' in self.ablation:
                shift_labels = labels.contiguous()
            else:
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = rec_loss
            if 'ce' in self.ablation:
                if self.mixup_loss:
                    mixup_loss = self.mixup_criterion(fc_output, y_a.squeeze(), y_b.squeeze(), lam)
                    loss += mixup_loss
                else:
                    loss += loss_fct(fc_output, attr_labels.squeeze())

            if 'continuation' in self.ablation:
                loss += cos_loss

            if 'consistency' in self.ablation:
                loss += consistency_loss

            # print("cos: ", cos_loss, "/ mse: ", mse, "/ consistency: ", (t+n), "/ logit: ",rec_loss )
            outputs = (loss,) + outputs

            # print("cos:",cos_loss)
            # print(input_ids)
            # print(attr_ids)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
