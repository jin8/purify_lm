from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, GPT2Config
import time
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
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs[0]
        hidden_states = outputs[-1]

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) /( attention_mask.sum(-1).unsqueeze(-1)+1e-8))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(-1).unsqueeze(-1)+1e-8)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(-1).unsqueeze(-1)+1e-8)
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

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #n_gpu = torch.cuda.device_count()
        self.ablation = config.ablation

        self.pooler = Pooler('avg')
        self.mlp = MLPLayer(config)

        self.sim = Similarity(temp=0.07)
        self.attr_embedding = nn.Embedding(2, config.hidden_size)

        self.block = transformers.models.gpt2.modeling_gpt2.Block(config.n_ctx, config, scale=True)

        self.discrim = nn.Linear(config.hidden_size, 2)

        self.alpha = config.alpha
        self.mse_loss = config.mse_loss

        self.supervised = config.supervised


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
        features = torch.stack([z1,z2], dim=1)
        batch_size = len(features)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(labels.device)
        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  #2*B x D
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
        attr_embed = self.attr_embedding(attr_labels)

        transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = transformer_outputs[0] + attr_embed
        
        if labels is not None:
            if 'cont' in self.ablation:
                pooler_output = self.pooler(attention_mask, transformer_outputs)
                mlp_output = F.normalize(self.mlp(pooler_output), dim=-1)
                mlp_output = mlp_output.view((batch_size, -1, mlp_output.size(-1)))
                if self.ablation == 'cont_att':
                    z1 = mlp_output
                    z2 = attr_embed
                if dist.is_initialized() and self.training:
                    z1, z2, _ = self.gather(z1, z2, attr_labels)
                cos_loss = self.self_contrastive_loss(z1, z2, diff=True)
            
        if not (self.ablation == 'gpt_att'):        
            if attention_mask is not None:
                assert batch_size > 0, "batch_size has to be defined and > 0"
                block_attention_mask = attention_mask.view(batch_size, -1)
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
            embed_hidden = embed_hidden[0]
            fc_output = self.discrim(embed_hidden.sum(1).squeeze(1))
        else:
            embed_hidden = hidden_states
        lm_logits = self.lm_head(embed_hidden)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            
            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = rec_loss
            if not (self.ablation == 'gpt_att'):
                loss += loss_fct(fc_output, attr_labels.squeeze())
            if 'cont' in self.ablation:
                loss += cos_loss 
            if self.mse_loss:
                mse_loss = nn.MSELoss()
                loss += mse_loss(embed_hidden, hidden_states.detach()).mean()
            
            outputs = (loss,) + outputs
              
            #print("cos:",cos_loss)
            #print(input_ids)
            #print(attr_ids)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
