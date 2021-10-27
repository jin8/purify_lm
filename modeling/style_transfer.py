from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, BertModel, BertConfig
from transformers.models.gpt2.modeling_gpt2 import Block
import time
import random
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths


def avg_pool(hidden, attention_mask):
    return ((hidden * attention_mask.unsqueeze(-1)).sum(1) / ( attention_mask.sum(-1).unsqueeze(-1)+1e-8))

class Projection(nn.Module):
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



class StyleGPT2(GPT2LMHeadModel):
    """Contrastive based GPT2"""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        num_styles = config.num_styles
        self.max_length = config.max_length
        self.hard_negative_weight = config.hard_negative_weight
        self.contrastive_factor = config.contrastive_factor
        #self.block_layer = Block(config.n_ctx, config, scale=True)
        self.style_embed = nn.Embedding(num_styles, config.n_embd)
        self.style_linear = nn.Linear(config.n_embd, num_styles)
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.project = Projection(config)
        self.sim = Similarity(temp=0.07)



    def contrastive_loss(self, z1, z2, z3=None):
        # Gather all embeddings if using distributed training
        device = z1.device
        if dist.is_initialized() and self.training:
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

            if z3 is not None:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)


        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        if z3 is not None:
            # Hard negative
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
            # Hard negative
            z2_z3_cos = self.sim(z3.unsqueeze(1), z2.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z2_z3_cos], 1)
        labels = torch.arange(cos_sim.size(0)).long().to(device)
        loss_fct = nn.CrossEntropyLoss()
        # Calculate loss with hard negatives
        if z3 is not None:
            z3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)
        return loss

    def get_lengths(self, tokens):
        lengths = torch.cumsum(tokens == self.eos_token_id, 1)
        lengths = (lengths == 0).long().sum(-1)
        lengths = lengths + 1 # +1 for <eos> token
        return lengths

    def self_reconstruct(self,
        input_ids=None,
        attention_mask=None,
        attr_labels=None,
        labels=None):

        style_emb = self.style_embed(attr_labels).unsqueeze(1)
        input_emb = self.transformer.wte(input_ids)

        input_emb = torch.cat([style_emb, input_emb], dim=1)
        input_attention_mask = torch.cat(
            [torch.ones_like(attention_mask[:,:1]),
            attention_mask], dim=1)


        outputs = self.transformer(
            inputs_embeds=input_emb,
            attention_mask=input_attention_mask)
        hidden_states=outputs[0]
        style_pred = self.style_linear(hidden_states[:,-1,:])

        #if attention_mask is not None:
        #    block_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        #    block_attention_mask = block_attention_mask[:, None, None, :]
        #    block_attention_mask = block_attention_mask.to(dtype=self.dtype)
        #    block_attention_mask = (1.0 - block_attention_mask) * -10000.0
        z1 = avg_pool(hidden_states, input_attention_mask)
        z1 = F.normalize(self.project(z1), dim=-1)
        z1 = z1.view((-1, z1.size(-1)))

        rev_style_emb = self.style_embed(1-attr_labels).unsqueeze(1)
        input_emb = self.transformer.wte(input_ids)

        rev_input_emb = torch.cat([rev_style_emb, input_emb], dim=1)
        rev_attention_mask = torch.cat(
            [torch.ones_like(attention_mask[:,:1]),
            attention_mask], dim=1)
        rev_outputs = self.transformer(
            inputs_embeds=rev_input_emb,
            attention_mask=rev_attention_mask)
        rev_hidden_states=rev_outputs[0]
        #orig_block_hidden_states = self.block_layer(
        #    hidden_states+style_emb,
        #    attention_mask=block_attention_mask,
        #)[0]
        z2 = avg_pool(rev_hidden_states, rev_attention_mask)
        z2 = F.normalize(self.project(z2), dim=-1)
        z2 = z2.view((-1, z2.size(-1)))
        #rev_block_hidden_states = self.block_layer(
        #    hidden_states+rev_style_emb,
        #    attention_mask=block_attention_mask,
        #)[0]
        rev_style_pred = self.style_linear(rev_hidden_states[:,-1,:])

        orig_lm_logits = self.lm_head(hidden_states)
        rev_lm_logits = self.lm_head(rev_hidden_states)

        rev_log_probs = F.log_softmax(rev_lm_logits, dim=-1)
        rev_soft_token = rev_log_probs.exp()[..., :-1, :].contiguous()
        rev_rev_input_emb = torch.matmul(rev_soft_token, self.transformer.wte.weight)
        rev_rev_input_emb = torch.cat([style_emb, rev_rev_input_emb], dim=1)
        rev_rev_attention_mask = torch.cat(
            [torch.ones_like(attention_mask[:,:1]),
            attention_mask], dim=1)

        rev_rev_outputs = self.transformer(
            inputs_embeds=rev_rev_input_emb,
            attention_mask=rev_rev_attention_mask)
        rev_rev_hidden_states=rev_rev_outputs[0]
        z3 = avg_pool(rev_rev_hidden_states, rev_rev_attention_mask)
        z3 = F.normalize(self.project(z3), dim=-1)
        z3 = z3.view((-1, z3.size(-1)))
        #rev_orig_block_hidden_states = self.block_layer(
        #    rev_hidden_states+style_emb,
        #    attention_mask=block_attention_mask,
        #)[0]
        rev_rev_style_pred = self.style_linear(rev_rev_hidden_states[:,-1,:])

        rev_rev_lm_logits = self.lm_head(rev_rev_hidden_states)
        loss_fct = nn.CrossEntropyLoss()
        # Shift so that tokens < n predict n

        shift_logits = orig_lm_logits[..., :-1, :].contiguous()
        rev_shift_logits = rev_rev_lm_logits[..., :-1, :].contiguous()

        shift_labels = labels.contiguous()

        rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        rev_rec_loss = 0.25*loss_fct(rev_shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        cont_loss = self.contrastive_loss(z1, z3, z2)
        pred_loss = loss_fct(style_pred, attr_labels) + loss_fct(rev_style_pred, 1-attr_labels) + loss_fct(rev_rev_style_pred, attr_labels)

        print('{:0.5f} \t {:0.5f} \t {:0.5f} \t {:0.5f} \n'.format(rec_loss, rev_rec_loss, cont_loss, pred_loss))
        loss = rec_loss + rev_rec_loss + cont_loss + pred_loss
        return (loss,)
