from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, BertModel, BertConfig
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
        config.add_cross_attention = True
        super().__init__(config)

        self.config = config
        num_styles = config.num_styles
        self.max_length = config.max_length
        self.hard_negative_weight = config.hard_negative_weight
        self.style_embed = nn.Embedding(num_styles, config.n_embd)
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.project = Projection(config)
        self.sim = Similarity(temp=0.07)
        self.generate = False
        self.differentiable_decode = False

    def contrastive_loss(self, z1, z2, z3=None):
        # Gather all embeddings if using distributed training
        batch_size = len(z1)
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

        labels = torch.arange(cos_sim.size(0)).long().to(device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if z3 is not None:
            z3_weight = self.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels.view(batch_size, -1))
        return loss

    def get_lengths(self, tokens):
        lengths = torch.cumsum(tokens == self.eos_token_id, 1)
        lengths = (lengths == 0).long().sum(-1)
        lengths = lengths + 1 # +1 for <eos> token
        return lengths

    def forward(self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            attr_labels=None,
            labels=None,
            use_cache=False,
            pretrain=False):
        if pretrain:
            self_recon_loss = self.self_reconstruct(input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            attr_labels=attr_labels,
            labels=labels,
            use_cache=use_cache)
            return self_recon_loss
        else:
            return self_recon_loss, cycle_recon_loss



    def self_reconstruct(self,
            input_ids=None,
            attention_mask=None,
            attr_labels=None,
            gen_input_ids=None,
            labels=None):

        batch_size = len(attr_labels)
        #print(attention_mask.size())
        max_enc_len = torch.max(attention_mask.sum(-1)).item()
        device = input_ids.device

        gen_style_emb = self.style_embed(attr_labels).unsqueeze(1)

        if len(input_ids.size()) == 2:
            enc_input_emb = self.transformer.wte(input_ids)
        else:
            enc_input_emb = torch.matmul(input_ids, self.transformer.wte.weight)

        enc_outputs = self.transformer(
            attention_mask=attention_mask,
            inputs_embeds=enc_input_emb)
        enc_hidden_states=enc_outputs[0]

        enc_pool = avg_pool(enc_hidden_states, attention_mask)
        enc_proj = F.normalize(self.project(enc_pool), dim=-1)
        enc_proj = enc_proj.view((batch_size, -1, enc_proj.size(-1)))

        dec_input_emb = self.transformer.wte(input_ids)
        dec_input_emb = torch.cat((gen_style_emb, dec_input_emb), 1)

        dec_attention_mask = torch.cat((torch.ones_like(attention_mask[:, :1]), attention_mask), 1)
        dec_attention_mask = dec_attention_mask.view(batch_size,-1)
        dec_attention_mask = dec_attention_mask.to(device)
        transformer_outputs = self.transformer(
            inputs_embeds=dec_input_emb,
            attention_mask=dec_attention_mask,
            encoder_hidden_states=enc_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        dec_hidden_states = transformer_outputs[0]

        dec_lm_logits = self.lm_head(dec_hidden_states)
        dec_log_probs = F.log_softmax(dec_lm_logits, dim=-1)
        dec_pool = avg_pool(dec_hidden_states, dec_attention_mask)
        dec_proj = F.normalize(self.project(dec_pool), dim=-1)
        dec_proj = dec_proj.view((batch_size, -1, dec_proj.size(-1)))
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()

            loss = \
                loss_fn(dec_lm_logits[..., :-1, :].contiguous().transpose(1, 2), labels)
            return (loss, dec_lm_logits[..., :-1, :], enc_proj, dec_proj)
        else:
            return dec_log_probs[..., :-1, :].contiguous()


    def generate_back(self,
            gen_input_ids=None,
            gen_attention_mask=None,
            input_ids=None,
            attention_mask=None,
            attr_labels=None,
            labels=None):

        batch_size = len(attr_labels)
        #print(attention_mask.size())
        max_enc_len = torch.max(gen_attention_mask.sum(-1)).item()

        device = input_ids.device

        gen_style_emb = self.style_embed(attr_labels).unsqueeze(1)

        if len(gen_input_ids.size()) == 2:
            enc_input_emb = self.transformer.wte(gen_input_ids)
        else:
            enc_input_emb = torch.matmul(gen_input_ids, self.transformer.wte.weight)
        print(max_enc_len)
        print(gen_attention_mask)
        print(enc_input_emb.size())
        enc_outputs = self.transformer(
            attention_mask=gen_attention_mask,
            inputs_embeds=enc_input_emb[:,:max_enc_len])
        enc_hidden_states=enc_outputs[0]
        print(enc_hidden_states.size())
        enc_pool = avg_pool(enc_hidden_states, gen_attention_mask)
        enc_proj = F.normalize(self.project(enc_pool), dim=-1)
        enc_proj = enc_proj.view((batch_size, -1, enc_proj.size(-1)))

        dec_input_emb = self.transformer.wte(input_ids)
        dec_input_emb = torch.cat((gen_style_emb, dec_input_emb), 1)

        dec_attention_mask = torch.cat((torch.ones_like(attention_mask[:, :1]), attention_mask), 1)
        dec_attention_mask = dec_attention_mask.view(batch_size,-1)
        dec_attention_mask = dec_attention_mask.to(device)
        print(dec_input_emb.size())
        print(dec_attention_mask.size())
        transformer_outputs = self.transformer(
            inputs_embeds=dec_input_emb,
            attention_mask=dec_attention_mask,
            encoder_hidden_states=enc_hidden_states,
            encoder_attention_mask=gen_attention_mask,
        )
        dec_hidden_states = transformer_outputs[0]

        dec_lm_logits = self.lm_head(dec_hidden_states)
        dec_log_probs = F.log_softmax(dec_lm_logits, dim=-1)
        dec_pool = avg_pool(dec_hidden_states, dec_attention_mask)
        dec_proj = F.normalize(self.project(dec_pool), dim=-1)
        dec_proj = dec_proj.view((batch_size, -1, dec_proj.size(-1)))
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()

            loss = \
                loss_fn(dec_lm_logits[..., :-1, :].contiguous().transpose(1, 2), labels)
            return loss, dec_lm_logits[..., :-1, :], \
            enc_proj, dec_proj
        else:
            return dec_log_probs[..., :-1, :].contiguous()


    def generate_by_attr(self,
            input_ids=None,
            attention_mask=None,
            attr_labels=None,
            labels=None,
            differentiable_decode=True):
        batch_size = len(attr_labels)
        #print(attention_mask.size())
        max_enc_len = torch.max(attention_mask.sum(-1)).item()
        device = input_ids.device

        gen_style_emb = self.style_embed(attr_labels).unsqueeze(1)

        if len(input_ids.size()) == 2:
            enc_input_emb = self.transformer.wte(input_ids)
        else:
            enc_input_emb = torch.matmul(input_ids, self.transformer.wte.weight)

        enc_outputs = self.transformer(
            attention_mask=attention_mask,
            inputs_embeds=enc_input_emb)
        enc_hidden_states=enc_outputs[0]
        #enc_lm_logits = self.enc_lm_head(enc_hidden_states)
        #enc_log_probs = F.log_softmax(enc_lm_logits / temperature, dim=-1)
        enc_pool = avg_pool(enc_hidden_states, attention_mask)
        enc_proj = F.normalize(self.project(enc_pool), dim=-1)
        enc_proj = enc_proj.view((batch_size, -1, enc_proj.size(-1)))

        dec_log_probs = []
        dec_lm_logits = []

        dec_hidden_states = []
        dec_input_emb = gen_style_emb
        prev_states = None
        max_length = self.max_length
        for k in range(max_length):
            dec_outputs = self.transformer(
                past_key_values=prev_states,
                attention_mask=torch.ones((batch_size,1)).to(device),
                inputs_embeds=dec_input_emb,
                encoder_hidden_states=enc_hidden_states,
                encoder_attention_mask=attention_mask,
                use_cache=True
            )
            dec_hidden_state = dec_outputs[0]
            prev_states = dec_outputs[1]
            dec_lm_logit = self.lm_head(dec_hidden_state)
            dec_log_prob = F.log_softmax(dec_lm_logit, dim=-1)
            dec_log_probs.append(dec_log_prob)
            dec_lm_logits.append(dec_lm_logit)
            dec_hidden_states.append(dec_hidden_state)


            if differentiable_decode:
                dec_input_emb = torch.matmul(dec_log_prob.exp(), self.transformer.wte.weight)
            else:
                dec_input_emb = self.transformer.wte(dec_log_prob.argmax(-1))

            #if (pred_tokens == self.eos_idx).max(-1)[0].min(-1)[0].item() == 1:
            #    break

        dec_log_probs = torch.cat(dec_log_probs, 1)
        dec_lm_logits = torch.cat(dec_lm_logits, 1)

        dec_hidden_states = torch.cat(dec_hidden_states, 1)
        gen_soft_tokens = dec_log_probs.exp()
        gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), self.eos_token_id)
        gen_pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        gen_pos_idx = gen_pos_idx.to(device)
        gen_mask = gen_pos_idx <= gen_lengths.unsqueeze(-1)

        dec_pool = avg_pool(dec_hidden_states, gen_mask)
        dec_proj = F.normalize(self.project(dec_pool), dim=-1)
        dec_proj = dec_proj.view((batch_size, -1, dec_proj.size(-1)))
        if labels is not None:
            loss_fn = nn.NLLLoss()

            loss = \
                loss_fn(dec_log_probs[..., :-1, :].contiguous().transpose(1, 2), labels)
            return loss, \
                enc_proj, dec_proj
        else:
            return dec_lm_logits[..., :-1, :].contiguous(), \
                enc_proj, dec_proj


    def cyclic_reconstruct(self,
            input_ids=None,
            attention_mask=None,
            attr_labels=None,
            labels=None):
        batch_size = len(attr_labels)
        device = input_ids.device
        rev_attr_labels = 1- attr_labels

        gen_lm_logits, z1, z2 = self.generate_by_attr(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attr_labels=rev_attr_labels,
            labels=None
        )
        gen_soft_tokens = F.log_softmax(gen_lm_logits, dim=-1).exp()
        gen_lengths = self.get_lengths(gen_soft_tokens.argmax(-1))-1
        print(gen_lengths)
        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(device)
        max_enc_len = torch.max(gen_lengths).item()
        gen_attention_mask = pos_idx[:, :max_enc_len] <= gen_lengths.unsqueeze(-1)

        cyc_rec_loss, _, _, z3 = self.generate_back(
            gen_input_ids=gen_soft_tokens,
            gen_attention_mask=gen_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            attr_labels=attr_labels,
            labels=labels
        )
        return cyc_rec_loss
