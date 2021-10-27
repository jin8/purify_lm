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
        super().__init__(config)

        self.config = config
        num_styles = config.num_styles
        self.max_length = config.max_length
        self.hard_negative_weight = config.hard_negative_weight
        self.contrastive_factor = config.contrastive_factor
        self.enc_model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.style_embed = nn.Embedding(num_styles, config.n_embd)
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.project = Projection(config)
        self.sim = Similarity(temp=0.07)
        for p in self.enc_model.parameters():
            p.requires_grad=False


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
        content_hidden_states = self.enc_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        content_emb = avg_pool(content_hidden_states, attention_mask)

        style_emb = self.style_embed(attr_labels).unsqueeze(1)
        input_emb = self.transformer.wte(input_ids)
        prior_emb = style_emb+content_emb.unsqueeze(1)
        input_emb = torch.cat([prior_emb, input_emb], dim=1)
        attention_mask = torch.cat(
            [torch.ones_like(attention_mask[:,:1]),
            attention_mask], dim=1)
        position_ids = attention_mask.cumsum(dim=1) - 1

        outputs = self.transformer(
            inputs_embeds=input_emb,
            attention_mask=attention_mask,
            position_ids=position_ids)
        hidden_states=outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
                lm_logits[..., :-1, :].contiguous().transpose(1, 2), labels)
        z1 = F.normalize(self.project(content_emb), dim=-1)
        z1 = z1.view((-1, z1.size(-1)))

        z2 = avg_pool(hidden_states, attention_mask)
        z2 = F.normalize(self.project(z2), dim=-1)
        z2 = z2.view((-1, z2.size(-1)))
        print('self: {:0.5f} '.format(
            loss.item()))
        #loss += contrast_loss
        return loss, z2


    def generate_to_original(self,
        gen_attr_labels=None,
        gen_input_ids=None,
        gen_attention_mask=None,
        orig_input_ids=None,
        orig_attention_mask=None,
        orig_attr_labels=None,
        orig_labels=None):

        batch_size = len(orig_attr_labels)
        device = orig_input_ids.device
        gen_input_emb = torch.matmul(
            gen_input_ids, self.transformer.wte.weight)

        content_hidden_states = self.enc_model.transformer(
            inputs_embeds=gen_input_emb,
            attention_mask=gen_attention_mask,
        )[0]

        content_emb = avg_pool(content_hidden_states, gen_attention_mask)
        orig_style_emb = self.style_embed(orig_attr_labels).unsqueeze(1)

        prior_emb = orig_style_emb+content_emb.unsqueeze(1)
        orig_input_emb = self.transformer.wte(orig_input_ids)

        orig_input_emb = torch.cat([prior_emb, orig_input_emb], dim=1)
        orig_attention_mask = torch.cat(
            [torch.ones_like(orig_attention_mask[:,:1]),
            orig_attention_mask], dim=1)
        orig_position_ids = orig_attention_mask.cumsum(dim=1) - 1

        orig_outputs = self.transformer(
            attention_mask=orig_attention_mask,
            inputs_embeds=orig_input_emb,
            position_ids=orig_position_ids)
        orig_hidden_states=orig_outputs[0]

        lm_logits = self.lm_head(orig_hidden_states)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(
                lm_logits[..., :-1, :].contiguous().transpose(1, 2), orig_labels)
        z1 = F.normalize(self.project(content_emb), dim=-1)
        z1 = z1.view((-1, z1.size(-1)))

        z2 = avg_pool(orig_hidden_states, orig_attention_mask)
        z2 = F.normalize(self.project(z2), dim=-1)
        z2 = z2.view((-1, z2.size(-1)))
        print('gen2orig: {:0.5f}'.format(loss))
        #loss += contrast_loss
        return loss, z1, z2


    def original_to_generate(self,
        orig_input_ids=None,
        orig_attention_mask=None,
        orig_attr_labels=None,
        gen_attr_labels=None):
        batch_size = len(orig_attr_labels)
        #print(attention_mask.size())
        max_enc_len = torch.max(orig_attention_mask.sum(-1)).item()
        device = orig_input_ids.device
        content_hidden_states = self.enc_model.transformer(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
        )[0]
        content_emb = avg_pool(content_hidden_states, orig_attention_mask)

        gen_style_emb = self.style_embed(gen_attr_labels).unsqueeze(1)
        prior_emb = gen_style_emb+content_emb.unsqueeze(1)
        gen_log_probs = []
        gen_hidden_states = []

        gen_input_emb = prior_emb
        gen_attention_mask = torch.ones((batch_size,1)).long().to(device)
        gen_position_ids = gen_attention_mask.cumsum(dim=1) - 1
        prev_states = None
        for k in range(self.max_length):
            gen_outputs = self.transformer(
                inputs_embeds=gen_input_emb[:,k:k+1],
                attention_mask=gen_attention_mask[:,k:k+1],
                position_ids=gen_position_ids[:,k:k+1],
                use_cache=True,
                past_key_values=prev_states
            )
            gen_hidden_state = gen_outputs[0][:,-1, :].contiguous()
            gen_hidden_state = gen_hidden_state.unsqueeze(1)
            gen_lm_logit = self.lm_head(gen_hidden_state)
            gen_log_prob = F.log_softmax(gen_lm_logit, dim=-1)
            gen_log_probs.append(gen_log_prob)
            gen_hidden_states.append(gen_hidden_state)

            gen_input_emb = torch.cat([gen_input_emb,
                torch.matmul(
                gen_log_prob.exp(), self.transformer.wte.weight)],dim=1)
            gen_position_ids = torch.cat([gen_position_ids, (gen_position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
            gen_attention_mask = torch.cat([gen_attention_mask, gen_attention_mask.new_ones((batch_size, 1))], dim=1)
        gen_log_probs = torch.cat(gen_log_probs, 1)
        gen_hidden_states = torch.cat(gen_hidden_states, 1)
        z1 = F.normalize(self.project(content_emb), dim=-1)
        z1 = z1.view((-1, z1.size(-1)))

        gen_soft_tokens = gen_log_probs.exp()
        gen_lengths = self.get_lengths(gen_soft_tokens.argmax(-1))
        gen_max_len = torch.max(gen_lengths).item()
        position_ids = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1)).to(device)
        gen_attention_mask = (position_ids <= gen_lengths.unsqueeze(-1)).long()

        z2 = avg_pool(gen_hidden_states, gen_attention_mask)
        z2 = F.normalize(self.project(z2), dim=-1)
        z2 = z2.view((-1, z2.size(-1)))
        loss = self.contrastive_loss(z1, z2)
        return loss, gen_log_probs, gen_attention_mask, z1, z2

    def self_cyclic_reconstruct(self,
        input_ids=None,
        attention_mask=None,
        attr_labels=None,
        labels=None):

        batch_size = len(attr_labels)
        device = input_ids.device
        rev_attr_labels = 1- attr_labels

        self_loss, _ = self.self_reconstruct(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attr_labels=attr_labels,
                labels=labels)

        orig2gen_loss, gen_log_probs, gen_attention_mask, z1, z2 = self.original_to_generate(
            orig_input_ids=input_ids,
            orig_attention_mask=attention_mask,
            orig_attr_labels=attr_labels,
            gen_attr_labels=rev_attr_labels
        )
        gen_soft_tokens = gen_log_probs.exp()

        gen2orig_loss, z3, z4 = self.generate_to_original(
            gen_input_ids=gen_soft_tokens,
            gen_attention_mask=gen_attention_mask,
            gen_attr_labels=rev_attr_labels,
            orig_input_ids=input_ids,
            orig_attention_mask=attention_mask,
            orig_attr_labels=attr_labels,
            orig_labels=labels)
        total_loss = gen2orig_loss + self_loss #orig2gen_loss + self_loss
        contrastive_loss = self.contrastive_loss(z1, z4, z2) #+ self.contrastive_loss(z3, z2, z4)
        print('contrastive: {:0.5f}'.format(contrastive_loss.item()))
        return total_loss + contrastive_loss
