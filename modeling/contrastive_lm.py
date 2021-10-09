<<<<<<< Updated upstream
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


class ContrastiveGPT2(GPT2LMHeadModel):
    """Contrastive based GPT2"""

    def __init__(self, config):
        super().__init__(config)

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #n_gpu = torch.cuda.device_count()

        self.pooler = Pooler('avg')
        self.mlp = MLPLayer(config)

        self.sim = Similarity(temp=0.07)
        self.attr_embedding = nn.Embedding(2, config.hidden_size)


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
        import pdb
        pdb.set_trace()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def self_contrastive_loss(self, z1, z2):
        loss_fct = nn.CrossEntropyLoss()
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        cos_loss = loss_fct(cos_sim, cos_labels)
        return cos_loss


    def forward(self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            labels=None,
            attr_labels=None,
            use_cache=False):
        batch_size = len(attr_labels)
        #print(attention_mask.size())
        if len(attr_labels.size()) == 1:
            attr_labels = attr_labels.unsqueeze(-1)

        if labels is not None:
            input_ids = torch.stack([input_ids, input_ids],dim=1)
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)

            attention_mask = torch.stack([attention_mask, attention_mask],dim=1)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)

            labels = torch.stack([labels, labels],dim=1)
            labels = labels.view((-1, labels.size(-1))) # (bs * num_sent len)

            attr_labels = torch.stack([attr_labels, attr_labels],dim=1)
            attr_labels = attr_labels.view((-1, attr_labels.size(-1))) # (bs * num_sent len)
        attr_embed = self.attr_embedding(attr_labels)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            encoder_hidden_states=attr_embed,
            encoder_attention_mask=torch.ones_like(attr_labels)
        )

        hidden_states = transformer_outputs[0]
        pooler_output = self.pooler(attention_mask, transformer_outputs)
        mlp_output = F.normalize(self.mlp(pooler_output), dim=-1)
        mlp_output = mlp_output.view((batch_size, -1, mlp_output.size(-1))) # (bs, num_sent, hidden)

        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            z1 = mlp_output[:, 0]
            z2 = mlp_output[:, 1]

            if self.supervised:
                if dist.is_initialized() and self.training:
                    z1, z2, attr_labels = self.gather(z1, z2, attr_labels)
                cos_loss = self.sup_contrastive_loss(z1, z2, attr_labels)
            else:
                if dist.is_initialized() and self.training:
                    z1, z2 = self.gather(z1, z2)
                cos_loss = self.self_contrastive_loss(z1, z2)
            loss_fct = nn.CrossEntropyLoss()

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss =  cos_loss + rec_loss
            outputs = (loss,) + outputs

            #print("cos:",cos_loss)
            #print(input_ids)
            #print(attr_ids)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
=======
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
from transformers import GPT2Tokenizer, BertTokenizer

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


class ContrastiveGPT2(GPT2LMHeadModel):
    """Contrastive based GPT2"""

    def __init__(self, config):
        super().__init__(config)

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #n_gpu = torch.cuda.device_count()
        self.ablation = config.ablation

        self.pooler = Pooler('avg')
        self.mlp = MLPLayer(config)

        self.sim = Similarity(temp=0.07)
        self.attr_embedding = nn.Embedding(2, config.hidden_size)

        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel(self.bert_config, add_pooling_layer=False)
        self.bert_pooler = Pooler('cls')
        self.bert_mlp = MLPLayer(self.bert_config)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.freeze_bert(10)

        self.supervised = config.supervised

    def freeze_bert(self, num_layer):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:num_layer]:
            for param in layer.parameters():
                param.requires_grad = False

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

    def self_contrastive_loss(self, z1, z2):
        loss_fct = nn.CrossEntropyLoss()
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        cos_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
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
        if self.ablation == 'cont_gpt' or self.ablation == 'cont_gpt_last':
            #print(attention_mask.size())

            if len(attr_labels.size()) == 1:
                attr_labels = attr_labels.unsqueeze(-1)

        if labels is not None:
            if self.ablation == 'cont_bert_gpt':
                attr_ids = torch.stack([attr_ids, attr_ids], dim=1)
                attr_ids = attr_ids.view((-1, attr_ids.size(-1)))

            input_ids = torch.stack([input_ids, input_ids],dim=1)
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)

            attention_mask = torch.stack([attention_mask, attention_mask],dim=1)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)

            labels = torch.stack([labels, labels],dim=1)
            labels = labels.view((-1, labels.size(-1))) # (bs * num_sent len)

            if self.ablation == 'cont_bert_gpt' and len(attr_labels.size()) == 1:
                attr_labels = attr_labels.unsqueeze(-1)

            if self.ablation == 'cont_gpt' or self.ablation == 'cont_gpt_last':
                attr_labels = torch.stack([attr_labels, attr_labels],dim=1)
                attr_labels = attr_labels.view((-1, attr_labels.size(-1))) # (bs * num_sent len)

        if self.ablation == 'cont_gpt_last':
            attr_embed = self.attr_embedding(attr_labels)

        if self.ablation == 'cont_bert_gpt':
            attr_attention_mask = (attr_ids != self.bert_tokenizer.pad_token_id).long()

            attr_outputs = self.bert(
                attr_ids,
                attention_mask=attr_attention_mask,
                output_hidden_states=False,
            )  # CLS

            attr_pooler_output = self.bert_pooler(attr_attention_mask, attr_outputs)
            bert_mlp_output = F.normalize(self.bert_mlp(attr_pooler_output), dim=-1)
            bert_mlp_output = bert_mlp_output.view((batch_size, -1, bert_mlp_output.size(-1)))

            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
             )
        if self.ablation == 'cont_gpt' or self.ablation == 'cont_gpt_last':
            #attribute vector encoding
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                encoder_attention_mask=torch.ones_like(attr_labels)
            )

        if self.ablation == 'cont_gpt_last':
            hidden_states = transformer_outputs[0] + attr_embed
        else:
            hidden_states = transformer_outputs[0]
        pooler_output = self.pooler(attention_mask, transformer_outputs)
        mlp_output = F.normalize(self.mlp(pooler_output), dim=-1)
        mlp_output = mlp_output.view((batch_size, -1, mlp_output.size(-1))) # (bs, num_sent, hidden)

        if self.ablation == 'cont_gpt' or self.ablation == 'cont_gpt_last':
            lm_logits = self.lm_head(hidden_states)
        if self.ablation == 'cont_bert_gpt':
            lm_logits = self.lm_head(hidden_states + attr_pooler_output.unsqueeze(1))
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            if self.ablation == 'cont_gpt' or self.ablation == 'cont_gpt_last':
                z1 = mlp_output[:, 0]
                z2 = mlp_output[:, 1]
            if self.ablation == 'cont_bert_gpt':
                z11, z21 = bert_mlp_output[:, 0], bert_mlp_output[:, 1]
                z12, z22 = mlp_output[:, 0], mlp_output[:, 1]
                z1 = z11 + z12
                z2 = z21 + z22


            if self.supervised:
                if dist.is_initialized() and self.training:
                    z1, z2, attr_labels = self.gather(z1, z2, attr_labels)
                cos_loss = self.sup_contrastive_loss(z1, z2, attr_labels)
            else:
                if dist.is_initialized() and self.training:
                    z1, z2 = self.gather(z1, z2)
                cos_loss = self.self_contrastive_loss(z1, z2)
            loss_fct = nn.CrossEntropyLoss()

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss =  cos_loss + rec_loss
            outputs = (loss,) + outputs

            #print("cos:",cos_loss)
            #print(input_ids)
            #print(attr_ids)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
>>>>>>> Stashed changes
