from pathlib import Path
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, BertModel, BertConfig
from transformers.generation_utils import top_k_top_p_filtering
import time
import random
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens

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

# TODO: convert to HuggingFace pipeline
class ContBERTGPT2Generation(nn.Module):

    def __init__(self, config, gpt2_tokenizer, bert_tokenizer, supervised=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel(self.bert_config, add_pooling_layer=False)
        self.bert_pooler = Pooler('cls')
        self.bert_mlp = MLPLayer(self.bert_config)

        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2LMHeadModel(self.gpt2_config)
        self.gpt2_pooler = Pooler('avg')
        self.gpt2_mlp = MLPLayer(self.gpt2_config)

        self.gpt2_tokenizer = gpt2_tokenizer
        self.bert_tokenizer = bert_tokenizer

        self.sim = Similarity(temp=0.07)

        self.freeze_bert(6)
        self.supervised = supervised

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


    def contrastive_loss(self, z1, z2, labels=None, mask=None, temperature=0.07, contrast_mode='all'):
        features = torch.stack([z1,z2], dim=1)

        batch_size, n_views, _ = features.size()

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        # similarity compute : (BN, D) X (D, BN) -> (BN, BN)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability: (BN, BN)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # all -> N, one -> 1
        # mask : (B, B) -> (BN, BN)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # logits_mask : (BN, BN)
        # assigning zeros to indicating itself
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )

        # same class to 1 and different classes to 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss



    def forward(self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            labels=None,
            attr_ids=None,
            attr_labels=None,
            use_cache=False):
        batch_size = len(attr_ids)


        if labels is not None:

            attr_ids = torch.stack([attr_ids, attr_ids],dim=1)
            attr_ids = attr_ids.view((-1, attr_ids.size(-1)))

            input_ids = torch.stack([input_ids, input_ids],dim=1)
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)

            attention_mask = torch.stack([attention_mask, attention_mask],dim=1)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)

            labels = torch.stack([labels, labels],dim=1)
            labels = labels.view((-1, labels.size(-1))) # (bs * num_sent len)

            if len(attr_labels.size()) == 1:
                attr_labels = attr_labels.unsqueeze(-1)

            #attr_labels = torch.stack([attr_labels, attr_labels],dim=1)
            #attr_labels = labels.view((-1, attr_labels.size(-1))) # (bs * num_sent len)

        attr_attention_mask = (attr_ids != self.bert_tokenizer.pad_token_id).long()

        attr_outputs = self.bert(
            attr_ids,
            attention_mask=attr_attention_mask,
            output_hidden_states=False,
        ) # CLS
        attr_pooler_output = self.bert_pooler(attr_attention_mask, attr_outputs)
        bert_mlp_output = F.normalize(self.bert_mlp(attr_pooler_output), dim=-1)
        bert_mlp_output = bert_mlp_output.view((batch_size, -1, bert_mlp_output.size(-1))) # (b
        transformer_outputs = self.gpt2.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        hidden_states = transformer_outputs[0]
        transformer_pooler_output = self.gpt2_pooler(attention_mask, transformer_outputs)
        transformer_mlp_output = F.normalize(self.gpt2_mlp(transformer_pooler_output), dim=-1)
        transformer_mlp_output = transformer_mlp_output.view((batch_size, -1, transformer_mlp_output.size(-1))) # (bs, num_sent, hidden)
        if labels is not None:
            z11, z21 = bert_mlp_output[:,0], bert_mlp_output[:,1]
            z12, z22 = transformer_mlp_output[:,0], transformer_mlp_output[:,1]
            z1 = z11 + z12
            z2 = z21 + z22


        lm_logits = self.gpt2.lm_head(hidden_states+attr_pooler_output.unsqueeze(1))
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:


            if self.supervised:
                if dist.is_initialized() and self.training:
                    z1, z2, attr_labels = self.gather(z1, z2, attr_labels)
                cos_loss = self.contrastive_loss(z1, z2, attr_labels)
            else:
                if dist.is_initialized() and self.training:
                    z1, z2 = self.gather(z1, z2)
                cos_loss = self.contrastive_loss(z1, z2)

            # Shift so that tokens < n predict n
            loss_fct = nn.CrossEntropyLoss()

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            rec_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss =  cos_loss + rec_loss
            outputs = (loss,) + outputs
            if torch.isnan(cos_loss):
                import pdb
                pdb.set_trace()
            #print("cos:",cos_loss)
            #print(input_ids)
            #print(attr_ids)
            print(cos_loss, rec_loss)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def convert_str_to_ids(self, text, tokenizer, keep_tokens=1.0):
        if isinstance(text, str):
            text = [text]
        batch_ids = []
        for t_i, t in enumerate(text):

            t = t.strip()

            t_tokens = tokenizer.tokenize(t)
            if keep_tokens == 1:
                num_keep_tokens = len(t_tokens)
            else:
                num_keep_tokens = int(len(t_tokens)*keep_tokens)
                if num_keep_tokens < 1:
                    num_keep_tokens = int(len(t_tokens)*0.5) + 1
            t_ids = torch.tensor(
                    tokenizer.convert_tokens_to_ids(t_tokens[:num_keep_tokens]), dtype=torch.long)


            batch_ids.append(t_ids)
        batch_ids = pad_sequence(batch_ids, padding_value=tokenizer.pad_token_id, batch_first=True)
        batch_attention_mask = (batch_ids != tokenizer.pad_token_id).long()
        batch_position_ids = batch_attention_mask.cumsum(dim=1) - 1
        return batch_ids, batch_attention_mask, batch_position_ids

    def generate(self,
                 prompt: Union[str, List[str], torch.LongTensor],
                 attr_text: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 0.9,
                 temperature: float = 1.0,
                 bad_words_ids: List[List[int]] = None,
                 keep_tokens=1.0) -> List[str]:
        input_ids, attention_mask, position_ids = self.convert_str_to_ids(prompt, self.gpt2_tokenizer,keep_tokens=keep_tokens)
        attr_ids, _, _ = self.convert_str_to_ids(attr_text, self.bert_tokenizer, keep_tokens=1.0)
        input_ids, attention_mask, position_ids, attr_ids =\
        [var.to(self.device) for var in [input_ids, attention_mask, position_ids, attr_ids]]

        batch_size, prompt_len = input_ids.shape
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        past_key_values = None

        for step in range(max_len):
            attr_attention_mask = (attr_ids != self.bert_tokenizer.pad_token_id).long()

            attr_outputs = self.bert(
                attr_ids,
                attention_mask=attr_attention_mask,
                output_hidden_states=False,
            )
            attr_pooler_output = self.bert_pooler(attr_attention_mask, attr_outputs)
            transformer_outputs = self.gpt2.transformer(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = transformer_outputs[0]
            logits = self.gpt2.lm_head(hidden_states+attr_pooler_output.unsqueeze(1))

            #logits = outputs[0] #logits
            # in the first decoding step, we want to use the 'real' last position for each sentence
            if step == 0:
                last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                next_token_logits = logits[range(batch_size), last_non_masked_idx, :]
            else:
                next_token_logits = logits[:, -1, :]

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                # TODO: use a vectorized operation
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # either append a padding token here if <EOS> has been seen or append next token
            tokens_to_add = next_tokens * unfinished_sents + self.gpt2_tokenizer.pad_token_id * (1 - unfinished_sents)

            # this updates which sentences have not seen an EOS token so far
            # if one EOS token was seen the sentence is finished
            eos_in_sents = tokens_to_add == self.gpt2_tokenizer.eos_token_id
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is an EOS in each sentence
            if unfinished_sents.max() == 0:
                break

            # Update input_ids, attention_mask and position_ids
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)

        decoded_outputs = [self.gpt2_tokenizer.decode(_input[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True) for _input in input_ids]
        #print([self.tokenizer.decode(_input, clean_up_tokenization_spaces=True) for _input in input_ids])
        if keep_tokens == 1:
            return decoded_outputs
        else:
            decoded_inital = [self.gpt2_tokenizer.decode(_input[:prompt_len], skip_special_tokens=True, clean_up_tokenization_spaces=True) for _input in input_ids]
            return decoded_outputs, decoded_inital
