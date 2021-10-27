import sys, os
sys.path.append(os.path.abspath('generation'))
sys.path.append(os.path.abspath('utils'))
sys.path.append(os.path.abspath('modeling'))
from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel, BertTokenizer
from transformers.generation_utils import top_k_top_p_filtering
from utils_fn import set_seed
from base_config import Config
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    TrainingArguments,
)
from style_transfer import StyleGPT2

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def avg_pool(hidden, attention_mask):
    return ((hidden * attention_mask.unsqueeze(-1)).sum(1) / ( attention_mask.sum(-1).unsqueeze(-1)+1e-8))

class StyleGPT2Generation:
    STOP_TOKEN = "<|endoftext|>"

    def __init__(self, model: Union[str, Path, GPT2PreTrainedModel] = 'gpt2',
                tokenizer: str = 'gpt2', seed: int = 42, local_rank=-1, gen_type='style-gpt2-none'):
        # Set up device
        # Set up device
        self.gen_type=gen_type
        print(local_rank)
        if local_rank == -1:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu")

        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)

        # Set up model
        if isinstance(model, Path) or isinstance(model, str):
            config = AutoConfig.from_pretrained(str(model))
            config.num_styles = 2
            config.max_length = 64
            config.hard_negative_weight = 0
            config.contrastive_factor = 0
            self.tokenizer = GPT2Tokenizer.from_pretrained(str(model), pad_token=self.STOP_TOKEN)

            model = StyleGPT2.from_pretrained(
            str(model),
            from_tf=bool(".ckpt" in str(model)),
            config=config)

        self.model = model.to(self.device)


        # Set up tokenizer
        # IMPORTANT: Note that setting the pad token like this in the constructor gives the pad_token the
        # pad_token_id = 50256, which normally belongs to the <EOS> token_id in GPT2. This is a very ugly
        # way that works at the moment of setting the pad_token_id to the <EOS> token that is already
        # included in the vocab size.
        print()
        #assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<GPT2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


    def generate_all(self,
                 prompt: Union[str, List[str]],
                 attr,
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 use_past: bool=False,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape
        attr_labels = attr.to(self.device)
        attr_labels = attr_labels.view(-1,1)
        pos_attr = self.model.style_embed(attr_labels)
        neg_attr = self.model.style_embed(1-attr_labels)

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):

                dec_outputs = self.model.transformer(
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    position_ids=position_ids
                )

                dec_hidden_state = dec_outputs[0]
                try:
                    if attention_mask is not None:
                        block_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
                        block_attention_mask = block_attention_mask[:, None, None, :]
                        block_attention_mask = block_attention_mask.to(dtype=self.dtype)
                        block_attention_mask = (1.0 - block_attention_mask) * -10000.0
                    pos_hidden_state = self.block_layer(
                        dec_hidden_state+pos_attr,
                        attention_mask=block_attention_mask,
                    )[0]
                    neg_hidden_state = self.block_layer(
                        dec_hidden_state+neg_attr,
                        attention_mask=block_attention_mask,
                    )[0]
                    pos_logits = self.model.lm_head(pos_hidden_state)
                    neg_logits = self.model.lm_head(neg_hidden_state)

                except:
                    pos_logits = self.model.lm_head(dec_hidden_state+pos_attr)
                    neg_logits = self.model.lm_head(dec_hidden_state+neg_attr)
                #logits = top_k_top_p_filtering(pos_logits, top_p=p)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_pos_token_logits = pos_logits[range(batch_size), last_non_masked_idx, :]
                    next_neg_token_logits = neg_logits[range(batch_size), last_non_masked_idx, :]

                    dec_hidden_states = dec_hidden_state

                else:
                    next_pos_token_logits = pos_logits[:, -1, :]
                    next_neg_token_logits = neg_logits[:, -1, :]

                    dec_hidden_states = torch.cat([dec_hidden_states, dec_hidden_state[:,-1,:].unsqueeze(1)], dim=1)

                if sample:

                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_pos_token_logits = next_pos_token_logits / temperature
                        next_neg_token_logits = next_neg_token_logits / temperature
                    #print(next_pos_token_logits)
                    #print(next_neg_token_logits)
                    # Top-p/top-k filtering
                    #next_token_logits = next_pos_token_logits
                    next_token_logits = 15*next_pos_token_logits - next_neg_token_logits
                    #print(next_token_logits)

                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    #print(next_token_logits)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    #print(probs)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs


    def generate_one(self,
                 prompt: Union[str, List[str]],
                 attr,
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 use_past: bool=False,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape
        attr_labels = attr.to(self.device)
        attr_labels = attr_labels.view(-1,1)
        pos_attr = self.model.style_embed(attr_labels)
        neg_attr = self.model.style_embed(1-attr_labels)

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        past = None
        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):
                if use_past:
                    if step == 0:
                        dec_outputs = self.model.transformer(
                            attention_mask=attention_mask,
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=use_past,
                            past=past,
                        )
                    else:
                        dec_outputs = self.model.transformer(
                            attention_mask=attention_mask[:,step].unsqueeze(1),
                            input_ids=input_ids[:,step].unsqueeze(1),
                            position_ids=position_ids[:,step].unsqueeze(1),
                            use_cache=use_past,
                            past=past,
                        )

                    dec_hidden_state = dec_outputs[0]
                    past = dec_outputs[1]
                else:
                    dec_outputs = self.model.transformer(
                        attention_mask=attention_mask,
                        input_ids=input_ids,
                        position_ids=position_ids
                    )

                    dec_hidden_state = dec_outputs[0]

                try:
                    if attention_mask is not None:
                        block_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
                        block_attention_mask = block_attention_mask[:, None, None, :]
                        block_attention_mask = block_attention_mask.to(dtype=self.dtype)
                        block_attention_mask = (1.0 - block_attention_mask) * -10000.0
                    pos_hidden_state = self.block_layer(
                        dec_hidden_state+pos_attr,
                        attention_mask=block_attention_mask,
                    )[0]
                    neg_hidden_state = self.block_layer(
                        dec_hidden_state+neg_attr,
                        attention_mask=block_attention_mask,
                    )[0]
                    pos_logits = self.model.lm_head(pos_hidden_state)
                    neg_logits = self.model.lm_head(neg_hidden_state)

                except:
                    pos_logits = self.model.lm_head(dec_hidden_state+pos_attr)
                    neg_logits = self.model.lm_head(dec_hidden_state+neg_attr)
                #logits = top_k_top_p_filtering(pos_logits, top_p=p)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_pos_token_logits = pos_logits[range(batch_size), last_non_masked_idx, :]
                    next_neg_token_logits = neg_logits[range(batch_size), last_non_masked_idx, :]

                    dec_hidden_states = dec_hidden_state

                else:
                    next_pos_token_logits = pos_logits[:, -1, :]
                    next_neg_token_logits = neg_logits[:, -1, :]

                    dec_hidden_states = torch.cat([dec_hidden_states, dec_hidden_state[:,-1,:].unsqueeze(1)], dim=1)

                if sample:

                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_pos_token_logits = next_pos_token_logits / temperature
                        next_neg_token_logits = next_neg_token_logits / temperature
                    #print(next_pos_token_logits)
                    #print(next_neg_token_logits)
                    # Top-p/top-k filtering
                    #next_token_logits = next_pos_token_logits
                    next_token_logits = next_pos_token_logits
                    #print(next_token_logits)

                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    #print(next_token_logits)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    #print(probs)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs


    def generate_attr(self,
                 prompt: Union[str, List[str]],
                 attr,
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 use_past: bool=False,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')
        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape
        attr_labels = attr.to(self.device)
        attr_labels = attr_labels.view(-1,1)
        style_emb = self.model.style_embed(attr_labels)
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):

                if use_past:
                    input_emb = self.model.transformer.wte(input_ids)
                    input_emb = torch.cat([style_emb, input_emb], dim=1)
                    if step == 0:
                        past=None
                        attention_mask = torch.cat(
                            [torch.ones_like(attention_mask[:,:1]),
                            attention_mask], dim=1)
                        position_ids = attention_mask.cumsum(dim=1) - 1
                        dec_outputs = self.model.transformer(
                            attention_mask=attention_mask,
                            inputs_embeds=input_emb,
                            position_ids=position_ids,
                            use_cache=use_past,
                            past_key_values=past,
                        )
                    else:
                        dec_outputs = self.model.transformer(
                            attention_mask=attention_mask[:,-1].unsqueeze(1),
                            inputs_embeds=input_emb[:,-1].unsqueeze(1),
                            position_ids=position_ids[:,-1].unsqueeze(1),
                            use_cache=use_past,
                            past_key_values=past,
                        )
                    dec_hidden_state = dec_outputs[0]
                    past = dec_outputs[1]
                else:
                    input_emb = self.model.transformer.wte(input_ids)
                    input_emb = torch.cat([style_emb, input_emb], dim=1)
                    if step == 0:
                        attention_mask = torch.cat(
                            [torch.ones_like(attention_mask[:,:1]),
                            attention_mask], dim=1)
                        position_ids = attention_mask.cumsum(dim=1) - 1

                    dec_outputs = self.model.transformer(
                        attention_mask=attention_mask,
                        inputs_embeds=input_emb,
                        position_ids=position_ids
                    )
                    dec_hidden_state = dec_outputs[0]
                logits = self.model.lm_head(dec_hidden_state)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = logits[range(batch_size), last_non_masked_idx, :]

                else:
                    next_token_logits = logits[:, -1, :]

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
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len+1:]]
        return decoded_outputs





    def generate(self,
                 prompt: Union[str, List[str]],
                 attr,
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 **model_kwargs) -> List[str]:

        #if self.gen_type == 'style-gpt2-none':
        #    return self.generate_none(prompt, attr, max_len, sample, k, p, temperature, **model_kwargs)
        if self.gen_type == 'style-gpt2-all':
            return self.generate_all(prompt, attr, max_len, sample, k, p, temperature, **model_kwargs)
        elif self.gen_type == 'style-gpt2-one':
            return self.generate_one(prompt, attr, max_len, sample, k, p, temperature, **model_kwargs)
        elif self.gen_type == 'style-gpt2-attr':
            return self.generate_attr(prompt, attr, max_len, sample, k, p, temperature, **model_kwargs)
