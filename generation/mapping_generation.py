from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2PreTrainedModel
from transformers import BertTokenizer
from transformers.generation_utils import top_k_top_p_filtering
import sys, os

sys.path.append(os.path.abspath('modeling'))
from mapping_lm import MappingGPT2
from utils_fn import set_seed

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


class MappingGeneration:
    STOP_TOKEN = "<|endoftext|>"

    def __init__(self, config, approach: str = 'cont_gpt', model: Union[str, Path] = 'gpt2',
                 past_key_values: bool = False, end_token_prediction=False, logit=False, softmax_logit=False,
                 logit_type=0, mixup_loss=False, alpha=1.0,
                 tokenizer: str = 'gpt2', seed: int = 42, supervised=False, local_rank=-1):
        self.past_key_values = past_key_values

        # logit control
        self.logit = logit
        self.softmax_logit = softmax_logit
        self.logit_type = logit_type
        if (not self.logit) and self.softmax_logit:
            assert ('wrong arguments')
        # Set up device
        if local_rank == -1:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:{}".format(local_rank) if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        set_seed(seed, n_gpu)

        self.approach = approach
        # Set up model
        self.alpha = alpha
        config.mixup_loss = mixup_loss

        if isinstance(model, Path) or isinstance(model, str):
            model = MappingGPT2.from_pretrained(
                str(model),
                from_tf=bool(".ckpt" in str(model)),
                config=config,
                cache_dir=config.cache_dir,
            )
        else:
            assert ()

        self.model = model.to(self.device)

        # Set up tokenizer
        # IMPORTANT: Note that setting the pad token like this in the constructor gives the pad_token the
        # pad_token_id = 50256, which normally belongs to the <EOS> token_id in GPT2. This is a very ugly
        # way that works at the moment of setting the pad_token_id to the <EOS> token that is already
        # included in the vocab size.
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token=self.STOP_TOKEN)

        # self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token='<|pad|>')
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.supervised = supervised

    def __repr__(self):
        return f'<GPT2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 **model_kwargs) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.model.eval()
        past = None
        attr_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if self.logit:
            neg_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for step in range(max_len):
                if self.past_key_values:
                    if step > 0:
                        outputs = self.model(input_ids[:, -1].unsqueeze(-1),
                                             attention_mask=attention_mask[:, -1].unsqueeze(-1),
                                             position_ids=position_ids[:, -1].unsqueeze(-1), past_key_values=past,
                                             use_cache=True, gen_approach=self.approach,
                                             attr_labels=attr_labels,
                                             **model_kwargs)

                    else:
                        outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                             past_key_values=past, use_cache=True, gen_approach=self.approach,
                                             attr_labels=attr_labels,
                                             **model_kwargs)
                    if self.logit:
                        if step > 0:
                            neg_outputs = self.model(input_ids[:, -1].unsqueeze(-1),
                                                     attention_mask=attention_mask[:, -1].unsqueeze(-1),
                                                     position_ids=position_ids[:, -1].unsqueeze(-1),
                                                     past_key_values=past, use_cache=True, gen_approach=self.approach,
                                                     attr_labels=neg_labels, **model_kwargs)

                        else:
                            neg_outputs = self.model(input_ids, attention_mask=attention_mask,
                                                     position_ids=position_ids, past_key_values=past, use_cache=True,
                                                     gen_approach=self.approach,
                                                     attr_labels=neg_labels, **model_kwargs)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                         use_cache=True, gen_approach=self.approach,
                                         attr_labels=attr_labels,
                                         **model_kwargs)
                    if self.logit:
                        neg_outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                                 use_cache=True, gen_approach=self.approach,
                                                 attr_labels=neg_labels, **model_kwargs)
                logits, past = outputs[0], outputs[1]
                if self.logit:
                    neg_logits = neg_outputs[0]
                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    if self.logit:
                        pos_token_logits = logits[range(batch_size), last_non_masked_idx, :]
                        neg_token_logits = neg_logits[range(batch_size), last_non_masked_idx, :]
                    else:
                        next_token_logits = logits[range(batch_size), last_non_masked_idx, :]
                else:
                    if self.logit:
                        pos_token_logits = logits[:, -1, :]
                        neg_token_logits = neg_logits[:, -1, :]
                    else:
                        next_token_logits = logits[:, -1, :]

                if self.logit:
                    if self.softmax_logit:
                        pos_token_logits = F.softmax(pos_token_logits, dim=-1)
                        neg_token_logits = F.softmax(neg_token_logits, dim=-1)

                    if self.logit_type == 0:  # subtract positive - negative
                        next_token_logits = pos_token_logits - neg_token_logits
                    elif self.logit_type == 1:  # positive - 0.5*negative
                        next_token_logits = pos_token_logits - 0.5 * neg_token_logits
                    elif self.logit_type == 2:
                        masked_logit = (pos_token_logits - neg_token_logits) < 0
                        next_token_logits = pos_token_logits + 1.5 * masked_logit * (
                                    pos_token_logits - neg_token_logits)
                    elif self.logit_type == 3:  # subtract only negative
                        masked_logit = (pos_token_logits - neg_token_logits) < 0
                        next_token_logits = pos_token_logits + masked_logit * (pos_token_logits - neg_token_logits)
                    elif self.logit_type == 4:
                        masked_logit = (pos_token_logits - neg_token_logits) < 0
                        next_token_logits = pos_token_logits + self.alpha * masked_logit * (
                                    pos_token_logits - neg_token_logits)

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
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs
