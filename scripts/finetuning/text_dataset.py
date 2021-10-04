"""
TextDataset class modified to include parameter for controlling the size of the dataset, in tokens
https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py
"""

import os
import pickle
import time
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        data_size: int=None,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info(f'Creating TextDataset from {file_path} with length (in number of tokens): {data_size})')

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                tokenized_text = tokenized_text[:data_size]     # truncate to the first data_size tokens

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class TextAttrDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_paths,
        block_size: int,
        data_size: int=None,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        self.examples = []
        self.attr_labels = []
        self.lens = []
        self.file2attr = {}
        self.attr = 0
        for file_path in file_paths:
            assert os.path.isfile(file_path), f"Input file path {file_path} not found"
            logger.info(f'Creating TextDataset from {file_path} with length (in number of tokens): {data_size})')

            block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

            directory, filename = os.path.split(file_path)
            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else directory,
                "cached_lm_{}_{}_{}".format(
                    tokenizer.__class__.__name__,
                    str(block_size),
                    filename,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        temp_examples, temp_lens, temp_attr_labels = pickle.load(handle)
                        self.examples += temp_examples
                        self.lens += temp_lens
                        self.attr_labels += temp_attr_labels
                        self.attr = list(set(temp_attr_labels))[0]
                        self.file2attr[file_path] = self.attr

                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )

                else:
                    logger.info(f"Creating features from dataset file at {directory}")
                    self.file2attr[file_path] = self.attr
                    with open(file_path, encoding="utf-8") as f:
                        text = f.read()
                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                    tokenized_text = tokenized_text[:data_size]     # truncate to the first data_size tokens

                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                        self.examples.append(
                            tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                        )
                        self.lens.append(len(tokenized_text[i : i + block_size]))
                        self.attr_labels.append(self.attr)
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should loook for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump((self.examples, self.lens, self.attr_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )
                    self.attr += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return (torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.lens[i], dtype=torch.long),
            torch.tensor(self.attr_labels[i], dtype=torch.long))
