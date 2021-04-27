import os
import sys
import math
from random import randint, shuffle, uniform
from random import random as rand
from random import sample as sample_func

from numpy import array
from numpy.random import choice

import torch

from biunilm.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_src_list, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, preprocess=[],
                 n_dial=-1, n_text=-1, accept_dtypes=[]):
        super(MyDataset).__init__()
        self.tokenizer = tokenizer  # tokenize function

        print("### I set minimum source length to 4.")
        self.min_src_len = 4  # TODO: !!!

        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        assert isinstance(preprocess, list)
        assert len(preprocess) == 1
        self.preprocess = preprocess
        self.n_condition = len(self.preprocess[0].c_indexer)

        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        assert file_oracle is None

        assert len(accept_dtypes) > 0

        # read the file into memory
        self.is_pretrain = True
        dial = []
        non_text = []
        c_text = []

        assert isinstance(file_src_list, list)
        for file_src in file_src_list:
            with open(file_src, "r", encoding='utf-8') as f:
                for index, line in enumerate(f):
                    if index % 500000 == 0:
                        print('Preprocess the {:}th line...'.format(index))
                        sys.stdout.flush()

                    src, cond, tgt, data_type = line.strip('\n').split('\t')[:4]
                    src_tk = tokenizer.tokenize(src.strip())
                    tgt_tk = tokenizer.tokenize(tgt.strip())
                    cond = cond.strip()

                    if len(src_tk) < self.min_src_len:
                        src_tk = []  # TODO: !!!

                    sample = (src_tk, tgt_tk, cond, data_type)

                    if len(tgt_tk) > 0 and len(cond) > 0:
                        if data_type in accept_dtypes:
                            if data_type == 'dial':
                                if len(src_tk):
                                    dial.append(sample)
                            elif data_type == 'mono':
                                if cond == '<nan>':
                                    non_text.append(sample)
                                else:
                                    c_text.append(sample)
                            else:
                                raise ValueError

        if 0 <= n_dial < len(dial):
            dial = sample_func(dial, n_dial)

        if 0 <= n_text < len(c_text):
            c_text = sample_func(c_text, n_text)

        print('Load {:} labeled dial samples.'.format(len(dial)))
        print('Load {:} labeled text samples.'.format(len(c_text)))
        print('Load {:} <nan> text samples.'.format(len(non_text)))

        if len(non_text):
            raise NotImplementedError  # I have not checked it.

        self.n_samples = len(dial) + len(c_text) + len(non_text)
        self.n_dial_samples = len(dial)  # 0215
        self.ex_list = [dial, c_text, non_text]

        self.index_map = {}
        index = 0
        for idx, _ in enumerate(dial):
            assert index not in self.index_map.keys()
            self.index_map[index] = (0, idx)
            index += 1

        for idx, _ in enumerate(c_text):
            assert index not in self.index_map.keys()
            self.index_map[index] = (1, idx)
            index += 1

        for idx, _ in enumerate(non_text):
            assert index not in self.index_map.keys()
            self.index_map[index] = (2, idx)
            index += 1

        assert list(self.index_map.keys()) == list(range(self.n_samples))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data_type, idx = self.index_map[index]
        instance = self.preprocess[0](self.ex_list[data_type][idx])
        return instance


class MySampler(torch.utils.data.Sampler):
    def __init__(self, my_dataset, batch_size, n_gpu, n_ctext=-1, equal_sample=False):
        assert isinstance(my_dataset, MyDataset)
        assert batch_size % n_gpu == 0

        self.batch_size = batch_size
        self.n_gpu = n_gpu
        self.batch_size_gpu = int(self.batch_size / self.n_gpu)

        self.n_samples = my_dataset.n_samples

        self.dial_index = []
        self.ctext_index = []
        self.non_index = []
        for index, p in my_dataset.index_map.items():
            if p[0] == 0:
                self.dial_index.append(index)
            elif p[0] == 1:
                self.ctext_index.append(index)
            elif p[0] == 2:
                self.non_index.append(index)
            else:
                raise ValueError

        print("### Train Set: dial {:}, ctext {:}, non-text {:}".format(len(self.dial_index),
                                                                        len(self.ctext_index),
                                                                        len(self.non_index)))

        if n_ctext > 0:
            self.n_ctext = min(n_ctext, self.batch_size_gpu)
            self.n_non = 0
            self.n_dial = 0

        else:
            self.n_non = 0
            if equal_sample:
                self.n_ctext = round(self.batch_size_gpu * 1 / 2) if len(self.ctext_index) else 0
            else:
                self.n_ctext = round(self.batch_size_gpu * 1 / 4) if len(self.ctext_index) else 0

            self.n_dial = self.batch_size_gpu - self.n_ctext - self.n_non

        print("### Sampler: dial {:}, ctext {:}, non-text {:}".format(self.n_dial, self.n_ctext, self.n_non))
        assert self.n_dial >= 0

        self.dial_gen = self.get_batch_index_generator(self.dial_index, self.n_dial)
        self.ctext_gen = self.get_batch_index_generator(self.ctext_index, self.n_ctext)
        self.non_gen = self.get_batch_index_generator(self.non_index, self.n_non)

    def __len__(self):
        # return math.ceil(self.n_samples / float(self.batch_size))
        return self.n_samples

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(self.n_samples / float(self.batch_size))):
            batch_index = []

            for i in range(self.n_gpu):
                batch_index_gpu = self.get_batch()
                batch_index.extend(batch_index_gpu)

            for index in batch_index:
                yield index

    def get_batch(self):
        batch_index = []
        if self.n_dial > 0:
            try:
                batch_index.extend(next(self.dial_gen))
            except StopIteration:
                self.dial_gen = self.get_batch_index_generator(self.dial_index, self.n_dial)
                batch_index.extend(next(self.dial_gen))

        if self.n_ctext > 0:
            try:
                batch_index.extend(next(self.ctext_gen))
            except StopIteration:
                self.ctext_gen = self.get_batch_index_generator(self.ctext_index, self.n_ctext)
                batch_index.extend(next(self.ctext_gen))

        if self.n_non > 0:
            try:
                batch_index.extend(next(self.non_gen))
            except StopIteration:
                self.non_gen = self.get_batch_index_generator(self.non_index, self.n_non)
                batch_index.extend(next(self.non_gen))

        return batch_index

    def get_batch_index_generator(self, a_list, batch_size):
        def get_batch_index(a_list, batch_size):
            assert isinstance(a_list, list)
            for start in range(0, len(a_list), batch_size):
                yield a_list[start:start + batch_size]

        assert isinstance(a_list, list)
        a_list = sample_func(a_list, len(a_list))
        generator = get_batch_index(a_list, batch_size)
        return generator


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0,
                 block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False,
                 mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False,
                 s2s_share_segment=False, pos_shift=False,
                 c_indexer=None, c_tfidf_map=None, tfidf_eps=1e-8, dial_mask_rate=0, only_mask_last=False, FGfree_indexer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.FGfree_indexer = FGfree_indexer

        # *ZY*
        self.dial_mask_rate = dial_mask_rate
        self.only_mask_last = only_mask_last  # TODO: to calculate perplexity 

        assert isinstance(c_tfidf_map, dict)
        self.c_tfidf_map = c_tfidf_map
        self.tfidf_eps = tfidf_eps

        self.nan_cond = '<nan>'
        assert isinstance(c_indexer, dict)
        if self.nan_cond not in c_indexer.keys():
            print('#'*10+'To add <nan> condition, we re-arranged c_indexer (+1)'+'#'*10)
            sys.stdout.flush()
            self.c_indexer = {self.nan_cond: 0}
            for i, u in enumerate(c_indexer.keys()):
                self.c_indexer[u] = i + 1
        else:
            self.c_indexer = c_indexer

        # Check
        assert sorted(list(self.c_indexer.values())) == list(range(len(self.c_indexer)))

        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

        assert self.has_oracle is False
        assert self.pos_shift is False  # I did not check this option
        assert self.num_qkv == 0

    def tfidf_mask(self, cid, cand_pos_tk, n_sample):
        tk_tfidf = []
        for _, tk in cand_pos_tk:
            try:
                tk_tfidf.append(max(self.c_tfidf_map[cid][tk], self.tfidf_eps))
            except KeyError:
                tk_tfidf.append(self.tfidf_eps)

        tk_tfidf = array(tk_tfidf)
        tk_tfidf = tk_tfidf / tk_tfidf.sum()

        tk_index = choice(range(len(tk_tfidf)), size=n_sample, replace=False, p=tk_tfidf).tolist()

        return [cand_pos_tk[idx][0] for idx in tk_index]

    def preprocess(self, tokens_a, tokens_b, cond, task_idx):

        # tokens_a = ['i', 'love', 'you']
        # tokens_b = ['you', 'like', 'me']
        # cond = '<nan>'
        # task_idx = 3

        try:
            cid = self.c_indexer[cond]
        except KeyError:
            print("Warning: {:} not in c_indexer".format(cond))
            cid = self.c_indexer[self.nan_cond]

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if len(tokens_a) > 0:
            if (task_idx == 3) and self.s2s_special_token:  # dial
                tokens = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]'] + tokens_b + ['[SEP]']
            else:
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

            num_tokens_a = len(tokens_a) + 2
            num_tokens_b = len(tokens_b) + 1

        else:  # text
            tokens = ['[CLS]'] + tokens_b + ['[SEP]']
            num_tokens_a = 0
            num_tokens_b = len(tokens_b) + 2

        effective_length = len(tokens_b)
        # if (task_idx != 3) and self.mask_source_words:
        #     effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens

        cand_pos_tk = []
        special_pos = set()  # will not be masked
        for i, tk in enumerate(tokens):
            if len(tokens_a) and (i >= len(tokens_a)+2) and (tk != '[CLS]'):  # TODO: mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                cand_pos_tk.append((i, tk))

            elif (len(tokens_a) == 0) and (i >= 1) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos_tk.append((i, tk))

            else:
                special_pos.add(i)

        if self.only_mask_last:
            cand_pos_tk = [(len(tokens)-2, tokens[-2])]

        # *ZY*
        if cond != self.nan_cond:
            if task_idx == 1:
                cand_pos = self.tfidf_mask(cond, cand_pos_tk, n_pred)
            elif (task_idx == 3) and (self.dial_mask_rate > 0.01) and (rand() < self.dial_mask_rate):
                cand_pos = self.tfidf_mask(cond, cand_pos_tk, n_pred)
            else:
                cand_pos = [p[0] for p in cand_pos_tk]
        else:
            cand_pos = [p[0] for p in cand_pos_tk]

        if self.only_mask_last:
            masked_pos = [len(tokens) - 2]
            n_real_pred = 1
        else:
            shuffle(cand_pos)
            masked_pos = set()
            max_cand_pos = max(cand_pos)

            for pos in cand_pos:  # Uniform Distribution Here
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:  # Avoid Overlapping
                    continue

                def _expand_whole_word(st, end):
                    # because of using WordPiece
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            n_real_pred = len(masked_pos)
            if n_real_pred > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]
                n_real_pred = n_pred

        masked_tokens = [tokens[pos] for pos in masked_pos]

        for pos in masked_pos:
            if self.only_mask_last or rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)

        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)

        mask_qkv = None

        is_next = 1

        if task_idx == 3:
            segment_ids = [0] * num_tokens_a + [1] * num_tokens_b
            input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
            input_mask[:num_tokens_a, :num_tokens_a].fill_(1)
            tril = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))
            input_mask[num_tokens_a:, :] = tril[num_tokens_a:, :]

        elif task_idx == 1:  # left-to-right
            segment_ids = [1] * len(tokens)
            input_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))

        elif task_idx == 0:  # bi-attn
            segment_ids = [0] * len(tokens)
            input_mask = torch.ones((self.max_len, self.max_len), dtype=torch.long)

        else:
            raise ValueError

        segment_ids.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_real_pred:
            n_pad = self.max_pred - n_real_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        # print("tokens, ", tokens)
        # print("input_ids, ", input_ids)
        # print("segment_ids, ", segment_ids)
        # print("masked_ids, ", masked_ids)
        # print("masked_pos, ", masked_pos)
        # print("input_mask, ", input_mask)
        # exit()

        return (num_tokens_a, num_tokens_b, input_ids, cid, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, is_next, task_idx)

    def preprocess_FGfree(self, tokens_a, tokens_b, cond, task_idx):
        def _get_attn_mask(n_words, num_tokens_a,
                          mask_pos_idx_map_sorted,
                          task_idx):

            if task_idx == 3:
                input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
                # Source
                input_mask[:num_tokens_a, :num_tokens_a].fill_(1)

                # Target
                tril = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))
                input_mask[num_tokens_a:, :] = tril[num_tokens_a:, :]

            elif task_idx == 1:
                input_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.long))

            else:
                raise ValueError("do not support task_idx {:}".format(task_idx))

            for i, (pos, idx) in enumerate(mask_pos_idx_map_sorted):
                input_mask[:, idx].fill_(0)
                input_mask[idx, idx].fill_(1)

            input_mask[n_words:, :].fill_(0)
            return input_mask

        # tokens_a = ['i', 'love', 'you']
        # tokens_b = ['you', 'like', 'me']
        # cond = '<nan>'
        # task_idx = 3

        try:
            cid = self.c_indexer[cond]
        except KeyError:
            print("Warning: {:} not in c_indexer".format(cond))
            cid = self.c_indexer[self.nan_cond]

        effective_length = len(tokens_b)
        # if (task_idx != 3) and self.mask_source_words:
        #     effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3 - n_pred, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if len(tokens_a) > 0:
            if (task_idx == 3) and self.s2s_special_token:  # dial
                tokens = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]'] + tokens_b + ['[SEP]']
            else:  # text
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

            num_tokens_a = len(tokens_a) + 2
            num_tokens_b = len(tokens_b) + 1

        else:  # text
            tokens = ['[CLS]'] + tokens_b + ['[SEP]']
            num_tokens_a = 0
            num_tokens_b = len(tokens_b) + 2

        cand_pos_tk = []
        special_pos = set()  # will not be masked
        for i, tk in enumerate(tokens):
            if len(tokens_a) and (i >= len(tokens_a)+2) and (tk != '[CLS]'):  # TODO: mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                cand_pos_tk.append((i, tk))

            elif (len(tokens_a) == 0) and (i >= 1) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos_tk.append((i, tk))

            else:
                special_pos.add(i)

        if self.only_mask_last:
            cand_pos_tk = [(len(tokens)-2, tokens[-2])]

        # *ZY*
        if cond != self.nan_cond:
            if task_idx == 1:
                cand_pos = self.tfidf_mask(cond, cand_pos_tk, n_pred)
            elif (task_idx == 3) and (self.dial_mask_rate > 0.01) and (rand() < self.dial_mask_rate):
                cand_pos = self.tfidf_mask(cond, cand_pos_tk, n_pred)
            else:
                cand_pos = [p[0] for p in cand_pos_tk]
        else:
            cand_pos = [p[0] for p in cand_pos_tk]

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)

        for pos in cand_pos:  # Uniform Distribution Here
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:  # Avoid Overlapping
                continue

            def _expand_whole_word(st, end):
                # because of using WordPiece
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
            n_real_pred = n_pred

        masked_tokens = [tokens[pos] for pos in masked_pos]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = ('[MASK]', tokens[pos])
            elif rand() < 0.5:  # 10%
                tokens[pos] = (get_random_word(self.vocab_words), tokens[pos])
            else:
                tokens[pos] = (tokens[pos], tokens[pos])

        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.FGfree_indexer(masked_tokens)

        # Token Indexing
        # input_ids = self.indexer(tokens)
        input_ids, position_ids, mask_pos_idx_map = self.FGfree_indexer(tokens, ret_ids_only=False)
        mask_pos_idx_map_sorted = sorted(mask_pos_idx_map.items(), key=lambda p: p[1])

        num_tokens_b += n_real_pred

        is_next = 1
        mask_qkv = None

        if task_idx == 3:
            segment_ids = [0] * num_tokens_a + [1] * num_tokens_b

        elif task_idx == 1:
            segment_ids = [1] * (num_tokens_a + num_tokens_b)

        elif task_idx == 0:
            segment_ids = [0] * (num_tokens_a + num_tokens_b)

        else:
            raise ValueError

        assert len(input_ids) == len(position_ids)
        assert len(input_ids) == len(segment_ids)

        n_words = len(input_ids)
        n_pad = self.max_len - n_words
        end_at = position_ids[-1] + 1

        # Zero Padding
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        position_ids.extend(list(range(end_at, end_at+n_pad)))

        assert len(input_ids) == len(position_ids)

        input_mask = _get_attn_mask(n_words, num_tokens_a, mask_pos_idx_map_sorted, task_idx)

        masked_pos = [mask_pos_idx_map[pos] for pos in masked_pos]

        # Zero Padding for masked target
        if self.max_pred > n_real_pred:
            n_pad = self.max_pred - n_real_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        # print("tokens, ", tokens)
        # print("input_ids, ", input_ids)
        # print("segment_ids, ", segment_ids)
        # print("position_ids, ", position_ids)
        # print("masked_ids, ", masked_ids)
        # print("masked_pos, ", masked_pos)
        # print("input_mask, ", input_mask[:n_words+2, :n_words+2])
        # exit()

        return (num_tokens_a, num_tokens_b, input_ids, cid, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, is_next, task_idx)

    def __call__(self, instance):
        tokens_a, tokens_b, cond, data_type = instance

        # print("instance: ", instance)

        if data_type == 'dial':
            task_idx = 3  # seq2seq
        elif data_type == 'mono':

            if len(tokens_a):  # TODO: Notice Here!
                tokens_b = tokens_a + ['[SEP]'] + tokens_b
                tokens_a = []

            if (rand() < 0.5) or (cond == '<nan>'):
                task_idx = 1  # generation
            else:
                task_idx = 0  # bi-attn, encoding
        else:
            raise ValueError

        if (self.FGfree_indexer is None) or (task_idx == 0):
            return self.preprocess(tokens_a, tokens_b, cond, task_idx)
        else:
            return self.preprocess_FGfree(tokens_a, tokens_b, cond, task_idx)


class Preprocess4Decoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False,
                 c_indexer=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

        # *ZY*
        self.nan_cond = '<nan>'
        assert isinstance(c_indexer, dict)
        if self.nan_cond not in c_indexer.keys():
            print('#'*10+'To add <nan> user, we re-arranged c_indexer (+1)'+'#'*10)
            sys.stdout.flush()
            self.c_indexer = {self.nan_cond: 0}
            for i, u in enumerate(c_indexer.keys()):
                self.c_indexer[u] = i + 1
        # Check
        assert sorted(list(self.c_indexer.values())) == list(range(len(self.c_indexer)))

    def __call__(self, instance):
        tokens_a, usrid, max_a_len = instance

        try:
            cid = self.c_indexer[usrid]
        except KeyError:
            print("Warning: {:} not in c_indexer".format(usrid))
            cid = self.c_indexer[self.nan_cond]

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a

        segment_ids = [0]*(len(padded_tokens_a)) \
            + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, cid, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx)
