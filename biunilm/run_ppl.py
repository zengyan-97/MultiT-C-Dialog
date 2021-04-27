"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import logging
import glob
import math
import json
import argparse
import random
import pickle
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader as seq2seq_loader
import torch.distributed as dist


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    # fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    # if (not fn_model_list) or (not fn_optim_list):
    #     return None

    if not fn_model_list:
        return None

    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   )
    if both_set:
        # *ZY*
        global_step = str(max(both_set))
        fn_model = [s for s in fn_model_list if global_step in s]
        assert len(fn_model) == 1
        # fn_optim = [s for s in fn_optim_list if global_step in s]
        # assert len(fn_optim) == 1

        tmp = Path(fn_model[0]).stem.split('.')[-2].strip().split('_')
        n_epoch = int(tmp[0].strip('e').strip())
        n_step = int(tmp[1].strip('s').strip())
        return [fn_model[0], None, int(global_step), n_epoch, n_step]
    else:
        return None


def pre_preprocess(train_flag, args, data_tokenizer, bi_uni_pipeline):
    train_flag = 'test'

    # TODO: PPL
    dial_src = os.path.join(args.data_dir, "dial.{:}".format(train_flag))
    dial_ppl_src = os.path.join(args.data_dir, "dial.{:}.ppl".format(train_flag))
    if not os.path.exists(dial_ppl_src):
        n_write = 0
        with open(dial_ppl_src, 'wt') as wf:
            with open(dial_src, 'rt') as rf:
                for line in rf:
                    src, usrid, tgt, data_type = line.strip().split('\t')[:4]
                    elems = tgt.strip().split(' ')

                    for idx in range(len(elems)):
                        word = elems[idx].strip()
                        if len(word):
                            wf.write('\t'.join([src, usrid, ' '.join(elems[:idx+1]), data_type])+'\n')
                            n_write += 1

        logger.info("Write {:} samples for perplexity calculation to {:}".format(n_write, dial_ppl_src))
    else:
        logger.info("Read ppl test file: {:}".format(dial_ppl_src))

    dataset = seq2seq_loader.MyDataset(
        [dial_ppl_src], args.eval_batch_size, data_tokenizer,
        args.max_seq_length, preprocess=bi_uni_pipeline, accept_dtypes=['dial'])

    return dataset


def validate(model, valid_dataloader, device, n_gpu):
    valid_ppl = 0
    n_samples = 0
    n_tokens = 0

    batch_size_gpu = int(valid_dataloader.batch_size / n_gpu)

    iter_bar = tqdm(valid_dataloader, desc='Iter (loss=X.XXX)')

    with torch.no_grad():
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(device) if t is not None else None for t in batch]

            num_tokens_a, num_tokens_b, input_ids, usrid_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
            oracle_pos, oracle_weights, oracle_labels = None, None, None
            input_mask = None
            assert segment_ids is not None

            loss_tuple = model(input_ids, usrid_ids, segment_ids, input_mask, lm_label_ids, is_next,
                               masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,
                               num_tokens_a=num_tokens_a, num_tokens_b=num_tokens_b,
                               masked_pos_2=oracle_pos, masked_weights_2=oracle_weights,
                               masked_labels_2=oracle_labels, mask_qkv=mask_qkv, is_ppl_eval=True)

            masked_lm_loss, next_sentence_loss, ppl = loss_tuple

            if n_gpu > 1:  # mean() to average on multi-gpu.
                # loss = loss.mean()
                masked_lm_loss = masked_lm_loss.mean()
                # next_sentence_loss = next_sentence_loss.mean()
                ppl = ppl.sum()

            # loss = masked_lm_loss + next_sentence_loss
            loss = masked_lm_loss
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

            valid_ppl += ppl.item()
            n_tokens += masked_weights.sum().item()
            n_samples += len(task_idx)

    # ppl = np.exp(valid_ppl / n_samples)
    ppl = np.exp(valid_ppl / n_tokens)  # n_tokens == n_samples, I masked one token per sample

    return ppl


def save(model, optimizer, args, i_epoch, i_step, global_step):
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(
        args.output_dir, "model.e{:}_s{:}.{:}.bin".format(i_epoch, i_step, global_step))
    torch.save(model_to_save.state_dict(), output_model_file)
    output_optim_file = os.path.join(
        args.output_dir, "optim.e{:}_s{:}.{:}.bin".format(i_epoch, i_step, global_step))
    torch.save(optimizer.state_dict(), output_optim_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_clayer', type=int, required=True,
                        help="n conditional layer")

    parser.add_argument('--gate', type=str, default="attn",
                        help="gate method: [attn|gate|gate_x2] ")

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--c_tfidf_map", type=str, required=True,
                        help="e.g. c_tfidf_map.pkl in args.data_dir")

    # parser.add_argument("--tgt_file", default=None, type=str,
    #                     help="The output data file name.")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default='',
                        type=str,
                        required=True,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path",
                        default=None,
                        type=str,
                        help="The file of pretraining optimizer.")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--valid_steps",
                        default=8192,
                        type=int)

    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")

    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")

    parser.add_argument('--max_len_a', type=int, default=0,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=0,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=20,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true',
                        help="Whether to do left to right training")
    parser.add_argument('--has_sentence_oracle', action='store_true',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")
    parser.add_argument('--relax_projection', action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    args = parser.parse_args()

    # Fine-tune use
    # assert Path(args.model_recover_path).exists(
    # ), "--model_recover_path doesn't exist"

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if args.local_rank == 0:
        dist.barrier()

    ###################################
    # *ZY*
    # Load User Mask
    with open(os.path.join(args.data_dir, args.c_tfidf_map), 'rb') as f:
        c_tfidf_map = pickle.load(f)

    # Get User Indexer
    c_indexer = {cid: index for index, cid in enumerate(sorted(list(c_tfidf_map.keys())))}
    logger.info("{:} conditions.".format(len(c_indexer)))

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

    if not args.tokenized_input:
        logger.warning("Strongly recommend using BertTokenizer(# Slow) before.")

    bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
    )), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
                                                         truncate_config={'max_len_a': args.max_len_a,
                                                                          'max_len_b': args.max_len_b,
                                                                          'trunc_seg': args.trunc_seg,
                                                                          'always_truncate_tail': args.always_truncate_tail},
                                                         mask_source_words=args.mask_source_words,
                                                         skipgram_prb=args.skipgram_prb,
                                                         skipgram_size=args.skipgram_size,
                                                         mask_whole_word=args.mask_whole_word, mode="s2s",
                                                         has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
                                                         s2s_special_token=args.s2s_special_token,
                                                         s2s_add_segment=args.s2s_add_segment,
                                                         s2s_share_segment=args.s2s_share_segment,
                                                         pos_shift=args.pos_shift, c_indexer=c_indexer,
                                                         c_tfidf_map=c_tfidf_map, only_mask_last=True)]

    logger.info("Preprocess Test Set...")
    valid_dataset = pre_preprocess('test', args, data_tokenizer, bi_uni_pipeline)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.eval_batch_size,
                                                   num_workers=args.num_workers, shuffle=False,
                                                   collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)

    special_num_here = 2048
    recover_step = _get_max_epoch_model(args.output_dir)
    # (fn_model[0], fn_optim[0], int(global_step), n_epoch, n_step)

    if recover_step:
        if recover_step[-1] % special_num_here == 0:
            n_finished_epoch = recover_step[-2] - 1
        else:
            n_finished_epoch = recover_step[-2]
            recover_step[-1] = 0  # step in an epoch
    else:
        n_finished_epoch = 0

    logger.info("### Finished {:} Epoch(s) ###".format(n_finished_epoch))

    amp_handle = None
    if args.fp16 and args.amp:
        raise NotImplementedError
        # from apex import amp
        # amp_handle = amp.init(enable_caching=True)
        # logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2

    type_vocab_size = 2  # V2

    c_indexer = torch.load(os.path.join(args.data_dir, 'c_indexer.pt'))
    n_condition = len(c_indexer)
    if '<nan>' not in c_indexer.keys():
        n_condition += 1

    num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
    relax_projection = 4 if args.relax_projection else 0
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if (recover_step is None) and (args.model_recover_path is None):
        raise ValueError

    else:
        if recover_step:
            assert args.model_recover_path is None  # TODO: automatically recover to most recent model
            logger.info("***** Recover model: {:} *****".format(recover_step[0]))
            model_recover = torch.load(recover_step[0], map_location='cpu')
            # recover_step == number of epochs
            assert isinstance(recover_step[2], int)
            global_step = recover_step[2]
        elif args.model_recover_path:
            logger.info("***** (ONLY)Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(
                args.model_recover_path, map_location='cpu')
            global_step = 0

        n_dial = 10  # FAKE
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=0,
            type_vocab_size=type_vocab_size, config_path=args.config_path, task_idx=3,
            num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings,
            label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=relax_projection,
            new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv, seg_emb=args.seg_emb,
            n_condition=n_condition, n_dial=n_dial, n_clayer=args.n_clayer, gate=args.gate)

    if args.local_rank == 0:
        dist.barrier()

    if args.fp16:
        model.half()
        if args.fp32_embedding:
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()

    model.to(device)
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelImbalance(model)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    model.eval()
    # logger.info("### First Valid")
    valid_loss = validate(model, valid_dataloader, device, n_gpu)
    logger.info("### PPL {:.3f}".format(valid_loss))


if __name__ == "__main__":
    main()
