import os
import sys
import time
import math
import random

from pytorch_pretrained_bert.tokenization import BertTokenizer


def simple_prep(sent, bert_tokenizer):
    return ' '.join(bert_tokenizer.tokenize(sent.strip()))


def list_to_txt(samples, wpath):
    with open(wpath, 'wt') as f:
        for s in samples:
            f.write('\t'.join(s)+'\n')


def text_to_bert(rpath, wpath, bert_model="bert-base-uncased", do_shuffle=False):
    wdir = os.path.dirname(wpath)
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    bert_model = bert_model.strip()
    if bert_model.endswith('uncased'):
        do_lower_case = True
    else:
        do_lower_case = False
    bert_tokenizer = BertTokenizer.from_pretrained(
        bert_model, do_lower_case=do_lower_case)

    samples = []
    start = time.time()
    with open(rpath, 'rt') as f:
        for index, line in enumerate(f):
            if (index+1) % 100000 == 0:
                print('{:}\t{:}\t{:.1f}min'.format(os.path.basename(rpath),
                                                        index, (time.time() - start) / 60))
                sys.stdout.flush()

            label, text = line.strip().split('\t')
            samples.append(('', label,
                            simple_prep(text, bert_tokenizer), 'mono'))

    sys.stdout.flush()

    if do_shuffle:
        random.shuffle(samples)

    list_to_txt(samples, wpath)


def dial_to_bert(rpath, wpath, bert_model="bert-base-uncased", do_shuffle=False):
    wdir = os.path.dirname(wpath)
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    bert_model = bert_model.strip()
    if bert_model.endswith('uncased'):
        do_lower_case = True
    else:
        do_lower_case = False
    tokenizer = BertTokenizer.from_pretrained(
        bert_model, do_lower_case=do_lower_case)

    samples = []
    start = time.time()
    with open(rpath, 'rt') as f:
        for index, line in enumerate(f):
            if (index+1) % 100000 == 0:
                print('{:}\t{:}\t{:.1f}min'.format(os.path.basename(rpath),
                                                        index, (time.time() - start) / 60))
                sys.stdout.flush()

            src, label, tgt = line.strip().split('\t')
            samples.append((simple_prep(src, tokenizer), label,
                            simple_prep(tgt, tokenizer), 'dial'))

    sys.stdout.flush()

    if do_shuffle:
        random.shuffle(samples)

    list_to_txt(samples, wpath)


def run_to_bert_file():
    rpath = sys.argv[1]
    wpath = sys.argv[2]

    datatype, _ = os.path.basename(rpath).split('.')
    assert datatype in ['dial', 'text']

    if datatype == 'dial':
        dial_to_bert(rpath, wpath)
    else:
        text_to_bert(rpath, wpath)


def run_to_bert_dir():
    from multiprocessing import Pool

    rdir = sys.argv[1]
    wdir = sys.argv[2]
    p = Pool(4)
    for data_type in ['dial', 'text']:
        for label in ['train', 'valid', 'test']:
            target = "{:}.{:}".format(data_type, label)
            assert os.path.exists(os.path.join(rdir, target))
            print(os.path.join(rdir, target))
            if data_type == 'dial':
                p.apply_async(dial_to_bert, args=(os.path.join(rdir, target),
                                                  os.path.join(wdir, target)))
            elif data_type == 'text':
                p.apply_async(text_to_bert, args=(os.path.join(rdir, target),
                                                  os.path.join(wdir, target)))

            else:
                raise ValueError

            print('\n')

    p.close()
    p.join()


if __name__ == '__main__':
    run_to_bert_dir()
