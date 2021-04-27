import os
import sys

import re
from itertools import chain
from numpy import mean

from nltk.tokenize import TweetTokenizer


NAN = '<nan>'


def read_ref(rpath, do_split=False):
    ref_list = []
    with open(rpath, 'rt') as f:
        for line in f:
            _, _, _, ref = line.strip().split('\t')
            ref = ref.strip()
            assert len(ref) > 0
            if do_split:
                ref_list.append(ref.split(' '))
            else:
                ref_list.append(ref)

    return ref_list


def read_tokenized_ref(rpath, do_split=False):
    ref_list = []
    with open(rpath, 'rt') as f:
        for line in f:
            _, _, ref, _ = line.strip().split('\t')
            ref = ref.strip().replace(' ##', '')
            assert len(ref) > 0
            if do_split:
                ref_list.append(ref.split(' '))
            else:
                ref_list.append(ref)

    return ref_list


def read_tokenized_src(rpath, do_split=False):
    assert do_split is False
    ref_list = []
    with open(rpath, 'rt') as f:
        for line in f:
            src, _, _, _ = line.strip().split('\t')
            src = src.strip().replace(' ##', '').strip()

            assert 'SEP' not in src
            assert len(src) > 0
            ref_list.append(src)

    return ref_list


def read_bert_gen(rpath, do_split=False):
    gen_list = []
    with open(rpath, 'rt') as f:
        for line in f:
            line = line.strip()

            if len(line) > 0:
                gen_list.append(line.strip())
            else:
                gen_list.append(NAN)

    return gen_list


def list_to_txt(samples, wpath, to_str=False):
    assert isinstance(samples, list)
    wdir = os.path.dirname(wpath)
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    with open(wpath, 'wt') as f:
        for s in samples:
            if to_str:
                s = str(s)
            f.write(s.strip() + '\n')


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    assert isinstance(sentence, list)
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def get_distinct(gen_list_, n, batch_size=32, ret_raw=False):
    assert isinstance(gen_list_, list)

    gen_list = []
    for i in range(0, len(gen_list_), batch_size):
        gen_list.append(' '.join(gen_list_[i:i+batch_size]))

    dist_list = []
    for gen in gen_list:
        if isinstance(gen, str):
            gen = gen.strip().split(' ')

        dist_list.append(distinct_n_sentence_level(gen, n))

    assert len(dist_list) == len(gen_list)

    if ret_raw:
        return dist_list
    else:
        return mean(dist_list)


def to_uni(sentence, tokenizer=None):
    def _replace(sentence, bef_token, aft_token):
        sentence = sentence.replace(" {:} ".format(bef_token), " {:} ".format(aft_token))
        sentence = sentence.replace("{:} ".format(bef_token), "{:} ".format(aft_token))
        sentence = sentence.replace(" {:}".format(bef_token), " {:}".format(aft_token))
        return sentence

    if tokenizer is not None:
        sentence = ' '.join(tokenizer.tokenize(sentence)).strip()
    else:
        sentence = sentence.strip()

    sentence = sentence.replace("n ' t", "n't")
    sentence = sentence.replace("' m", "'m")
    sentence = sentence.replace("' s", "'s")
    sentence = sentence.replace("' re", "'re")
    sentence = sentence.replace("' d", "'d")
    sentence = sentence.replace("' ve", "'ve")
    sentence = sentence.replace("' ll", "'ll")

    # e.g. what's who's
    sentence = re.sub("([a-z])n't", r"\1 n't", sentence)
    sentence = _replace(sentence, "i'm", "i 'm")
    sentence = re.sub("([a-z])'s", r"\1 's", sentence)
    sentence = re.sub("([a-z])'re", r"\1 're", sentence)
    sentence = re.sub("([a-z])'d", r"\1 'd", sentence)
    sentence = re.sub("([a-z])'ve", r"\1 've", sentence)
    sentence = re.sub("([a-z])'ll", r"\1 'll", sentence)

    sentence = sentence.replace('. . .', '...')

    return sentence.strip()


def do_eval(rdir, model_name, do_to_uni=False):

    if 'gpt' in model_name:
        print("### TweetTokenizer")
        sys.stdout.flush()
        tokenizer = TweetTokenizer()
    else:
        tokenizer = None

    ref_list = read_tokenized_ref(rdir+'dial.test')
    src_list = read_tokenized_src(rdir+'dial.test')

    gen_list = read_bert_gen(rdir + '{:}.preds.txt'.format(model_name))
    print('Read Bert')

    if do_to_uni:
        ref_list = [to_uni(s, tokenizer) for s in ref_list]
        src_list = [to_uni(s, tokenizer) for s in src_list]
        gen_list = [to_uni(s, tokenizer) for s in gen_list]

    list_to_txt(ref_list, './tmp/ref.txt')
    list_to_txt(src_list, './tmp/src.txt')
    list_to_txt(gen_list, './tmp/{:}.txt'.format(model_name))

    avg_len = [len(s.strip().split(' ')) for s in gen_list]
    print("Average Len: {:}".format(mean(avg_len)))
    print('\n')

    print('Eval {:} Distinct...'.format(model_name))
    sys.stdout.flush()
    gen_res = [' '.join(gen_list)]
    dist1 = get_distinct(gen_res, 1)
    dist2 = get_distinct(gen_res, 2)
    dist3 = get_distinct(gen_res, 3)
    dist4 = get_distinct(gen_res, 4)
    print("Dist1: {:.3f}, Dist2: {:.3f}, Dist3: {:.3f}, Dist4: {:.3f}".format(dist1, dist2, dist3, dist4))

    print('Eval {:}...'.format(model_name))
    sys.stdout.flush()
    os.system("nlg-eval --hypothesis=tmp/{:}.txt --references=tmp/ref.txt".format(model_name))


if __name__ == '__main__':
    rdir = sys.argv[1].strip()
    model_name = sys.argv[1].strip()  # '{:}.preds.txt'.format(model_name)
    do_eval(rdir, model_name, do_to_uni=True)
