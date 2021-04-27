import os
import sys
import pickle
from collections import Counter
from numpy import sum, log10, mean


def do_pickle(obj, path):
    if not path.endswith('.pkl'):
        print("Recommend to end with '.pkl'.")

    assert os.path.exists(os.path.dirname(path))

    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('{:} have pickled.'.format(path))


def get_tfidf(rpath,
             wdir,
             show_sample=False):

    if not os.path.exists(wdir):
        os.mkdir(wdir)

    wpath = os.path.join(wdir, "c_tfidf_map.pkl")

    print('#'*20)
    print("Notice: {:} should be tokenized by Bert First.".format(os.path.basename(rpath)))
    print('#'*20)

    c_vocab_map = {}
    c_counter = {}
    with open(rpath, 'rt') as f:
        for index, line in enumerate(f):
            if (index+1) % 200000 == 0:
                print("{:}\t{:}...".format(os.path.basename(rpath), index))
                sys.stdout.flush()

            _, label, text, _ = line.strip().split('\t')
            if label not in c_vocab_map.keys():
                c_vocab_map[label] = {}

            for w in text.strip().split(' '):
                try:
                    c_vocab_map[label][w] += 1
                except KeyError:
                    c_vocab_map[label][w] = 1

            try:
                c_counter[label] += 1
            except KeyError:
                c_counter[label] = 1

    # 一些情况
    print('#'*20)
    c_counter = list(c_counter.values())
    print('{:} conditions; min: {:}, max: {:}, avg: {:}'.format(len(c_counter), min(c_counter), max(c_counter), mean(c_counter)))
    print('#'*20)
    sys.stdout.flush()

    print("# Get tf")
    sys.stdout.flush()
    c_sum_map = {label: sum(list(c_vocab_map[label].values())) for label in c_vocab_map.keys()}
    c_tfvocab_map = {}
    for label, vocab in c_vocab_map.items():
        c_tfvocab_map[label] = {w: n/c_sum_map[label] for w, n in vocab.items()}

    print("# Get idf")
    sys.stdout.flush()
    word_counter = Counter()
    for _, vocab in c_vocab_map.items():
        word_counter += Counter(vocab.keys())

    n_labels = len(c_vocab_map)
    word_idf_map = {w: log10(n_labels/n_occur) for w, n_occur in word_counter.items()}

    print("# Get tf-idf")
    sys.stdout.flush()
    c_tfidfvocab_map = {}
    for label, tfvocab_map in c_tfvocab_map.items():
        c_tfidfvocab_map[label] = {w: tf * word_idf_map[w] for w, tf in tfvocab_map.items()}

    do_pickle(c_tfidfvocab_map, wpath)

    # Write samples
    if show_sample:
        def write_sample(label):
            with open('./{:}.tmp.txt'.format(label), 'wt') as f:
                res = sorted(c_tfidfvocab_map[label].items(), key=lambda p:p[1], reverse=True)
                for w, tfidf in res:
                    f.write("{:}\t{:.8f}\n".format(w, tfidf))

            print("tfidf of {:} in ./{:}.tmp.txt".format(label, label))

        write_sample('nba')
        write_sample('movies')

    return c_tfidfvocab_map


if __name__ == '__main__':
    data_dir = sys.argv[1]
    read_filen = sys.argv[2]
    rpath = os.path.join(data_dir, read_filen)
    get_tfidf(rpath, data_dir, show_sample=True)
