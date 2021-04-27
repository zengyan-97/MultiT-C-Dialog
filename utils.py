import os


def tmp(data_type, dial_mask_rate):
    from random import random as rand
    if data_type != 'dial':
        print(1)
    elif data_type == 'dial' and dial_mask_rate > 0 and rand() < dial_mask_rate:
        print(1)
    else:
        print(0)




if __name__ == '__main__':
    rdir = "./data/pretrain_bert_V2/"
    wdir = "./data/pretrain_bert_V2/debug/"

    for filen in os.listdir(rdir):
        if filen.endswith('train') or filen.endswith('valid'):
            print(filen)
            os.system('head -100 {:}  > {:}'.format(rdir+filen, wdir+filen))
