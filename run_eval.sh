# run decoding
DATA_DIR=./data/newtmp
export PYTORCH_PRETRAINED_BERT_CACHE=./cache_tmp/bert-base-uncased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0


# run decoding
python biunilm/decode_seq2seq.py \
  --n_clayer 2 \
  --model_recover_path ./saved/tmp/bert_save/model.e2_s1.2.bin \
  --output_file ${DATA_DIR}/tmp.preds.txt \
  --batch_size 64 --beam_size 4 --max_tgt_length 36 --min_len 10 --length_penalty 0 \
  --data_dir ${DATA_DIR} \
  --input_file dial.test --split test \
  --bert_model bert-base-uncased --do_lower_case --s2s_special_token \
  --mode s2s \
  --tokenized_input \
  --max_seq_length 80 \
  --forbid_duplicate_ngrams --ngram_size 2 

