DATA_DIR=./data/small_2k_bert
OUTPUT_DIR=./saved/usr_2step_fine_1M
export PYTORCH_PRETRAINED_BERT_CACHE=./cache_tmp/bert-base-uncased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0


# --n_text 3000000

python biunilm/run_train.py \
  --n_text -1 --n_dial 0 --early_stop \
  --n_clayer 2 \
  --seed 42 \
  --do_preprocess \
  --do_train --do_eval --num_train_epochs 10 --valid_steps 4096 \
  --data_dir ${DATA_DIR} --tokenized_input --mask_source_words \
  --c_tfidf_map c_tfidf_map.pkl \
  --s2s_special_token --mask_prob 0.25 --max_pred 20 \
  --skipgram_prb 0.2 --skipgram_size 3 \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --bert_model bert-base-uncased --do_lower_case \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --max_seq_length 80 --max_position_embeddings 80 \
  --train_batch_size 80 --eval_batch_size 80 --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 --warmup_proportion 0.1 --label_smoothing 0

