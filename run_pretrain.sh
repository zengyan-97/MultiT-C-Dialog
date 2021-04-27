DATA_DIR=./data/pretrain_dial_data
OUTPUT_DIR=./saved/pretrained
export PYTORCH_PRETRAINED_BERT_CACHE=./cache_tmp/bert-base-uncased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0


python biunilm/run_train.py \
  --n_clayer 2 --gate attn --FGfree --early_stop --n_text 0 \
  --seed 42 \
  --num_train_epochs 10 --valid_steps 4096 \
  --data_dir ${DATA_DIR} --tokenized_input --mask_source_words \
  --c_tfidf_map c_tfidf_map.pkl \
  --s2s_special_token --mask_prob 0.25 --max_pred 20 \
  --skipgram_prb 0.2 --skipgram_size 3 \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --bert_model bert-base-uncased --do_lower_case \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --max_seq_length 80 --max_position_embeddings 80 \
  --train_batch_size 74 --eval_batch_size 74 --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 --warmup_proportion 0.1 --label_smoothing 0

