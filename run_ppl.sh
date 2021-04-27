DATA_DIR=./data/newtmp
OUTPUT_DIR=./saved/tmp
export PYTORCH_PRETRAINED_BERT_CACHE=./saved/bert-base-uncased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0


python biunilm/run_ppl.py \
  --n_clayer 2 \
  --seed 42 \
  --data_dir ${DATA_DIR} --tokenized_input \
  --c_tfidf_map c_tfidf_map.pkl \
  --s2s_special_token --max_pred 20 \
  --skipgram_prb 0.2 --skipgram_size 3 \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --bert_model bert-base-uncased --do_lower_case \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --max_seq_length 80 --max_position_embeddings 80 \
  --train_batch_size 80 --eval_batch_size 512 --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 --warmup_proportion 0.1 --label_smoothing 0

