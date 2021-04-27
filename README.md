# MultiT-C-Dialog


This code is the official pytorch implementation of [A Simple and Efficient Multi-Task Learning Approach for Conditioned Dialogue Generation](https://arxiv.org/abs/2010.11140).<br>

<p align="left"><img width="95%" src="img/model.png" /></p>


In our experiments, we fine-tuned BERT for conditioned dialogue generation. \
Recently, researchers have proposed some large pre-trained dialogue models by utilizing Reddit/Twitter data. \
These models all utilize auto-regressive training objective. \
It is easy to apply this objective to conditioned language/dialogue generation task. \
However, the conditioned language encoding task in our approach applies bi-directional attention, and mask language modeling objective is thus needed. 


## Requirements
```
python3.7
torch==1.1.0
```
Then, run this: 
```
pip install .
```


Notice that when you modify the code in ./biunilm/ or ./pytorch_pretrained_bert/, \
you need to re-run this command: 
```
pip install .
```
Or, directly update the corresponding code in: 
```
xxx/anaconda3/envs/xxx/lib/python3.7/site-packages/biunilm
xxx/anaconda3/envs/xxx/lib/python3.7/site-packages/pytorch_pretrained_bert
```

## Download Data
Download [Persona Reddit](https://files.pushshift.io/reddit/) and [Topic-related Dialogue](https://github.com/nouhadziri/THRED).
We leave the data cleaning / filtering process to users. 
Process the data into labeled dialogue corpus: 
```
dial.train
dial.valid
dial.test
### each file consists of lines in the form of: 
# dialogue-context \t condition-label \t response
### for multi-turn dialogue, concatenate the turns in context using [SEP]
```
and labeled text corpus:
```
text.train
text.valid
text.test
# each file consists of lines in the form of: 
# condition-label \t text
```

## Preprocessing
Please, tokenize the dataset in advance: 
```
python ./pre_tokenize.py $rdir $wdir
```
Then, calculate TF-IDF scores in advance: 
```
python ./get_tfidf.py $datadir $rfilen

# $rpath can be the combination of text.train and dial.train (after tokenization)
```



## Model Training

Further pre-train on a dialogue corpus (optional):
```
sh -x ./pretrain.sh
# use <nan> as the condition label when preprocessing the dataset
```
  
  
Use our approach to fine-tune on a labeled dialogue corpus and a labeled text corpus: 
```
sh -x ./train.sh
```
where DATA_DIR should contain the two corpora. Some options are: 
```
--n_text: set the number of text samples
--n_dial: set the number of dialogue samples
--FGfree: eliminating finetune-generation discrepancy
--model_recover_path: load pre-trained model 
```
  
  
Or, apply sequential fine-tuning: 
```
sh -x ./run_2step_pre.sh
sh -x ./run_2step_ft.sh
```


Tips: If labeled text corpus is limited, use our approach to avoid catastrophic forgetting (training on small text corpus will largely erase the pre-training result). \
If labeled text corpus is sufficient, use sequential fine-tuning. In this case, the final training goal is optimizing dialogue generation, and it will be better. 



## Model Evaluation

Calculate perplexity on the dialogue data: 
```
sh -x ./run_ppl.sh
```
This command will automatically load the latest checkpoint in ${OUTPUT_DIR}. 


Generate responses: 
```
sh -x ./run_eval.sh
```


We provide a evaluation scrip:
```
python eval.py $rdir $model
```


## Acknowledgments
Our code is based on [UniLM](https://github.com/microsoft/unilm/tree/master/unilm-v1). Thanks!


## Citation

```bibtex
@misc{zeng2021simple,
      title={A Simple and Efficient Multi-Task Learning Approach for Conditioned Dialogue Generation}, 
      author={Yan Zeng and Jian-Yun Nie},
      year={2021},
      eprint={2010.11140},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If activating --FGfree option, please cite: 
```bibtex
@misc{zeng2020opendomain,
      title={Open-Domain Dialogue Generation Based on Pre-trained Language Models}, 
      author={Yan Zeng and Jian-Yun Nie},
      year={2020},
      eprint={2010.12780},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



## Contact
For help using this code, please submit a GitHub issue.
For serious problems, please contact Yan Zeng ([yan.zeng@umontreal.ca](mailto:yan.zeng@umontreal.ca)).


