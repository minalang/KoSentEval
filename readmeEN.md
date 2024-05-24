# KoSentEval: A Study of Korean Sentence Embedding Evaluation

This repository contains code for out our paper<br>
[KoSentEval:A Study of Korean Sentence Embedding Evaluation](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003066128), 
[PDF](https://jiisonline.org/files/DLA/20240331160929_10.%EC%A0%95%EB%AF%BC%ED%99%94.pdf)<br>
KoSentEval is a library for evaluating the quality of Korean sentence embeddings. It is motivated by [SentEval](https://github.com/facebookresearch/SentEval), but tried to capture linguistic features of Korean such as subject ommision and honorifics.
KoSentEval currently includes 2 downsteam tasks and 8 probing tasks.

## Dependencies

It follows dependency of [senteval](https://github.com/facebookresearch/SentEval):

* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0


## Task information

| Task         	| Evaluation Feature            | Data 	| Explanation	|
|:------------:|:---------------------:|:-----------:|:----------|
| Sentence Similarity(KorSTS) | Quality of Sentence                 	| [KorSTS](https://github.com/kakaobrain/kor-nlu-datasets/tree/master/KorSTS) | 두 문장 임베딩의 코사인 유사도와 0-5점 사이로 라벨링 된 점수 간의 상관관계를 평가 |
| Semantic Search | conv AI                 	| [rlhf korean dataset](https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset) | 응답후보 중 질문 문장에 대한 가장 적합한 문장을 도출.  가장 유사도가 높은 문장을 정답으로 산출 |
| Length | Surface Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 어절을 기준으로 문장의 길이를 계산하여 분류 |
| Word Content | Surface Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 데이터의 중빈도 어휘 목록 1000개를 선정하여 주어진 문장을 어휘에 따라 분류 |
| SubjOmission | Syntactic Feature                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | 문장의 주어 유무 판별 |
| Predicate | Syntactic Feature                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | 루트 노드의 하위 노드인 서술어의 논항이 되는 문장 구성성분 예측 |
| Tense | Semantic Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 문장의 시제(과거, 비과거) 분류 |
| Sentiment | semantic feature                	| [NSMC](https://github.com/e9t/nsmc) | 문장의 극성(부정, 긍정) 분류 |
| SentType | semantic feature                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)| 문장의 유형(선택의문문,  설명의문문, 요구, 금지) 분류 |
| Honorifics |semantic feature                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)+[smile style dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)| 존댓말, 반말 문장 분류 |

## Getting start with Colab
### Tasks except semantic search

Move to examples folder<br>
```cd KoSentEval/KoSentEval/examples```  

You can either select models in example file(.py),
```!python MODEL_NAME.py```

Or type model in huggingface  
```
model = AutoModel.from_pretrained('YOUR_MODEL_NAME').to(device)
tokenizer = AutoTokenizer.from_pretrained('YOUR_MODEL_NAME')
```
This is controllable parameters in this library. Example below is setting used in this paper:
```
# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 1, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 5, 'epoch_size': 5}
```

Explanation is in below:  
```
# senteval parameters
task_path                   # path to SentEval datasets (required)
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation for MR/CR/SUB/MPQA.
```
Parameters of classifier:  
```
nhid:                       # number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before training stops
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
```

### Semantic Search
move to ```semantic_search``` folder  
```cd semantic_search```

You can either select models in example file(.py), 
```!python YOUR_MODEL_NAME.py```

Or type model in huggingface 
```
model = AutoModel.from_pretrained('YOUR_MODEL_NAME').to(device)
tokenizer = AutoTokenizer.from_pretrained('YOUR_MODEL_NAME')
```

## Citation
```
@article{ART003066128,
author={Jung, M., & Song, M.},
title={ KoSentEval: A Study of Korean Sentence Embedding Evaluation},
journal={Journal of Intelligence and Information Systems},
issn={2288-4866},
year={2024},
volume={30},
number={1},
pages={179-199}
}
```