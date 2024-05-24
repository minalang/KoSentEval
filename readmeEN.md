# KoSentEval: A Study of Korean Sentence Embedding Evaluation

This repository contains code for out our paper<br>
[KoSentEval:A Study of Korean Sentence Embedding Evaluation](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003066128), 
[PDF](https://jiisonline.org/files/DLA/20240331160929_10.%EC%A0%95%EB%AF%BC%ED%99%94.pdf)<br>
KoSentEval is a library for evaluating the quality of Korean sentence embeddings. It is motivated by [SentEval](https://github.com/facebookresearch/SentEval), but tried to capture linguistic features of Korean such as subject omission and honorifics.
KoSentEval currently includes 2 downsteam tasks and 8 probing tasks.

## Dependencies

It follows dependency of [senteval](https://github.com/facebookresearch/SentEval):

* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0


## Task information

| Task         	| Evaluation Feature            | Data 	| Explanation	|
|:------------:|:---------------------:|:-----------:|:----------|
| Sentence Similarity(KorSTS) | Quality of Sentence                 	| [KorSTS](https://github.com/kakaobrain/kor-nlu-datasets/tree/master/KorSTS) | Evaluate the correlation between cosine similarity of two-sentence embedding and scores labeled between 0-5 |
| Semantic Search | conv AI                 	| [rlhf korean dataset](https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset) | Derive the most appropriate sentence for the question sentence among the answer candidates. The most similar sentence is calculated as the correct answer. |
| Length | Surface Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | Classify sentences based on their length |
| Word Content | Surface Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | Select 1000 medium frequency vocabulary lists of data and classify given sentences according to vocabulary |
| SubjOmission | Syntactic Feature                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | Determine the presence or absence of a subject in a sentence |
| Predicate | Syntactic Feature                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | Prediction of Sentence Components for Descriptors Sub-Nodes of Root Nodes |
| Tense | Semantic Feature                	| [openkorpos](https://github.com/openkorpos/openkorpos) | Classification of tenses (past and non-past) in sentences |
| Sentiment | semantic feature                	| [NSMC](https://github.com/e9t/nsmc) | Polarity (negative, positive) classification of sentences |
| SentType | semantic feature                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)| Classification of sentence types (optional questions, explanatory questions, requests, prohibitions) |
| Honorifics |semantic feature                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)+[smile style dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)| Classification of honorifics and informal sentences |

## Getting start with Colab
### Tasks except Semantic Search

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

Explanation of parameters is in below:  
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
