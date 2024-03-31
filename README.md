# KoSentEval: 한국어 문장 임베딩 평가 연구

## 개요
한국어 문장 임베딩 품질 및 언어적 속성 평가를 위해 개발한 10가지 평가 태스크 라이브러리


## 태스크 정보

| Task     	| 평가 속성         | 활용데이터 	| 설명	|
|----------	|----------------	|-----------:|----------:|
| 문장유사도(KorSTS) | 임베딩품질                 	| [KorSTS](https://github.com/kakaobrain/kor-nlu-datasets/tree/master/KorSTS) | 두 문장 임베딩의 코사인 유사도와 0-5점 사이로 라벨링 된 점수 간의 상관관계를 평가 |
| 의미검색(Semantic Search) | 임베딩품질                 	| [rlhf korean dataset](https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset) | 응답후보 중 질문 문장에 대한 가장 적합한 문장을 도출.  가장 유사도가 높은 문장을 정답으로 산출 |
| 문장 길이 분류(Length) | 표층적 속성                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 어절을 기준으로 문장의 길이를 계산하여 분류 |
| 어휘 분류(Word Content) | 표층적 속성                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 데이터의 중빈도 어휘 목록 1000개를 선정하여 주어진 문장을 어휘에 따라 분류 |
| 주어 생략 여부(SubjOmission) | 통사적 속성                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | 문장의 주어 유무 판별 |
| 서술어의 논항 예측(SubjOmission) | 통사적 속성                	| [KLUE-DP](https://klue-benchmark.com/tasks/71/overview/description) | 루트 노드의 하위 노드인 서술어의 논항이 되는 문장 구성성분 예측 |
| 시제(Tense) | 의미적 속성                	| [openkorpos](https://github.com/openkorpos/openkorpos) | 문장의 시제(과거, 비과거) 분류 |
| 극성분류(Sentiment) | 의미적 속성                	| [NSMC](https://github.com/e9t/nsmc) | 문장의 극성(부정, 긍정) 분류 |
| 문장유형분류(SentType) | 의미적 속성                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)| 문장의 유형(선택의문문,  설명의문문, 요구, 금지) 분류 |
| 경어법분류(Honorifics) | 의미적 속성                	| [StyleKQC](https://github.com/cynthia/stylekqc)+[paraKQC](https://github.com/warnikchow/paraKQC)+[smile style dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)| 존댓말, 반말 문장 분류 |


