# :rocket: Semantic Textual Similarity (STS)

## :closed_book: 프로젝트 개요
본 프로젝트는 Semantic Textual Similarity(STS) task를 주제로 하고 있습니다. 특정 문장 쌍에 대해 문장간의 유사도를 나타내는 label score를 학습하고, 이를 통해 두 문장간의 유사도를 평가하는 모델을 개발하는 것이 목적입니다. 
이러한 모델을 기반으로 정보 추출, 질문-답변, 중복 문장 탐지 등 다양한 기능을 구현할 수 있을 것입니다.

## :100: 프로젝트 최종 성적
|  | score | rank |
| --- | --- | --- |
| Public | 0.9374 | :3rd_place_medal: |
| Private | 0.9461 | :1st_place_medal: |

## :family_man_man_boy_boy: 멤버 소개
|강경준|김재겸|원호영|유선우|
|:---:|:---:|:---:|:---:|

## :balance_scale: 역할 분담
|팀원| 역할 |
|:---:| --- |
| 강경준 | EDA 및 데이터 검수, 데이터 증강, 모델링 및 실험 코드 관리, LoRA 적용 |
| 김재겸 | EDA 및 데이터 검수, 데이터 전처리 및 증강 실험, 모델 서치 및 파라미터 튜닝 등 모델 성능 개발, 앙상블 |
| 원호영 | EDA 및 데이터 검수, 데이터 증강 조사⋅실험 및 관련 코드관리, 모델 서치 및 실험 |
| 유선우 | EDA 및 데이터 검수, 텍스트 정제, 프로젝트 구조 관리, 모델 실험 및 파라미터 튜닝 프로젝트 수행 절차 및 방법 |

## :computer: 개발/협업 환경
- 컴퓨팅 환경
	- V100 서버 (VS code와 SSH로 연결하여 사용)
- 협업 환경
  	- ![notion](https://img.shields.io/badge/Notion-FFFFFF?style=flat-square&logo=Notion&logoColor=black) ![github](https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white) ![WandB](https://img.shields.io/badge/WeightsandBiases-FFBE00?style=flat-square&logo=WeightsandBiases&logoColor=white)
- 의사소통
  	- ![zoom](https://img.shields.io/badge/Zoom-0B5CFF?style=flat-square&logo=Zoom&logoColor=white)

## :bookmark_tabs: 데이터 설명
- 데이터 구성
	- id : 각 데이터의 index
	- source : 문장의 출처, (petition, NSMC, slack)
		- petition : 국민청원 게시판 제목 데이터
		- NSMC : 네이버 영화 감성 분석 코퍼스
		- slack : 업스테이지(Upstage) 슬랙 데이터
	- sentence_1, sentence_2 : 추출한 문장
	- label : 문장 쌍에 대한 유사도, 0 ~ 5점, 소수점 첫째 자리

## :card_index_dividers: 프로젝트 구조
```
. level1-semantictextsimilarity-nlp-16
├─ .gitihub
├─ data_loader
│  ├─ __init__.py
│  ├─ datasets.py
│  └─ data_loaders.py
├─ data
│  ├─ dev.csv
│  ├─ sample_submission.csv
│  ├─ test.csv
│  └─ train.csv
├─ model
│  ├─ __init__.py
│  └─ model.py
├─ notebook
│  └─ 
├─ utils
│  ├─ __init__.py
│  ├─ util.py
│  ├─ augmentation.py
│  ├─ preprocessing.py
│  ├─ weighted_voting.py
│  ├─ get_module_name.py
│  ├─ filtered_ids.txt
│  └─ korean_stopwords.txt
├─ .flake8
├─ .gitignore
├─ README.md
├─ config.yaml
├─ requirements.txt
├─ train.py
└─ test.py
```

## :book: 프로젝트 수행 결과
- 전처리
	- 이상치 제거
		- 직접 데이터를 검수하여 label score가 지나치게 애매한 데이터 삭제
		- 삭제 과정에서는 각자 맡은 부분에 대해서 후보군을 선정하고, 다같이 교차 검증하는 방식으로 안전하게 데이터 삭제 진행
		- 최종적으로 전체 데이터의 2% 가량 삭제(약 150개) 
	- 텍스트 정제
		- 특수문자 제거
			- 이모티콘 등 특수문자 제거
			- ?, !, ; 의 경우 점수의 차이에 영향을 미치기 때문에 포함
		- 맞춤법 처리
			- 네이버 맞춤법 검사기 기반의 맞춤법 교정 패키지 활용
			- 전반적으로 결과 괜찮았으나, 고유명사가 분절되는 등의 문제 있었음
		- 반복 문자 제거
			- 특정 문자가 지나치게 반복되는 경우 축약 처리 적용
			- 낮은 반복수에 대해서는 차이 반영에 의미가 있다고 판단
				- 예시) ㅋ, ㅋㅋ, ㅋㅋㅋ
			- 따라서, 3개 이상 반복시에만 3개로 축약
		- 불용어 제거
			- 한국어, 영어 각각에 대한 불용어 제거 시도
			- 성능 저하로 인한 미적용
- 증강
	- 임의 토큰 삭제
		- 임의로 토큰을 삭제하여 데이터의 다양성 확보
		- 적절한 토큰화를 위해 한국어 형태소 분석기 활용
		- 임의로 토큰을 삭제한 문장이 다양성 확보에 크게 도움이 되지 않고, 증강 데이터의 품질 개선이 어려워 미적용
	- 역번역 데이터 증강
		- 한국어 -> 영어 -> 한국어로의 번역 과정을 통해 데이터의 다양성 확보
		- 번역 결과 문장의 품질이 좋지 않은 경우 많아 미적용
	- 유의어 대체 데이터 증강
		- 유의어 대체를 위해 Masekd LM 활용
		- 특정 토큰을 [MASK]토큰으로 대체한 뒤, 언어모델을 통해 해당 토큰을 추론하여 데이터 증강
		- 결과 문장에 대해 좋은 성능을 도출하기 어려워 미적용
- 모델링
	- 모델 선택시 고려사항
		- 한국어 특화 모델
		- 적절한 파라미터 사이즈 : 컴퓨팅 리소스를 고려해 1B 미만의 모델 중 적절한 크기의 모델 선정
	- 학습 과정 적용사항
		- LoRA
			- Low Rank Weight를 추가함으로써 pre-trained model을 학습 데이터에 맞게 미세조정
		- Loss function
			- MSE 위주로 학습
			- 성능이 잘 나오는 모델에 한해, MAE 추가로 학습하여 모델의 일반화 성능 개선 도모
		- 토큰 추가
			- <PERSON>토큰 추가
		- Early Stopping
			- 학습이 지나치게 오래 진행되어 학습 데이터에 과적합 되는경우 방지
- 앙상블
	- Soft Voting (Weighted Mean) 적용
		- 가중치 설정 방식
			- Model Score 직접 반영
				- validation score 기준으로 산정
				- score가 높은 모델에 더 높은 가중치를 효과적으로 부여 가능
				- $w_i = \dfrac{corr_i}{\sum_{i=1}^n corr_i}, \ i : 1,2,,n, \ n : \text{number of seleceted models}$
			- Naive Method
				- validation correlation의 소수점 둘째 자리 수를 weight로 설정
				- 직관적인 weight 설정 가능
