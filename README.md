# SKY CASTER — 항공편 도착 지연 예측 서비스

사용자가 예약하려는 항공편의 지연 가능성을 사전에 예측하여, 동일 노선 내에서 상대적으로 안정적인 항공편을 선택할 수 있도록 돕는 서비스입니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [기술 스택](#기술-스택)
3. [데이터셋](#데이터셋)
4. [전처리 요약](#전처리-요약)
5. [모델 학습 및 성능](#모델-학습-및-성능)
6. [시스템 구조](#시스템-구조)
7. [실행 방법](#실행-방법)
8. [한계점](#한계점)
9. [산출물](#산출물)

---

## 프로젝트 개요

### 배경

항공편 지연은 비즈니스 출장객의 회의 일정 취소, 여행객의 숙박·투어 예약 손실 등 직접적인 피해를 초래한다. 기존 항공 예약 서비스는 스케줄과 가격 정보를 제공하지만, 개별 항공편의 지연 위험도를 사전에 알려주는 기능은 없다.

### 목표

- 항공편 출발 전 확정된 스케줄 정보와 운항 당일 기상·운영 데이터를 바탕으로 도착 지연 여부를 예측한다.
- 동일 목적지 노선에서 지연 가능성이 낮은 선택지를 제시한다.

### 문제 정의

| 항목 | 내용 |
|------|------|
| 문제 유형 | 이진 분류 |
| 타겟 변수 | 도착 지연 여부 (`ArrDelayMinutes >= 15` → 지연(1), 그 외 → 정시(0)) |
| 주요 평가 지표 | ROC-AUC (최우선), Average Precision, Recall (지연 클래스) |

---

## 기술 스택

| 구분 | 사용 기술 |
|------|---------|
| Language | Python 3.11 |
| ML | XGBoost, LightGBM, Scikit-learn, PyTorch |
| HPO | Optuna |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | MySQL |
| 환경 관리 | Conda (`environment.yml`) |

---

## 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | Kaggle / BTS TranStats On-Time Performance |
| 수집 기간 | 2018.01 ~ 2022.07 |
| 전체 데이터 수 | 약 2,889만 건 |
| 대상 공항 수 | 377개 |
| 지연 비율 | 약 17.5% (15분 이상 도착 지연 기준) |
| 원본 피처 수 | 61개 |
| 최종 입력 피처 수 | 38개 |
| Train / Test 분리 | 시간 기준 6.5 : 3.5 |

Train/Test 분리는 랜덤 분리가 아닌 **시간 기준 분리**를 적용했다. 항공편 지연 예측은 과거 데이터로 미래 항공편을 예측하는 문제이므로, 미래 데이터 분포가 학습 과정에 유입되는 것을 방지하기 위함이다.

---

## 전처리 요약

### 핵심 원칙

예측 시점(항공편 출발 전)에 알 수 없는 데이터는 입력 피처로 사용하지 않는다. `TaxiOut`, `TaxiIn`, `AirTime`, `ActualElapsedTime` 등 실제 운항 후 확정되는 컬럼은 직접 사용 대신 과거 평균 집계값으로 대체하거나 제거했다.

### 주요 처리 단계

| 단계 | 내용 |
|------|------|
| 결항/회항 제거 | 운항하지 않은 항공편 제거 |
| Leakage 컬럼 제거 | 운항 후 확정되는 컬럼(`TaxiOut`, `AirTime` 등) 제거 |
| 시간 순환 인코딩 | `CRSDepTime`, `CRSArrTime`, `Month`를 sin/cos로 변환 |
| 집계 피처 생성 | 공항별·시간대별 평균 TaxiOut/TaxiIn, 노선 평균 비행시간 생성 (Train 기준 계산, Test에 매핑) |
| 이진 피처 생성 | 강수/적설 발생 여부(`has_precip`, `has_snow`) |
| 다중공선성 제거 | VIF 분석으로 중복 피처 12개 제거 |
| 최종 피처 수 | 61개 → 79개(파생 포함) → 50개(1차 정리) → **38개** |

### 최종 입력 피처 (38개)

| 그룹 | 피처 |
|------|------|
| 날짜/계절성 (6) | `Month`, `DayofMonth`, `DayOfWeek`, `is_weekend`, `month_sin`, `month_cos` |
| 항공사 (3) | `Marketing_Airline_Network`, `Operating_Airline`, `is_codeshare` |
| 공항 (2) | `Origin`, `Dest` |
| 예정 시간 (5) | `CRSDep_sin`, `CRSDep_cos`, `CRSArr_sin`, `CRSArr_cos`, `CRSElapsedTime` |
| 날씨-출발지 (8) | `precipitation_mm`, `snowfall_cm`, `windspeed_max_kmh`, `windgusts_max_kmh`, `temp_max_c`, `cloudcover_mean_pct`, `has_precip`, `has_snow` |
| 날씨-도착지 (8) | (출발지와 동일 구성) |
| 공항 혼잡도 (4) | `origin_taxiout_mean`, `origin_hour_taxiout_mean`, `dest_taxiin_mean`, `dest_hour_taxiin_mean` |
| 스케줄 여유도 (2) | `expected_elapsed_over_schedule`, `expected_elapsed_hour_over_schedule` |

---

## 모델 학습 및 성능

### 후보 모델

| 모델 | 유형 | 범주형 처리 |
|------|------|-----------|
| XGBoost | 트리 기반 부스팅 | `CategoricalDtype` (네이티브 지원) |
| LightGBM | 트리 기반 부스팅 | `CategoricalDtype` (네이티브 지원) |
| RandomForest | 트리 기반 배깅 | `TargetEncoder` |
| Stacking | 메타 앙상블 (XGB+LGBM+RF → LogisticRegression) | 각 베이스 모델 방식 동일 |
| FCNN | 딥러닝 (2단계 구조) | `LabelEncoder` + Embedding Layer |

HPO: XGBoost, LightGBM, RandomForest에 **Optuna 70 trials** 적용 (최적화 목표: ROC-AUC)

클래스 불균형 대응: 트리 모델 — 클래스 가중치 `√(n / (2 × count[c]))` / FCNN — `pos_weight = (n_neg / n_pos) × 1.5`

### 전체 모델 성능 비교 (Test 기준, 929,965건)

| 모델 | ROC-AUC | Avg Precision | Accuracy | 지연 Precision | 지연 Recall | 지연 F1 |
|------|---------|--------------|---------|--------------|-----------|--------|
| **XGBoost** | **0.8437** | **0.5356** | 0.70 | 0.55 | 0.39 | 0.46 |
| Stacking | 0.8412 | 0.5326 | **0.71** | **0.61** | 0.27 | 0.37 |
| LightGBM | 0.8376 | 0.5244 | 0.69 | 0.54 | 0.39 | 0.46 |
| RandomForest | 0.8278 | 0.5135 | 0.67 | 0.49 | 0.50 | 0.49 |
| FCNN | 0.6844 | 0.5176 | — | 0.42 | **0.76** | **0.54** |

### 최종 모델: XGBoost

ROC-AUC 0.8437로 전체 모델 중 최고 성능. Average Precision도 0.5356으로 가장 높다.
단일 모델로 Stacking 대비 배포 및 유지보수가 간단하며, 범주형 변수 네이티브 지원으로 전처리 파이프라인이 단순하다.

**XGBoost 혼동 행렬 (Test 기준)**

| | 예측: 정시(0) | 예측: 지연(1) |
|--|-------------|-------------|
| **실제: 정시(0)** | 530,361 (TN) | 96,473 (FP) |
| **실제: 지연(1)** | 183,728 (FN) | 119,403 (TP) |

**운영 목적별 대안 모델**

| 목적 | 권장 모델 | 근거 |
|------|---------|------|
| 종합 성능 | XGBoost | ROC-AUC 0.8437 최고 |
| 오경보 최소화 | Stacking | 지연 Precision 0.61 최고 |
| 미탐지 최소화 | RandomForest | 지연 Recall 0.50 (트리 모델 중 최고) |
| 지연 탐지 극대화 | FCNN | 지연 Recall 0.76 (ROC-AUC 저하 감수) |

### FCNN 구조

```
[1단계 - Static Branch]
  입력: 수치형 12개 + 범주형 임베딩 4종 (총 60-dim)
  임베딩: Marketing_Airline → 8, Operating_Airline → 8, Origin → 16, Dest → 16
  Linear(60→256) → BatchNorm → ReLU → Dropout(0.3)
  Linear(256→128) → BatchNorm → ReLU → Dropout(0.3)
  Linear(128→64) → static_repr (64-dim)

[2단계 - Dynamic Stage]
  입력: Concat(static_repr(64), 동적 특성 22개) → 86-dim
  Linear(86→256) → BatchNorm → ReLU → Dropout(0.3)
  Linear(256→128) → BatchNorm → ReLU → Dropout(0.3)
  Linear(128→64) → Linear(64→1)
```

학습 설정: BCEWithLogitsLoss, AdamW (lr=1e-3, weight_decay=1e-4), batch_size=2048, Early Stopping (patience=10, Val AUC 기준)

---

## 시스템 구조

```
SKN29_Proj2/
├── back/
│   ├── app/
│   │   ├── api/        # FastAPI 라우터
│   │   ├── service/    # 비즈니스 로직
│   │   ├── infra/      # DB 연결
│   │   ├── config.py
│   │   └── main.py
│   ├── models/         # 학습된 모델 pkl 파일 (별도 다운로드 필요)
│   └── sql/            # DB 테이블 생성 쿼리
├── front/
│   └── app.py          # Streamlit 프론트엔드
├── ml/
│   ├── preprocessing/  # 전처리 스크립트
│   ├── models/         # 모델 학습 스크립트
│   └── rev_02/
├── reports/
│   ├── 1_ai_preprocessing_result_report.md
│   ├── 2_model_training_report.md
│   └── 3_model_metadata.md
├── .env.sample
├── environment.yml
└── README.md
```

---

## 실행 방법

### 1. 환경 설정

```bash
conda env create -f environment.yml
conda activate skn2nd
```

### 2. 환경 변수 설정

`.env.sample`을 참고하여 `.env` 파일을 생성한다.

### 3. DB 초기화

`back/sql` 경로의 쿼리를 참고하여 룩업 테이블을 생성한다.

### 4. 모델 파일 추가

아래 링크에서 모델 pkl 파일을 다운로드한 후 `back/models/` 경로에 추가한다.

[모델 파일 다운로드 (Google Drive)](https://drive.google.com/drive/folders/1pAmWlBoEVxiQ_WKY4FSxfizFR6jLmMRM?usp=sharing)

### 5. 서버 실행

```bash
# FastAPI 백엔드
uvicorn back.app.main:app

# Streamlit 프론트엔드
python -m streamlit run front/app.py
```

---

## 한계점

| 항목 | 내용 |
|------|------|
| 기상 데이터 | 학습 시 실제 관측값 사용, 예측 시 기상 예보 데이터 사용으로 오차 발생 가능 |
| 공항 API | 실시간 공항 운영 데이터 미연동, 임의 데이터로 실험 |
| 미탐지율 | XGBoost 기준 지연 Recall 0.39 — 실제 지연의 61%를 정시로 예측 |
| 드리프트 검증 | Data Drift 실험을 위한 미래 데이터 미확보 |

드리프트 감지 기준: ROC-AUC < 0.80 → 재학습 트리거
증분 학습: XGBoost (`xgb_model` 이어 학습), LightGBM (`init_model` 이어 학습), RandomForest (`warm_start=True`)

---

## 산출물

| 문서 | 경로 |
|------|------|
| 데이터 전처리 결과서 | [reports/1_ai_preprocessing_result_report.md](reports/1_ai_preprocessing_result_report.md) |
| 모델 학습 결과서 | [reports/2_model_training_report.md](reports/2_model_training_report.md) |
| 모델 메타데이터 | [reports/3_model_metadata.md](reports/3_model_metadata.md) |
