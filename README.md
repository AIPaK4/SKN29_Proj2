# SKY CASTER — 항공편 도착 지연 예측 서비스

사용자가 예약하려는 항공편의 지연 가능성을 사전에 예측하여, 동일 노선 내에서 상대적으로 안정적인 항공편을 선택할 수 있도록 돕는 서비스입니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [기술 스택](#기술-스택)
3. [데이터셋](#데이터셋)
4. [탐색적 데이터 분석](#탐색적-데이터-분석)
5. [데이터 전처리](#데이터-전처리)
6. [모델 학습 및 성능](#모델-학습-및-성능)
7. [모델 해석](#모델-해석)
8. [시스템 구조](#시스템-구조)
9. [실행 방법](#실행-방법)
10. [한계점](#한계점)

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

Train/Test 분리는 랜덤 분리가 아닌 **시간 기준 분리**를 적용했다. 과거 데이터로 미래 항공편을 예측하는 실제 서비스 상황을 반영하기 위함이다. 집계 피처(평균 TaxiOut/TaxiIn 등)는 Train 기준으로만 계산하고 Test에 매핑하여 데이터 누수를 방지했다.

---

## 탐색적 데이터 분석

### 주요 발견사항

| 발견사항 | 전처리 반영 |
|---------|-----------|
| 정상 항공편이 지연 항공편보다 훨씬 많음 (약 82.5% vs 17.5%) | 이진 분류 + 클래스 불균형 대응 |
| 도착 지연 시간은 0 근처에 몰리고 긴 꼬리 분포 | 지연 시간 회귀 대신 15분 이상 여부 이진 분류 |
| 출발/도착 지연률은 유사한 흐름 | 타겟은 서비스 목적에 맞게 도착 지연 기준으로 설정 |
| 출발 시간대가 늦어질수록 지연 가능성 증가 | CRS 시간 sin/cos 인코딩 |
| 강수량·적설량은 0 값이 대부분 | 연속형 값 외에 발생 여부 이진 피처 추가 생성 |
| 실제 운항 후에만 알 수 있는 컬럼 다수 존재 | Leakage 컬럼 제거 |
| 시간/거리/스케줄 계열 피처 간 중복성 존재 | VIF 분석 기반 피처 제거 |

### 연도별 지연 추이

2020년에 지연률이 크게 낮아졌다가 2021~2022년에 다시 증가했다. 항공편 지연이 전체 항공 수요, 운항 환경, 공항 혼잡도와 함께 변화함을 나타낸다.

---

## 데이터 전처리

### 핵심 원칙

예측 시점(항공편 출발 전)에 알 수 없는 데이터는 입력 피처로 사용하지 않는다. `TaxiOut`, `TaxiIn`, `AirTime`, `ActualElapsedTime` 등 실제 운항 후 확정되는 컬럼은 직접 사용 대신 과거 평균 집계값으로 대체하거나 제거했다.

### 결측값 처리

| 결측 유형 | 대상 컬럼 예시 | 처리 방법 |
|---------|------------|---------|
| 결항/회항으로 인한 결측 | `ArrDelayMinutes`, `AirTime`, `ActualElapsedTime` | 결항/회항 항공편 전체 제거 |
| 운항 후 확정되는 값 | `DepTime`, `TaxiOut`, `TaxiIn`, `WheelsOn`, `WheelsOff` | 최종 입력 피처에서 제거 |
| 날씨 데이터 결측 | 출발지/도착지 날씨 변수 | 결측 행 제거 (결측률 낮고 전체 데이터 규모 큼) |
| 스케줄 정보 결측 | `CRSElapsedTime`, `CRSDepTime`, `CRSArrTime` | 결측 또는 비정상 HHMM 행 제거 |

### 이상값 처리

도착 지연 시간은 long-tail 분포를 가지므로 무조건 제거하지 않고 분위수 기준으로 확인 후 실제 가능한 값은 유지했다. 극단적 지연도 실제 서비스에서 중요한 위험 신호일 수 있기 때문이다.

### 피처 엔지니어링

#### 시간 순환 인코딩

`CRSDepTime`, `CRSArrTime`은 HHMM 형식 숫자이지만, 23:50과 00:10처럼 실제로 가까운 시각이 숫자상 멀리 표현되는 문제가 있다. 분 단위로 변환 후 sin/cos 인코딩을 적용했다.

| 생성 피처 | 설명 |
|---------|-----|
| `CRSDep_sin`, `CRSDep_cos` | 예정 출발 시각의 하루 순환성 반영 |
| `CRSArr_sin`, `CRSArr_cos` | 예정 도착 시각의 하루 순환성 반영 |
| `month_sin`, `month_cos` | 12월과 1월이 이어지는 계절성 반영 |

#### Leakage 방지를 위한 집계 피처

직접 사용 불가한 컬럼을 과거 평균값으로 대체했다. 집계값은 Train 데이터에서만 계산하고 Test에 매핑한다.

| 원본 컬럼 | 대체 피처 | 의미 |
|---------|---------|-----|
| `TaxiOut` | `origin_taxiout_mean` | 출발 공항의 평균 TaxiOut |
| `TaxiOut` | `origin_hour_taxiout_mean` | 출발 공항 + 시간대별 평균 TaxiOut |
| `TaxiIn` | `dest_taxiin_mean` | 도착 공항의 평균 TaxiIn |
| `TaxiIn` | `dest_hour_taxiin_mean` | 도착 공항 + 시간대별 평균 TaxiIn |
| `AirTime` | `route_airtime_mean` | 노선별 평균 비행시간 |

#### 다중공선성 제거 (VIF 분석, 12개)

`expected_elapsed_mean`, `expected_elapsed_hour_mean`, `schedule_buffer_hour`, `schedule_buffer`, `route_hour_airtime_mean`, `route_airtime_mean`, `origin_temp_mean_c`, `origin_temp_min_c`, `dest_temp_mean_c`, `dest_temp_min_c`, `Distance`, `Route`

### 피처 변화 요약

| 단계 | 피처 수 |
|------|------:|
| 원본 | 61개 |
| 파생변수 추가 | 79개 |
| 1차 정리 | 50개 |
| 다중공선성 제거 후 최종 | **38개** |

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

### 클래스 불균형 대응

전체 데이터 기준 정시 약 82.5% / 지연 약 17.5%로 불균형이다. SMOTE 등 오버샘플링은 시간 기준 분리 구조를 깨뜨릴 수 있어 적용하지 않고, 모델 학습 단계에서 클래스 가중치를 부여하는 방식을 택했다.

| 방법 | 적용 여부 | 근거 |
|------|---------|-----|
| SMOTE | 미적용 | 시간 기준 분리 구조 훼손 가능 |
| 클래스 가중치 | 적용 | 지연 클래스 학습 신호 강화 |
| 임계값 조정 | 운영 단계 고려 | 서비스 목적에 따라 Precision/Recall 균형 조정 |

---

## 모델 학습 및 성능

### 모델링 전략

| 항목 | 내용 |
|------|------|
| HPO | XGBoost, LightGBM, RandomForest — Optuna 70 trials (최적화 목표: ROC-AUC) |
| Stacking 분리 | 학습 데이터 70/30 (베이스 학습 / 블렌딩 세트) |
| FCNN | Early Stopping (patience=10, Val AUC 기준) |
| 클래스 불균형 (트리) | 가중치 `√(n / (2 × count[c]))` |
| 클래스 불균형 (FCNN) | `pos_weight = (n_neg / n_pos) × 1.5` |
| 최종 모델 선정 기준 | Test ROC-AUC 최고값 |

### 후보 모델

| 모델 | 유형 | 범주형 처리 |
|------|------|-----------|
| XGBoost | 트리 기반 부스팅 | `CategoricalDtype` (네이티브 지원) |
| LightGBM | 트리 기반 부스팅 | `CategoricalDtype` (네이티브 지원) |
| RandomForest | 트리 기반 배깅 | `TargetEncoder` |
| Stacking | 메타 앙상블 (XGB+LGBM+RF → LogisticRegression) | 각 베이스 모델 방식 동일 |
| FCNN | 딥러닝 (2단계 구조) | `LabelEncoder` + Embedding Layer |

### 전체 모델 성능 비교 (Test 기준, 929,965건)

| 모델 | ROC-AUC | Avg Precision | Accuracy | 지연 Precision | 지연 Recall | 지연 F1 |
|------|---------|--------------|---------|--------------|-----------|--------|
| **XGBoost** | **0.8437** | **0.5356** | 0.70 | 0.55 | 0.39 | 0.46 |
| Stacking | 0.8412 | 0.5326 | **0.71** | **0.61** | 0.27 | 0.37 |
| LightGBM | 0.8376 | 0.5244 | 0.69 | 0.54 | 0.39 | 0.46 |
| RandomForest | 0.8278 | 0.5135 | 0.67 | 0.49 | 0.50 | 0.49 |
| FCNN | 0.6844 | 0.5176 | — | 0.42 | **0.76** | **0.54** |

### 모델별 상세 결과

#### XGBoost

하이퍼파라미터 탐색 범위 (Optuna, 70 trials):

| 파라미터 | 탐색 범위 |
|---------|---------|
| `n_estimators` | 100 ~ 1,000 |
| `max_depth` | 3 ~ 10 |
| `learning_rate` | 0.01 ~ 0.30 (log scale) |
| `subsample` | 0.6 ~ 1.0 |
| `colsample_bytree` | 0.6 ~ 1.0 |
| `min_child_weight` | 1 ~ 10 |
| `reg_lambda` | 0.001 ~ 10.0 (log scale) |

혼동 행렬 (Test):

| | 예측: 정시(0) | 예측: 지연(1) |
|--|-------------|-------------|
| **실제: 정시(0)** | 530,361 (TN) | 96,473 (FP) |
| **실제: 지연(1)** | 183,728 (FN) | 119,403 (TP) |

#### LightGBM

혼동 행렬 (Test):

| | 예측: 정시(0) | 예측: 지연(1) |
|--|-------------|-------------|
| **실제: 정시(0)** | 525,651 (TN) | 101,183 (FP) |
| **실제: 지연(1)** | 183,664 (FN) | 119,467 (TP) |

#### RandomForest

혼동 행렬 (Test):

| | 예측: 정시(0) | 예측: 지연(1) |
|--|-------------|-------------|
| **실제: 정시(0)** | 469,988 (TN) | 156,846 (FP) |
| **실제: 지연(1)** | 152,233 (FN) | 150,898 (TP) |

지연 Recall 0.50으로 트리 모델 중 최고 — 미탐지 최소화가 목표일 경우 대안.

#### Stacking

구조:
```
베이스 레이어 (학습 데이터 70%):
  XGBoost / LightGBM / RandomForest

메타 레이어 (블렌딩 세트 30%):
  LogisticRegression (C=1.0, max_iter=1000)
  입력: [P_xgb, P_lgbm, P_rf] → 3차원 확률 벡터
```

혼동 행렬 (Test):

| | 예측: 정시(0) | 예측: 지연(1) |
|--|-------------|-------------|
| **실제: 정시(0)** | 574,238 (TN) | 52,596 (FP) |
| **실제: 지연(1)** | 221,220 (FN) | 81,911 (TP) |

지연 Precision 0.61로 전체 모델 중 최고 — 오경보 최소화가 목표일 경우 대안. 단, 지연 Recall 0.27로 전체 최저.

#### FCNN

구조:
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
  (Sigmoid 없음 — BCEWithLogitsLoss 학습, 예측 시 외부에서 sigmoid 적용)
```

학습 설정:

| 파라미터 | 값 |
|---------|---|
| 손실 함수 | BCEWithLogitsLoss |
| `pos_weight` | `(n_neg / n_pos) × 1.5` |
| 옵티마이저 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 배치 크기 | 2,048 |
| 최대 Epoch | 300 |
| Early Stopping | patience=10 (Val AUC 기준) |
| LR Scheduler | ReduceLROnPlateau (mode="max", patience=3) |

최적 임계값 0.52 기준 분류 보고서:

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| 정시(0) | 0.81 | 0.48 | 0.61 |
| 지연(1) | 0.42 | 0.76 | 0.54 |

지연 Recall 0.76으로 트리 모델 대비 높으나 ROC-AUC 0.6844로 낮다.

### 최종 모델: XGBoost

ROC-AUC 0.8437, Average Precision 0.5356으로 전체 모델 중 최고 성능. 단일 모델로 Stacking 대비 배포 및 유지보수가 간단하며, 범주형 변수 네이티브 지원으로 전처리 파이프라인이 단순하다.

**운영 목적별 대안 모델**

| 목적 | 권장 모델 | 근거 |
|------|---------|-----|
| 종합 성능 | XGBoost | ROC-AUC 0.8437 최고 |
| 오경보 최소화 | Stacking | 지연 Precision 0.61 최고 |
| 미탐지 최소화 | RandomForest | 지연 Recall 0.50 (트리 모델 중 최고) |
| 지연 탐지 극대화 | FCNN | 지연 Recall 0.76 (ROC-AUC 저하 감수) |

---

## 모델 해석

### 정적/동적 피처 분리 설계

| 구분 | 설명 | 예시 |
|------|-----|-----|
| 정적 (Static) | 비행 출발 전 확정, 당일 변화 없음 | 스케줄, 노선, 항공사 |
| 동적 (Dynamic) | 운항 당일 수집 | 기상 조건, 공항 혼잡도 |

트리 기반 모델은 정적·동적 피처를 단일 입력 벡터로 결합하여 처리한다. FCNN만 2단계 구조로 정적 피처를 먼저 인코딩한 뒤 동적 피처와 결합한다.

### 데이터 드리프트 모니터링

- 감지 기준: ROC-AUC < 0.80 → 재학습 트리거

| 모델 | 증분 학습 방식 |
|------|-------------|
| XGBoost | `xgb_model=model.get_booster()` 로 이어 학습 |
| LightGBM | `init_model=model` 로 이어 학습 |
| RandomForest | `warm_start=True` + 트리 50개 추가 |

증분 학습 후에도 ROC-AUC < 0.80이면 전체 재학습 권고.

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
