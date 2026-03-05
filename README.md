# Project Overview
본 프로젝트는 **신용카드 고객 데이터를 기반으로 고객 이탈(Churn)을 예측하는 Machine Learning 시스템**을<br />
구축하는 것을 목표로 한다.

## 프로젝트 주요 기능
* 고객 행동 패턴 분석
* 고객 세그먼트 분석 (Clustering)
* 고객 이탈 예측 모델 개발
* Streamlit 기반 예측 서비스 제공
* MLflow 기반 실험 관리

---

## Dataset
본 프로젝트는 Kaggle에서 제공하는 데이터셋을 사용한다.

### Dataset
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

데이터는 **신용카드 고객의 이용 패턴과 이탈 여부를 포함한 데이터**로 구성되어있다.

### Target
```text
Attrition_Flag
```
* Existing Customer
* Attrited Customer

#### 주요 Feature
| Feature               | 설명       |
| --------------------- | -------- |
| Customer_Age          | 고객 나이    |
| Income_Category       | 소득 수준    |
| Months_on_book        | 고객 유지 기간 |
| Total_Trans_Amt       | 총 거래 금액  |
| Total_Trans_Ct        | 총 거래 횟수  |
| Avg_Utilization_Ratio | 카드 사용 비율 |

---

# 개발 규칙

## Branch Strategy

본 프로젝트는 **Organization Repository를 Fork하여 Personal
Repository에서 개발 후 Pull Request를 보내는 방식**으로 GitHub를
관리한다.

------------------------------------------------------------------------

### Organization Repository

    main        → 배포 가능 상태 (Protected)
    develop     → 통합 개발 브랜치 (Pull Requests 타깃 브랜치)

------------------------------------------------------------------------

### Personal Fork Repository

    main        → 기능 통합 브랜치 (Pull Requests 대상 브랜치)
    feature/*   → 기능 개발 브랜치
    fix/*       → 버그 수정 브랜치

------------------------------------------------------------------------

# Commit Convention

## 기본 원칙

-   **헤더(첫 줄)는 영어로 작성한다.**
-   **본문(body)은 한글로 작성한다.** (변경 이유 / 영향 / 주의사항 포함)
-   모든 커밋은 **Issue와 반드시 연결한다.**

커밋 본문 마지막 줄에는 반드시 아래 형식을 포함한다.

    Refs: #이슈번호

------------------------------------------------------------------------

## Commit Type

### feat

새로운 기능을 추가할 때 사용한다.

-   새로운 모델 추가
-   Streamlit 페이지 추가
-   Docker 서비스 추가

------------------------------------------------------------------------

### fix

버그 수정 시 사용한다.

-   예측 오류 수정
-   코드 오류 수정

------------------------------------------------------------------------

### docs

문서 변경 시 사용한다.

-   README 수정
-   코드 주석 보강

------------------------------------------------------------------------

### refactor

기능 변경 없이 코드 구조를 개선할 때 사용한다.

-   함수 분리
-   파일 구조 재정리
-   중복 코드 제거

------------------------------------------------------------------------

### chore

환경 설정 또는 빌드 관련 변경 시 사용한다.

-   `.gitignore` 수정
-   `requirements` 수정
-   폴더 구조 생성

------------------------------------------------------------------------

### test

테스트 코드 추가 또는 수정 시 사용한다.

-   pytest 추가
-   단위 테스트 작성
-   테스트 커버리지 개선

------------------------------------------------------------------------

## Type 사용 기준 요약

  상황               Type
  ------------------ ----------
  새로운 기능 추가   feat
  오류 수정          fix
  문서 변경          docs
  코드 구조 개선     refactor
  환경 / 설정 작업   chore
  테스트 코드 작업   test

------------------------------------------------------------------------

## Sample Commit Convention

    <type>(scope): <subject>

    - 변경 내용 설명
    - 왜 변경했는지
    - 영향 범위 또는 주의사항

    Refs: #123

------------------------------------------------------------------------

# Development Workflow

## Issue 생성 (Org Repo)

-   모든 작업은 **Issue 기반으로 시작한다.**
-   작업 전 반드시 Issue를 생성한다.

------------------------------------------------------------------------

## Branch Naming Rule

    feature/<기능명>
    fix/<버그명>

예시

    # feature branch
    feature/churn-prediction-model
    feature/customer-clustering
    feature/streamlit-predict-page
    
    # fix branch
    fix/prediction-threshold-error
    fix/data-preprocessing-bug

## Feature / Fix 브랜치 생성 규칙
### 전제

main은 Fork 생성 시 자동 생성되는 기본 브랜치이며, 개인 통합 브랜치 역할을 한다.

기능 개발 및 버그 수정은 반드시 $feature/*$, $fix/*$ 브랜치에서 수행한다.

---

### 1) 작업 시작 전: 최신 상태 동기화

먼저 Org의 develop을 내 fork main에 반영한다.
```git
git checkout main
git fetch upstream
git merge upstream/develop
git push origin main
```
### 2) 기능 개발 브랜치 생성

main에서 기능 브랜치를 만든다.
```git
git checkout main
git pull origin main
git checkout -b feature/<기능명>
```
예시
```git
git checkout -b feature/customer-clustering
```

---

### 3) 버그 수정 브랜치 생성

main에서 버그 수정 브랜치를 만든다.
```git
git checkout main
git pull origin main
git checkout -b fix/<버그명>
```
예시
```git
git checkout -b fix/prediction-threshold-error
```

### 4) 작업 중 최신 코드 반영(권장)

업스트림 변경이 많으면 아래 순서로 최신화한다.

1. upstream/develop → 내 main
```git
git checkout main
git fetch upstream
git merge upstream/develop
git push origin main
```
2. 내 main → 작업 브랜치
**권장(rebase)**
```git
git checkout feature/customer-clustering
git rebase main
git push origin feature/customer-clustering --force-with-lease
```
또는(merge)
```git
git checkout feature/customer-clustering
git merge main
git push origin feature/customer-clustering
```

---

### 5) PR 흐름 요약

작업 완료: feature/* 또는 fix/* → 내 fork main 으로 먼저 통합

이후: 내 fork main → Org develop 로 PR 생성

------------------------------------------------------------------------

## Commit

    <type>(scope): <subject>

    - 변경 내용 설명
    - 왜 변경했는지
    - 영향 범위 또는 주의사항

    Refs: #123

------------------------------------------------------------------------

## Pull Request 규칙

-   PR 대상 브랜치는 항상 **Organization/develop**
-   `main` 브랜치로 직접 PR을 보내지 않는다.

------------------------------------------------------------------------

## Merge 기준

-   최소 **1회 이상 리뷰 후 병합**
-   기능 테스트 완료 후 병합
-   충돌 해결 후 최신 상태 유지

------------------------------------------------------------------------

# Project Structure

## Root Directory

-   `.gitignore` : Git에서 추적하지 않을 파일 정의\
-   `LICENSE` : 프로젝트 라이선스 파일\
-   `structure.txt` : 현재 폴더 구조 기록 파일\
-   `README.md` : 프로젝트 설명문

------------------------------------------------------------------------

## app

Streamlit 기반 **UI 레이어**

    app/
     ├─ common/
     └─ pages/

-   `common/` : 공통 UI 컴포넌트 및 유틸리티
-   `pages/` : 멀티 페이지 구성 영역

※ 향후 `main.py`가 서비스 진입점 역할 수행

------------------------------------------------------------------------

## artifacts

학습 및 실행 결과 저장 영역

    artifacts/
     ├─ logs/
     ├─ metrics/
     └─ models/

-   `logs/` : 학습 및 실행 로그
-   `metrics/` : 모델 평가 지표 결과
-   `models/` : 학습 완료 모델 파일

※ 추후 MLflow 도입 시 확장 가능

------------------------------------------------------------------------

## config

환경 설정 및 하이퍼파라미터 설정 파일 위치

------------------------------------------------------------------------

## Database

데이터베이스 관련 파일

-   초기 스키마
-   SQL 스크립트
-   마이그레이션 파일

※ Docker Compose 도입 시 초기화 스크립트 포함 가능

------------------------------------------------------------------------

## notebooks

EDA 및 실험용 Jupyter Notebook 저장

-   운영 코드와 실험 코드 분리 목적
-   **개인 영문명으로 폴더 생성 후 안에서 개발**

------------------------------------------------------------------------

## src

핵심 비즈니스 로직 및 ML 코드

    src/
     ├─ data/
     └─ models/

-   `data/` : 데이터 로딩 및 전처리 로직
-   `models/` : 모델 학습 및 예측 로직

※ 향후 `features/`, `inference/`, `utils/` 등 확장 가능

------------------------------------------------------------------------

# 설계 원칙

-   UI (`app`)와 ML 로직 (`src`) 분리
-   실험 코드 (`notebooks`)와 운영 코드 분리
-   결과물 (`artifacts`) 분리로 재현성 확보

------------------------------------------------------------------------

## Programming Style (Functional + OOP)

본 프로젝트는 **Functional Programming과 Object-Oriented Programming을
혼합하여 사용한다.**

-   **실험 코드 (`notebooks`)**
    -   전처리, 학습, 평가 로직은 **함수(`def`) 중심**으로 작성한다.
-   **운영 코드 (`src`)**
    -   데이터 전처리 및 Feature Engineering 로직은
        **함수형(Functional)**으로 작성한다.
-   **서빙 로직 (`src/inference`)**
    -   전처리와 모델을 묶는 **얇은 예측기 클래스(Predictor)**를
        사용한다.
-   **Streamlit UI (`app`)**
    -   페이지 구성은 **함수 기반 구조**를 사용한다.

------------------------------------------------------------------------

## Naming Convention

본 프로젝트는 **Python PEP8 스타일 가이드를 따른다.**

-   변수명 → **snake_case**
-   함수명 → **snake_case**
-   클래스명 → **PascalCase**
-   DataFrame 컬럼명 → **snake_case**

### Example

``` python
# 변수
user_age = 30

# 함수
def preprocess_data(df):
    return df

# 클래스
class ChurnPredictor:
    pass

# DataFrame 컬럼 예시
subscription_length
usage_frequency
watch_hours
support_calls
```

------------------------------------------------------------------------

### Naming Rule

-   약어 사용을 최소화한다.
-   의미가 명확한 이름을 사용한다.

예시

❌ 좋지 않은 예

``` python
df2
x1
```

✅ 좋은 예

``` python
user_usage_df
churn_probability
```

------------------------------------------------------------------------

# Project Development Phases

## Phase 0 --- Baseline System

-   데이터 EDA 및 기본 분석
-   Feature Engineering
-   Clustering 기반 고객 세그먼트 분석
-   기본 Churn Prediction 모델 구축
-   Streamlit 기반 예측 서비스 구현

---

## Phase 1 --- System Extension

-   GPU 기반 모델 학습 실험
-   CPU vs GPU 성능 비교
-   MLflow 기반 Experiment Tracking 도입
-   모델 성능 비교 및 최적 모델 선정

---

## Phase 2 --- Production-ready System

- 모델 해석(Feature Importance / SHAP) 추가
- Threshold 최적화(Precision–Recall trade-off 기반)
- Inference 파이프라인 정리 (`src/inference/predictor.py` 중심)
- 예측 결과/입력 데이터 간단 모니터링(분포 체크 및 로그 저장)
- Streamlit 서비스 고도화(세그먼트 기반 결과/설명 표시)