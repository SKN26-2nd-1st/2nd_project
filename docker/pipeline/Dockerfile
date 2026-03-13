# =========================================================================
# Project:
# - Credit Card Churn
#
# Module:
# - datapipeline
#
# File: docker/pipeline/Dockerfile
#
# Purpose:
# - CSV → 전처리(Unknown 처리) → MySQL 적재를 위한 파이프라인 실행 환경 구축
# - builder/runtime 멀티 스테이지로 이미지 용량 및 빌드 도구 포함 최소화
# - (현재 단계) 엔트리포인트(CMD) 없이 compose/run에서 실행
# - (현재 단계) 모델 훈련은 하지 않으며, 추후 모델 파일 로드하여 구동만 수행
#
# Author: 조동휘
# Created: 2026-03-06
#
# Updated:
# - 2026-03-10: initial version (조동휘)
# - 2026-03-10: 프로젝트 명 반영 및 이미지 이름 수정 (조동휘)
# =========================================================================

# =========================================================================
# Stage 1: Builder
#
# Purpose:
# - requirements.txt 기반 wheel 파일 생성
# - (필요 시) 컴파일 도구를 이 단계에서만 사용
# =========================================================================
FROM ccrm_project_python_base:py312 AS builder

# Container 작업 경로
WORKDIR /app

# pipeline package 목록 복사
COPY requirements.txt .

# requirements.txt 기반 wheel 파일 생성
RUN python -m pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt



# =========================================================================
# Stage 2: Runtime
#
# Purpose:
# - builder에서 만든 wheel만 설치하여 실행 환경 구성
# - 빌드 단계 파일/캐시 제거로 이미지 용량 최소화
# =========================================================================
FROM ccrm_project_python_base:py312

# Container 작업 경로
WORKDIR /app

# wheel 파일 복사
COPY --from=builder /wheels /wheels

# wheel 파일로 패키지 설치 후 wheel 제거
RUN python -m pip install --no-cache-dir /wheels/* \
 && rm -rf /wheels