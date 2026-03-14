"""
=========================================================================
Project:
- Customer Relationship Management

Module:
- utils

File:
- prediction.py

Purpose:
- 이탈 예측 시뮬레이션 로직을 제공합니다.

Author: @nobrain711
Created: 2026-03-13

Updated:
- 2026-03-13: initial version (@nobrain711)
=========================================================================
"""


def predict_churn_logic(
    model_name: str,
    age: int,
    amount: int,
    income: str,
    revolving: int,
) -> float:
    """
    입력값을 기반으로 고객 이탈 확률을 계산합니다.

    Args:
        model_name: 사용할 모델명
        age: 고객 연령
        amount: 총 결제 금액
        income: 소득 구간
        revolving: 리볼빙 잔액

    Returns:
        float: 0~1 사이의 이탈 확률
    """
    base = 0.45

    if amount < 3000:
        base += 0.2

    if revolving > 2500:
        base += 0.15

    weights = {
        "HistGradientBoosting": 1.2,
        "XGBoost": 1.15,
        "Random Forest": 1.05,
        "EasyEnsemble": 1.3,
        "LightGBM": 1.1,
        "Logistic Regression": 0.85,
    }

    prob = min(base * weights.get(model_name, 1.0), 0.995)

    return prob