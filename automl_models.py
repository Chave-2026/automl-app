# -*- coding: utf-8 -*-
"""
커스텀 모델 클래스 — 별도 모듈로 두어 클래스 정체성이 실행마다 바뀌지 않게 함.
(Streamlit 재실행 시 메인 스크립트의 클래스는 재정의되어 joblib 저장이 실패하므로,
 저장 대상이 되는 커스텀 추정기는 반드시 임포트 가능한 모듈에 정의한다.)
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class PLSR(PLSRegression):
    """PLS 회귀 — 예측값을 1차원으로 반환(지표 계산 호환)."""
    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


class XGBClassifierStr(XGBClassifier):
    """문자열 클래스 라벨도 받도록 내부에서 라벨 인코딩하는 XGBoost 분류기."""
    def fit(self, X, y, **kw):
        self._le = LabelEncoder()
        super().fit(X, self._le.fit_transform(y), **kw)
        return self

    def predict(self, X, **kw):
        return self._le.inverse_transform(super().predict(X, **kw))
