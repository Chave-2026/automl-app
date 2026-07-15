# -*- coding: utf-8 -*-
"""
Automatic Machine Learning — 버튼형 머신러닝 앱
Rancimat 산화안정성(OSI) 예측을 포함해, 어떤 표(CSV/Excel) 데이터가 와도
같은 파이프라인으로 자동 전처리 + 여러 모델 비교(테스트 + 5-fold 교차검증) + 예측.

FT-IR처럼 변수(파수)가 많은 스펙트럼은 변수 1000개 이하가 되도록
정수 배율을 자동 계산해 구간 평균으로 축소합니다.

실행:  streamlit run automl_app.py
"""

import io
import math
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    cross_val_predict, train_test_split, KFold, StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from automl_models import PLSR, XGBClassifierStr

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
)

st.set_page_config(page_title="Automatic Machine Learning", page_icon="🧪", layout="wide")

# ANN(MLP) 등에서 나오는 수렴 경고는 결과에 영향 없어 숨김 (로그 정리용)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 우측 상단 기본 실행 아이콘(Running man 등) 숨김 — 대신 아래 진행바로 시간을 표시
st.markdown('<style>[data-testid="stStatusWidget"]{visibility:hidden;}</style>',
            unsafe_allow_html=True)

# 그래프: Times New Roman + 마이너스 기호 정상화 + 축/제목 글씨 크기
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
AX_FS = 14        # 축 제목 크기
TITLE_FS = 18     # 그래프 제목 크기

CV_FOLDS = 5          # 교차검증 fold 수 (고정)
TEST_SIZE = 0.20      # 홀드아웃(테스트) 비율
SPEC_MIN = 100        # 이 이상 파수 열이 있으면 '스펙트럼'으로 인식
SPEC_TARGET = 1000    # 스펙트럼 축소 후 목표 변수 수
STABLE_TOL = 0.10     # 안정성 판정: 교차검증이 테스트 대비 ±10% 이내면 안정

MODEL_DESC = {
    "MLR": "다중 선형회귀 — 여러 변수의 선형 결합으로 타깃을 예측하는 기본 모델",
    "Ridge": "릿지 — 계수를 규제해 과적합을 줄인 선형모델(변수 많을 때 강함)",
    "Lasso": "라쏘 — 불필요한 변수 계수를 0으로 만들어 변수 선택까지 하는 선형모델",
    "PLSR": "부분최소제곱 회귀 — 스펙트럼 분석의 표준, 상관 높은 변수를 잠재성분으로 압축",
    "RF": "랜덤포레스트 — 여러 결정트리를 평균내는 비선형 앙상블",
    "XGBoost": "XGBoost — 정규화·결측치 처리·병렬화가 강화된 고성능 부스팅",
    "SVR": "서포트벡터 회귀 — 여유폭 안의 오차는 무시하는 커널 기반 회귀",
    "SVC": "서포트벡터 분류 — 클래스 경계를 최대 마진으로 찾는 분류기",
    "k-NN": "k-최근접이웃 — 가까운 k개 이웃의 값으로 예측",
    "Logistic": "로지스틱 회귀 — 선형 경계로 확률을 추정하는 기본 분류모델",
    "ANN": "인공신경망 — 은닉층으로 비선형 관계를 학습",
}


# ----------------------------------------------------------------------------
# 스펙트럼 자동 축소 (FT-IR 파수처럼 이름이 숫자인 열을 구간 평균으로 다운샘플)
# ----------------------------------------------------------------------------
def _as_float(name):
    try:
        return float(name)
    except (ValueError, TypeError):
        return None


def spectral_columns(feature_cols):
    return [c for c in feature_cols if _as_float(c) is not None]


def auto_factor(n_spec, target=SPEC_TARGET):
    if n_spec <= target:
        return 1
    return math.ceil(n_spec / target)


def reduce_spectral(df, feature_cols, factor):
    spec = spectral_columns(feature_cols)
    others = [c for c in feature_cols if c not in spec]
    if factor <= 1 or len(spec) < 2 * factor:
        return df[feature_cols].copy(), list(feature_cols)
    spec_sorted = sorted(spec, key=_as_float)
    out, new_cols = {}, []
    for i in range(0, len(spec_sorted), factor):
        grp = spec_sorted[i:i + factor]
        name = f"{np.mean([_as_float(c) for c in grp]):.1f}"
        out[name] = df[grp].mean(axis=1)
        new_cols.append(name)
    res = pd.DataFrame(out, index=df.index)
    for c in others:
        res[c] = df[c].values
        new_cols.append(c)
    return res, new_cols


def apply_preprocess(df, feature_cols, opts):
    """스펙트럼(파수) 열에 SNV / Savitzky-Golay 미분을 적용. 비스펙트럼 열은 그대로."""
    opts = opts or {}
    spec = spectral_columns(feature_cols)
    if not spec or not (opts.get("snv") or opts.get("sg")):
        return df[feature_cols].copy()
    spec_sorted = sorted(spec, key=_as_float)
    M = df[spec_sorted].to_numpy(dtype=float)
    if opts.get("snv"):   # 산란 보정: 행별 평균 0, 표준편차 1
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
        M = (M - mu) / sd
    if opts.get("sg"):    # Savitzky-Golay 평활/미분
        from scipy.signal import savgol_filter
        w = int(opts.get("sg_window", 11))
        if w % 2 == 0:
            w += 1
        w = min(w, M.shape[1] if M.shape[1] % 2 == 1 else M.shape[1] - 1)
        poly = min(int(opts.get("sg_poly", 2)), w - 1)
        deriv = int(opts.get("sg_deriv", 1))
        M = savgol_filter(M, window_length=w, polyorder=poly, deriv=deriv, axis=1)
    out = df[feature_cols].copy()
    out[spec_sorted] = M
    return out


def prepare_X(df, feature_cols, factor, opts):
    """전처리(SNV/SG) → 스펙트럼 축소 순으로 적용해 최종 입력행렬을 만든다."""
    dfp = apply_preprocess(df, feature_cols, opts)
    return reduce_spectral(dfp, feature_cols, factor)


# ----------------------------------------------------------------------------
# 모델 / 전처리 / 유틸
# ----------------------------------------------------------------------------
def get_models(task, pls_nc=10):
    if task == "회귀":
        return {
            "MLR": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "PLSR": PLSR(n_components=pls_nc),
            "RF": RandomForestRegressor(n_estimators=300, random_state=0),
            "XGBoost": XGBRegressor(n_estimators=300, random_state=0,
                                    verbosity=0, n_jobs=1),
            "SVR": SVR(),
            "k-NN": KNeighborsRegressor(),
            "ANN": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500,
                                random_state=0),
        }
    return {
        "Logistic": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=300, random_state=0),
        "XGBoost": XGBClassifierStr(n_estimators=300, random_state=0,
                                    verbosity=0, n_jobs=1),
        "SVC": SVC(probability=True),
        "k-NN": KNeighborsClassifier(),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                             random_state=0),
    }


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                  sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)]), num_cols, cat_cols


def fmt_elapsed(sec):
    sec = int(sec)
    return f"{sec // 60}m {sec % 60}s" if sec >= 60 else f"{sec}s"


def key_hyperparams(model):
    keys = ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate",
            "alpha", "C", "epsilon", "gamma", "kernel", "n_neighbors",
            "weights", "l1_ratio"]
    p = model.get_params()
    return {k: p[k] for k in keys if k in p}


def _num(raw):
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def render_centered(df, na_rep="-"):
    """표를 가운데 정렬해 표시. 열이 너무 많으면 성능상 기본 표로 대체."""
    if df.shape[1] > 25 or len(df) > 200:
        st.dataframe(df, width="stretch")
        return
    sty = (df.style.format(precision=2, na_rep=na_rep).hide(axis="index")
           .set_table_styles([
               {"selector": "table",
                "props": [("border-collapse", "collapse"), ("margin", "6px 0")]},
               {"selector": "th, td",
                "props": [("border", "1px solid rgba(128,128,128,0.35)"),
                          ("text-align", "center"), ("padding", "5px 14px")]},
           ]))
    st.markdown(f'<div style="overflow-x:auto;">{sty.to_html()}</div>',
                unsafe_allow_html=True)


def _read_table(upload):
    """CSV / Excel 업로드를 DataFrame으로 읽는다."""
    if upload.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(upload)
    else:
        try:
            df = pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            df = pd.read_csv(upload, encoding="cp949")
    df.columns = df.columns.astype(str)
    return df


def batch_predict_ui(pipe, red_info, feat_cols, tgt, task, key):
    orig = red_info["orig_features"]
    factor = red_info["factor"]
    opts = red_info.get("preprocess", {})

    # 입력 변수가 6개 이하면 직접 입력도 선택 가능
    if len(orig) <= 6:
        method = st.radio("입력 방식", ["직접 입력", "파일 등록"],
                          horizontal=True, key=key + "_method")
    else:
        method = "파일 등록"

    # ---------- 직접 입력 ----------
    if method == "직접 입력":
        st.write("입력 변수 값을 넣고, 비교할 실험값(실제값)을 입력하면 "
                 "예측값과 정확도를 함께 보여줍니다.")
        vals = {}
        cols = st.columns(min(3, len(orig)))
        for i, c in enumerate(orig):
            with cols[i % len(cols)]:
                vals[c] = _num(st.text_input(str(c), value="0", key=f"{key}_in_{i}"))
        actual = _num(st.text_input(f"실험값 (실제 {tgt}, 없으면 0)", value="0",
                                    key=f"{key}_actual"))
        if st.button("예측하기", key=key + "_predict"):
            Xnew, _ = prepare_X(pd.DataFrame([vals]), orig, factor, opts)
            pred = pipe.predict(Xnew[feat_cols])[0]
            if task == "회귀":
                pred = round(float(pred), 2)
                rec = {"실제값": (round(actual, 2) if actual else None),
                       "예측값": pred,
                       "정확도(%)": (round(pred / actual * 100, 1) if actual else None)}
            else:
                rec = {"실제값": None, "예측값": pred}
            render_centered(pd.DataFrame([rec]))
        return

    # ---------- 파일 등록 ----------
    st.write("입력 변수(원본 포함) CSV 또는 Excel을 올리면 전체 행을 한 번에 예측합니다.")
    bp = st.file_uploader("예측용 파일 (CSV / Excel)", type=["csv", "xlsx", "xls"], key=key)
    if bp is None:
        return
    newdf = _read_table(bp)
    missing = [c for c in orig if c not in newdf.columns]
    if missing:
        st.error(f"필요한 원본 열 {len(missing)}개가 없습니다. 예: {missing[:5]}")
        return
    Xnew, _ = prepare_X(newdf, orig, factor, opts)
    preds = pipe.predict(Xnew[feat_cols])
    pcol = f"모델 예측_{tgt}"
    out = newdf.copy()
    out[pcol] = np.round(preds, 2) if task == "회귀" else preds

    # 원본 데이터 보기(입력 변수가 많아도 그대로 확인)
    st.write("전체 데이터")
    render_centered(out)

    # 실제값 vs 예측값 비교표 — 실제값(정답) 열을 직접 선택
    st.write("실제값 vs 예측값")
    non_feat = [c for c in newdf.columns if c not in orig]
    default_idx = (non_feat.index(tgt) + 1) if tgt in non_feat else 0
    actual_col = st.selectbox("실제값(정답) 열 선택 — 정확도 계산용",
                              ["(없음)"] + non_feat, index=default_idx,
                              key=key + "_actcol")
    comp = pd.DataFrame({"예측값": out[pcol].values}, index=newdf.index)
    if actual_col != "(없음)":
        if task == "회귀":
            actual = pd.to_numeric(newdf[actual_col], errors="coerce").to_numpy()
            comp.insert(0, "실제값", np.round(actual, 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = np.where(actual != 0, preds / actual * 100.0, np.nan)
            comp["정확도(%)"] = np.round(acc, 1)
        else:
            comp.insert(0, "실제값", newdf[actual_col].astype(str).values)
            comp["일치"] = np.where(
                newdf[actual_col].astype(str).to_numpy() == np.asarray(preds).astype(str),
                "O", "X")
    render_centered(comp)


# ----------------------------------------------------------------------------
# 첫 화면 — 모드 선택
# ----------------------------------------------------------------------------
st.title("Automatic Machine Learning")
st.caption("코딩 없이 버튼으로 돌리는 머신러닝 · 어떤 데이터든 같은 파이프라인으로 자동 처리")

if "mode" not in st.session_state:
    st.session_state.mode = None

with st.sidebar:
    if st.session_state.mode:
        if st.button("← 처음 화면으로"):
            st.session_state.pop("mode", None)
            st.session_state.pop("result", None)
            st.rerun()

if st.session_state.mode is None:
    # 이 화면에서만 주입되는 랜딩 전용 스타일 (컬럼을 카드처럼)
    st.markdown("""
    <style>
    div[data-testid="stColumn"], div[data-testid="column"] {
        background: #FFFFFF;
        border: 1px solid #DDEAE0;
        border-radius: 18px;
        padding: 30px 28px 22px 28px;
        box-shadow: 0 6px 22px rgba(46,139,111,0.09);
    }
    div[data-testid="stColumn"] p { color:#4A5A52; font-size:0.95rem; }
    .aml-hero {
        margin: 6px 0 26px 0; padding: 22px 26px;
        border-radius: 18px; color:#FFFFFF;
        background: linear-gradient(120deg, #2E8B6F 0%, #3FA98A 100%);
        box-shadow: 0 6px 22px rgba(46,139,111,0.22);
    }
    .aml-hero h2 { color:#FFFFFF; margin:0 0 4px 0; font-size:1.6rem; }
    .aml-hero p  { color:#EAF6EF; margin:0; font-size:0.98rem; }
    .aml-card-emoji { font-size:2.2rem; line-height:1; }
    .aml-card-title { font-size:1.25rem; font-weight:700; margin:8px 0 6px 0;
                      color:#20302A; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="aml-hero"><h2>무엇을 할까요?</h2>'
        '<p>데이터만 있으면 코딩 없이 여러 모델을 자동으로 학습·비교하고, '
        '저장한 모델로 새 데이터를 예측할 수 있습니다.</p></div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="aml-card-emoji">🤖</div>'
                    '<div class="aml-card-title">새 예측모델 만들기</div>',
                    unsafe_allow_html=True)
        st.write("CSV·Excel 데이터로 여러 모델을 자동 학습·비교하고, 최적 모델을 저장합니다.")
        st.write("")
        if st.button("새 모델 만들기", width="stretch"):
            st.session_state.mode = "train"
            st.rerun()
    with c2:
        st.markdown('<div class="aml-card-emoji">📂</div>'
                    '<div class="aml-card-title">기존 결과 불러와서 예측하기</div>',
                    unsafe_allow_html=True)
        st.write("저장해둔 `.joblib` 모델을 불러와, 새 데이터를 바로 예측합니다.")
        st.write("")
        if st.button("기존 모델 불러오기", width="stretch"):
            st.session_state.mode = "predict"
            st.rerun()

    st.markdown(
        '<div style="position:fixed; left:0; right:0; bottom:0; width:100%; '
        'box-sizing:border-box; background:#e6f4d8; color:#243024; '
        'border-top:1px solid #c5dda0; padding:8px 20px; text-align:center; '
        'font-size:0.8rem; z-index:1000;">'
        'Sungkyunkwan University · '
        '<a href="https://sites.google.com/view/lees-lipid-lab" target="_blank" '
        'style="color:#2e6b2e;">Lee\'s Lipid Lab</a> · '
        'Manager: 김세혁 (ksh312013@gmail.com)</div>',
        unsafe_allow_html=True)
    st.stop()


# ----------------------------------------------------------------------------
# 모드 2 — 저장한 .joblib 불러와 예측
# ----------------------------------------------------------------------------
if st.session_state.mode == "predict":
    st.header("📂 저장한 모델(.joblib)로 예측")
    mf = st.file_uploader("모델 파일 (.joblib)", type=["joblib", "pkl"])
    if mf is None:
        st.info("이전에 저장한 `.joblib` 모델 파일을 올려주세요.")
        st.stop()
    try:
        bundle = joblib.load(mf)
    except Exception as e:
        st.error(f"모델을 불러올 수 없습니다: {e}")
        st.stop()
    lp = bundle["pipeline"]
    lfeat = bundle["features"]
    ltgt = bundle.get("target", "target")
    ltask = bundle.get("task", "회귀")
    lred = bundle.get("reduce_info", {"orig_features": lfeat, "factor": 1})
    st.success(f"불러온 모델 · 타깃 **{ltgt}** · 유형 **{ltask}** · 입력 변수 {len(lfeat)}개")
    batch_predict_ui(lp, lred, lfeat, ltgt, ltask, key="load_batch")
    st.stop()


# ----------------------------------------------------------------------------
# 모드 1 — 새 예측모델 만들기 (사이드바: 데이터 → 변수 선택)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1) 데이터")
    up = st.file_uploader("CSV / Excel 업로드", type=["csv", "xlsx", "xls"])

if up is not None:
    with st.spinner("파일 불러오는 중..."):
        if up.name.lower().endswith(("xlsx", "xls")):
            df = pd.read_excel(up)
        else:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0)
                df = pd.read_csv(up, encoding="cp949")
        df.columns = df.columns.astype(str)
    source = f"업로드: {up.name}"
else:
    st.info("왼쪽 사이드바에서 CSV 또는 Excel 파일을 올려주세요. "
            "입력 변수(X) 열들과 예측할 타깃(Y) 열 하나로 만들면 됩니다.")
    st.stop()

st.subheader("데이터 미리보기")
st.write(f"출처: **{source}**  ·  {df.shape[0]}행 × {df.shape[1]}열")
st.dataframe(df.head(10), width="stretch")

with st.sidebar:
    st.header("2) 변수 선택")
    target = st.selectbox("타깃(예측할 값)", df.columns, index=len(df.columns) - 1)
    feature_candidates = [c for c in df.columns if c != target]
    features = st.multiselect("입력 변수", feature_candidates, default=feature_candidates)

    y_raw = df[target]
    auto_task = "회귀"
    if (y_raw.dtype == object) or (y_raw.nunique() <= max(10, int(0.05 * len(y_raw)))
                                   and y_raw.nunique() <= 15):
        auto_task = "분류"
    st.header("3) 문제 유형")
    task = st.radio("자동 판별됨 (필요시 변경)", ["회귀", "분류"],
                    index=0 if auto_task == "회귀" else 1)

    # 스펙트럼으로 인식되면 SNV는 자동 적용(축소 전)
    n_spec = len(spectral_columns(features))
    preprocess = {"snv": True} if n_spec >= SPEC_MIN else {}
    if n_spec >= SPEC_MIN:
        if n_spec > SPEC_TARGET:
            fac = auto_factor(n_spec)
            st.caption(f"🧬 스펙트럼 {n_spec}개 감지 → SNV 자동 적용 + 자동 축소 1/{fac} "
                       f"(약 {math.ceil(n_spec / fac)}개, 구간 평균)")
        else:
            st.caption(f"🧬 스펙트럼 {n_spec}개 감지 → SNV 자동 적용")

    st.caption("모델 예측 + 5-fold 교차검증")
    run = st.button("🚀 학습 시작", type="primary", width="stretch")

if not features:
    st.warning("입력 변수를 하나 이상 선택하세요.")
    st.stop()


# ----------------------------------------------------------------------------
# 학습 — 테스트(80/20) + 5-fold 교차검증
# ----------------------------------------------------------------------------
def run_training(df, features, target, task, preprocess):
    data = df[features + [target]].dropna(subset=[target]).copy()
    if task == "회귀":
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target])
    y = data[target]
    if task == "분류":
        y = y.astype(str)

    factor = auto_factor(len(spectral_columns(features)))
    X, feat_used = prepare_X(data, features, factor, preprocess)
    reduce_info = {"orig_features": list(features), "factor": factor,
                   "preprocess": preprocess, "used_features": feat_used}

    pre, num_cols, cat_cols = build_preprocessor(X)
    pls_nc = max(2, min(10, X.shape[1], int(len(X) * 0.6)))
    models = get_models(task, pls_nc)

    strat = y if task == "분류" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=0, stratify=strat)
    cv = (StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
          if task == "분류" else
          KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0))

    rows, preds_store, test_store, fitted, errors = [], {}, {}, {}, []
    t0 = time.time()
    prog = st.progress(0.0, text="진행 중.. 0s")
    for i, (name, model) in enumerate(models.items(), 1):
        base = Pipeline([("pre", pre), ("model", model)])
        try:
            yp_cv = cross_val_predict(clone(base), X, y, cv=cv)
            hold = clone(base); hold.fit(Xtr, ytr); yp_te = hold.predict(Xte)
            full = clone(base); full.fit(X, y)
            if task == "회귀":
                rmse_te = np.sqrt(mean_squared_error(yte, yp_te))
                rmse_cv = np.sqrt(mean_squared_error(y, yp_cv))
                rows.append({
                    "모델": name,
                    "Test_R2": r2_score(yte, yp_te),
                    "Test_RMSE": rmse_te,
                    "Test_MAE": mean_absolute_error(yte, yp_te),
                    "Test_RPD": np.std(yte, ddof=1) / rmse_te if rmse_te else np.nan,
                    "CV_R2": r2_score(y, yp_cv),
                    "CV_RMSE": rmse_cv,
                    "CV_MAE": mean_absolute_error(y, yp_cv),
                    "CV_RPD": np.std(y, ddof=1) / rmse_cv if rmse_cv else np.nan,
                })
            else:
                rows.append({
                    "모델": name,
                    "Test_Acc": accuracy_score(yte, yp_te),
                    "Test_F1": f1_score(yte, yp_te, average="macro"),
                    "CV_Acc": accuracy_score(y, yp_cv),
                    "CV_F1": f1_score(y, yp_cv, average="macro"),
                })
            preds_store[name] = yp_cv
            test_store[name] = (np.asarray(yte), np.asarray(yp_te))
            fitted[name] = full
        except Exception as e:
            errors.append((name, str(e)))
        prog.progress(i / len(models),
                      text=f"진행 중.. {fmt_elapsed(time.time() - t0)} ({name})")
    prog.empty()

    board = pd.DataFrame(rows)
    sort_key = "Test_R2" if task == "회귀" else "Test_F1"
    if not board.empty and sort_key in board.columns:
        board = board.sort_values(sort_key, ascending=False).reset_index(drop=True)
    return (data, X, y, board, preds_store, test_store, fitted,
            num_cols, cat_cols, errors, reduce_info)


def diagnose_target(y, task):
    msgs = []
    n_bad = int(pd.to_numeric(y, errors="coerce").isna().sum())
    if task == "회귀":
        if n_bad == len(y):
            msgs.append("타깃 열이 전부 숫자가 아닙니다. **문제 유형을 '분류'로 바꾸거나** "
                        "숫자형 타깃 열을 고르세요.")
        elif n_bad > 0:
            msgs.append(f"타깃에 숫자로 바꿀 수 없는 값이 {n_bad}개 있어 해당 행은 제외됩니다.")
    return msgs


def to_grouped(board, task):
    b = board.set_index("모델")
    b.index.name = None   # 인덱스 이름 행(빈칸) 제거
    if task == "회귀":
        b = b[["CV_R2", "CV_RMSE", "CV_MAE", "CV_RPD",
               "Test_R2", "Test_RMSE", "Test_MAE", "Test_RPD"]]
        b.columns = pd.MultiIndex.from_tuples(
            [("5-fold CV", "R²"), ("5-fold CV", "RMSE"),
             ("5-fold CV", "MAE"), ("5-fold CV", "RPD"),
             ("Test", "R²"), ("Test", "RMSE"),
             ("Test", "MAE"), ("Test", "RPD")])
    else:
        b = b[["CV_Acc", "CV_F1", "Test_Acc", "Test_F1"]]
        b.columns = pd.MultiIndex.from_tuples(
            [("5-fold CV", "Acc"), ("5-fold CV", "F1"),
             ("Test", "Acc"), ("Test", "F1")])
    return b


def plot_pair(y_true, y_pred, task, name):
    """회귀: 실제vs예측 + 잔차 / 분류: 혼동행렬 (영문 라벨, 제목=모델명)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task == "회귀":
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_xlabel("Actual value", fontsize=AX_FS)
            ax.set_ylabel("Predicted value", fontsize=AX_FS)
            ax.set_title(name, fontsize=TITLE_FS)
            # 논문용 통계 박스 (R2 · RMSE · RPD)
            _rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            _rpd = np.std(y_true, ddof=1) / _rmse if _rmse else float("nan")
            ax.text(0.04, 0.96,
                    f"$R^2$ = {r2_score(y_true, y_pred):.3f}\n"
                    f"RMSE = {_rmse:.3g}\nRPD = {_rpd:.2f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=AX_FS - 2,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_pred, y_true - y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
            ax.axhline(0, color="r", ls="--")
            ax.set_xlabel("Predicted value", fontsize=AX_FS)
            ax.set_ylabel("Residual", fontsize=AX_FS)
            ax.set_title(name, fontsize=TITLE_FS)
            st.pyplot(fig)
    else:
        labels = sorted(pd.Series(y_true).unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for (r, cc), v in np.ndenumerate(cm):
            ax.text(cc, r, str(v), ha="center", va="center")
        ax.set_xlabel("Predicted value", fontsize=AX_FS)
        ax.set_ylabel("Actual value", fontsize=AX_FS)
        ax.set_title(name, fontsize=TITLE_FS)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)


def render_table(board, task):
    disp = to_grouped(board, task)
    sty = (disp.style.format(precision=4)
           .set_table_styles([
               {"selector": "table",
                "props": [("border-collapse", "collapse"), ("margin", "6px 0")]},
               {"selector": "th, td",
                "props": [("border", "1px solid rgba(128,128,128,0.35)"),
                          ("text-align", "center"), ("padding", "6px 18px")]},
               {"selector": "th.row_heading",
                "props": [("min-width", "150px"), ("font-weight", "600")]},
           ]))
    st.markdown(f'<div style="overflow-x:auto;">{sty.to_html()}</div>',
                unsafe_allow_html=True)


if run:
    for m in diagnose_target(df[target], task):
        st.warning(m)
    st.session_state.pop("sg_re", None)   # 이전 SG 재분석 결과 초기화
    st.session_state["result"] = run_training(df, features, target, task, preprocess)

if "result" not in st.session_state:
    st.info("왼쪽에서 변수를 고르고 **🚀 학습 시작**을 눌러주세요.")
    st.stop()

(data, X, y, board, preds_store, test_store, fitted, num_cols, cat_cols,
 errors, reduce_info) = st.session_state["result"]

if board.empty:
    st.error("모든 모델 학습에 실패했습니다. 아래 원인을 확인하세요.")
    for name, msg in errors:
        st.write(f"- **{name}**: {msg}")
    st.info("가장 흔한 원인: 문제 유형(회귀/분류)이 타깃과 안 맞거나, "
            "입력 변수에 예측에 못 쓰는 텍스트(예: 샘플 이름/ID) 열이 섞인 경우입니다.")
    st.stop()

_pp = reduce_info.get("preprocess", {})
_pp_txt = []
if _pp.get("snv"):
    _pp_txt.append("SNV")
if _pp.get("sg"):
    _pp_txt.append(f"SG(win={_pp.get('sg_window', 11)}, deriv={_pp.get('sg_deriv', 1)})")
if reduce_info["factor"] > 1 or _pp_txt:
    parts = []
    if _pp_txt:
        parts.append("전처리 " + " + ".join(_pp_txt))
    if reduce_info["factor"] > 1:
        parts.append(f"축소 1/{reduce_info['factor']}")
    st.info(f"🧬 {' · '.join(parts)} → 최종 입력 변수 {X.shape[1]}개")

if errors:
    with st.expander(f"⚠️ 일부 모델 실패 ({len(errors)}개) — 원인 보기"):
        for name, msg in errors:
            st.write(f"- **{name}**: {msg}")

# ----- 예측 결과 표 (5-fold CV 기준 자동 정렬) -----
st.header("📊 예측 결과")
st.markdown(
    "- **5-fold CV**: 데이터를 5등분해 번갈아 학습·검증한 평균 성능입니다. "
    "운에 덜 흔들려 **모델의 일반적 실력**을 보여주며, "
    "여기서는 모델 선택·저장 기준으로 씁니다.\n"
    "- **Test**: 전체 데이터 중 80%를 분리해 학습한 뒤, 학습에 전혀 쓰지 않은 "
    "나머지 20%로 딱 한 번 평가한 성능으로, **새 데이터에 가까운 실제 모델 점검**입니다. "
    "표본이 적으면 값이 크게 흔들릴 수 있습니다.")

sort_col = "CV_R2" if task == "회귀" else "CV_F1"
board = board.sort_values(sort_col, ascending=False).reset_index(drop=True)

best_name = board.iloc[0]["모델"]
best_pipe = fitted[best_name]
best_model = best_pipe.named_steps["model"]

render_table(board, task)

hp = key_hyperparams(best_model)
hp_txt = ", ".join(f"`{k}={v}`" for k, v in hp.items()) if hp else "(추가 설정 없음)"
st.markdown(f"✅ **최적 모델: {best_name}**  \n핵심 설정: {hp_txt}")

# 최적 모델 한 줄 설명만
st.caption(f"• **{best_name}** — {MODEL_DESC.get(best_name, '')}")

# 과적합/안정성 해석 — '안정적이지 않을 때만' 출력 (테스트 기준 ±10%)
if task == "회귀":
    t, c = board.iloc[0]["Test_R2"], board.iloc[0]["CV_R2"]
else:
    t, c = board.iloc[0]["Test_F1"], board.iloc[0]["CV_F1"]
denom = abs(t) if abs(t) > 0.05 else 0.05
rel = (c - t) / denom
if rel > STABLE_TOL:
    st.warning(f"⚠️ **과적합 주의** — 5-fold CV({c:.3f})가 Test({t:.3f})보다 "
               f"{rel * 100:.0f}% 높습니다. ")
elif rel < -STABLE_TOL:
    st.info(f"Test({t:.3f})가 5-fold CV({c:.3f})보다 높습니다 — 표본이 적을 때 "
            "Test 20%가 우연히 맞았던 경우가 많습니다. ")


# ----------------------------------------------------------------------------
# 그래프 (Times New Roman · 제목은 모델명만)
# ----------------------------------------------------------------------------
st.header("📈 그래프")
st.subheader("5-fold CV")
plot_pair(y, preds_store[best_name], task, best_name)
st.subheader("Test (20%)")
yte_arr, yp_te = test_store[best_name]
plot_pair(yte_arr, yp_te, task, best_name)

# 변수 중요도 — 변수가 10개 이하면 실제 개수만큼 표시
st.subheader("변수 중요도")


def _draw_importance(fnames, fvals, xlabel):
    n_total = len(fnames)
    n_show = min(10, n_total)
    imp = (pd.DataFrame({"f": fnames, "v": fvals})
           .sort_values("v", ascending=True).tail(n_show))
    st.caption(f"상위 {n_show}개" if n_total > n_show else f"전체 {n_total}개")
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.45 * n_show)))
    ax.barh(imp["f"].astype(str), imp["v"], color="#4C78A8")
    ax.set_xlabel(xlabel, fontsize=AX_FS)
    st.pyplot(fig)


def _to_original_feature(name, num_cols, cat_cols):
    """원핫 확장 컬럼명을 원래 입력 변수 이름으로 되돌린다."""
    if name in num_cols:
        return name
    for c in cat_cols:                     # 'Sample ID_SMP_011' -> 'Sample ID'
        if name == c or name.startswith(c + "_"):
            return c
    return name


try:
    pre_step = best_pipe.named_steps["pre"]
    try:
        names = np.array([n.split("__", 1)[-1]
                          for n in pre_step.get_feature_names_out()])
    except Exception:
        names = np.array(X.columns, dtype=str)

    vals, xlab = None, "Importance"
    if hasattr(best_model, "coef_"):
        vals, xlab = np.abs(np.ravel(best_model.coef_)), "Coefficient"
    elif hasattr(best_model, "feature_importances_"):
        vals, xlab = best_model.feature_importances_, "Importance"

    if vals is not None and len(vals) == len(names):
        # 원핫 더미(Sample ID_SMP_001 …)를 원래 변수 하나로 합침
        origin = [_to_original_feature(n, num_cols, cat_cols) for n in names]
        agg = (pd.DataFrame({"f": origin, "v": vals})
               .groupby("f", as_index=False)["v"].sum())
        _draw_importance(agg["f"].to_numpy(), agg["v"].to_numpy(), xlab)
    elif X.shape[1] <= 30:
        scoring = "r2" if task == "회귀" else "accuracy"
        pi = permutation_importance(best_pipe, X, y, n_repeats=10,
                                    random_state=0, scoring=scoring)
        _draw_importance(np.array(X.columns, dtype=str), pi.importances_mean,
                         "Importance")
    else:
        st.info("이 모델은 내장 중요도가 없고 변수가 많아, 중요도 그래프는 생략했습니다. "
                "(Ridge/Lasso/RF를 고르면 표시됩니다)")
except Exception as e:
    st.info(f"변수 중요도를 계산할 수 없습니다: {e}")


# ----------------------------------------------------------------------------
# Savitzky-Golay 미분 재분석 (분석 후 SG를 추가 적용해 최적 모델을 재평가·비교)
# ----------------------------------------------------------------------------
if len(spectral_columns(reduce_info["orig_features"])) >= SPEC_MIN:
    st.header("🔧 Savitzky–Golay 미분 재분석")
    st.caption("SG 미분을 추가로 적용해 **모든 모델을 다시 학습·비교**하고, 새 최적 모델을 "
               "도출해 기본 결과와 비교합니다.")
    sg_deriv = st.slider("미분 차수 (0=평활만, 1=1차, 2=2차)", 0, 2, 1, key="sg_deriv_re")
    if st.button("SG 적용해 재분석", key="sg_run"):
        sgpp = {"snv": True, "sg": True, "sg_window": 11, "sg_poly": 2,
                "sg_deriv": sg_deriv}
        st.session_state["sg_re"] = {
            "deriv": sg_deriv,
            "result": run_training(df, features, target, task, sgpp)}

    sg_re = st.session_state.get("sg_re")
    if sg_re:
        (sdata, sX, sy, sboard, spreds, stest, sfitted, snum, scat, serr,
         sred) = sg_re["result"]
        if sboard.empty:
            st.error("SG 재분석에서 모든 모델이 실패했습니다.")
        else:
            _ck = "CV_R2" if task == "회귀" else "CV_F1"
            sboard = sboard.sort_values(_ck, ascending=False).reset_index(drop=True)
            sbest = sboard.iloc[0]["모델"]
            sbest_pipe = sfitted[sbest]
            base_cv = board.sort_values(_ck, ascending=False).iloc[0]
            st.markdown(f"**SG 미분{sg_re['deriv']}차 재분석 · 5-fold CV 최적 모델: {sbest}**")
            cc1, cc2 = st.columns(2)
            cc1.metric(f"기본 5-fold CV 최적 ({base_cv['모델']})", f"{base_cv[_ck]:.3f}")
            cc2.metric(f"SG 5-fold CV 최적 ({sbest})", f"{sboard.iloc[0][_ck]:.3f}",
                       delta=f"{sboard.iloc[0][_ck] - base_cv[_ck]:+.3f} vs 기본")
            st.caption("SG 적용 후 전체 리더보드 (5-fold CV 기준 정렬):")
            render_table(sboard, task)
            buf_sg = io.BytesIO()
            joblib.dump({"pipeline": sbest_pipe, "features": list(sX.columns),
                         "target": target, "task": task, "reduce_info": sred}, buf_sg)
            st.download_button(
                f"💾 SG 미분{sg_re['deriv']}차 5-fold CV 최적모델({sbest}) 저장(.joblib)",
                buf_sg.getvalue(), f"model_{sbest}_SG{sg_re['deriv']}.joblib",
                key="save_sg")
            st.caption("이 모델은 예측 시 SG 미분도 자동 적용됩니다.")


# ----------------------------------------------------------------------------
# 모델 저장
# ----------------------------------------------------------------------------
st.header("💾 모델 저장")
st.caption(f"예측용으로 5-fold CV 기준 최적 모델을 저장합니다: **{best_name}**")
buf = io.BytesIO()
joblib.dump({"pipeline": best_pipe, "features": list(X.columns),
             "target": target, "task": task, "reduce_info": reduce_info}, buf)
st.download_button(f"{best_name} 모델(.joblib) 내려받기 — 5-fold CV 기준 최적",
                   buf.getvalue(), f"model_{best_name}.joblib")
