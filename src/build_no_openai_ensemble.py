from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ALLOWED_SEGMENTS = [
    "fit_and_sizing",
    "material_and_quality",
    "style_and_appearance",
    "overall_wearability_and_value",
]


def extract_title(text: str) -> str:
    text = str(text).strip()
    if "\n" in text:
        return text.split("\n", 1)[0][:180]
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    if len(parts) > 1 and len(parts[0]) <= 140:
        return parts[0]
    return text[:140]


def count_markers(texts: list[str], patterns: list[str]) -> np.ndarray:
    arr = np.zeros(len(texts), dtype=float)
    for i, text in enumerate(texts):
        t = str(text).lower()
        arr[i] = sum(min(len(re.findall(p, t)), 3) for p in patterns)
    return arr


def rule_features(texts: list[str]) -> np.ndarray:
    pats = {
        "pos": [
            r"love", r"loved", r"like", r"great", r"perfect", r"favorite", r"recommend", r"keeper", r"worth",
            r"happy", r"glad", r"beautiful", r"cute", r"comfortable",
            r"подоба", r"сподоб", r"люб", r"улюб", r"рекоменд", r"чудов", r"гарн", r"ідеал", r"залиш",
        ],
        "neg": [
            r"not worth", r"do not recommend", r"would not recommend", r"save your money", r"return", r"returned",
            r"sending back", r"disappointed", r"not for me", r"cannot wear", r"can't wear", r"unwearable",
            r"hard pass", r"deal breaker", r"wanted to love",
            r"не рекоменд", r"не радж", r"не варте", r"повер", r"розчар", r"не для мене", r"не змогла носити",
        ],
        "contrast": [
            r"but", r"however", r"although", r"unfortunately", r"sadly", r"though", r"still",
            r"але", r"однак", r"проте", r"хоча", r"на жаль", r"втім",
        ],
        "fit": [
            r"size", r"sizing", r"fit", r"fits", r"runs large", r"runs small", r"too big", r"too small",
            r"tight", r"loose", r"length", r"bust", r"chest", r"shoulder", r"sleeve", r"armhole",
            r"waist", r"hips", r"petite", r"tall",
            r"розмір", r"посад", r"сидить", r"тісн", r"завелик", r"замал", r"коротк", r"довг",
            r"плеч", r"пахв", r"пройм", r"груд", r"талі", r"стег",
        ],
        "material": [
            r"fabric", r"material", r"quality", r"seam", r"stitch", r"lining", r"lace", r"zipper", r"button",
            r"hole", r"tear", r"ripped", r"unravel", r"dye", r"wash", r"pilling", r"shrunk", r"scratchy",
            r"itchy", r"see-through", r"sheer", r"defect",
            r"ткан", r"матеріал", r"якіс", r"шв", r"нитк", r"підклад", r"мереж", r"блискав", r"ґудз",
            r"гудз", r"дір", r"фарбу", r"пран", r"брак", r"пошит",
        ],
        "style": [
            r"style", r"look", r"color", r"print", r"pattern", r"photo", r"picture", r"online",
            r"flattering", r"unflattering", r"ugly", r"pretty", r"cute", r"beautiful", r"design", r"silhouette",
            r"стиль", r"вигляд", r"колір", r"фото", r"онлайн", r"дизайн", r"гарн", r"мил", r"негар", r"не як",
            r"не пас", r"не лест",
        ],
        "overall": [
            r"comfortable", r"comfy", r"wear", r"wearable", r"office", r"church", r"practical", r"versatile",
            r"worth", r"price", r"value", r"expensive", r"overpriced", r"hard to put", r"hard to take",
            r"easy to wear", r"daily", r"everyday",
            r"зруч", r"носити", r"офіс", r"церк", r"ціна", r"варте", r"дорого", r"практич", r"універс",
        ],
    }

    blocks = [count_markers(texts, patterns) for patterns in pats.values()]
    lengths = np.array([[np.log1p(len(str(t))), np.log1p(len(str(t).split()))] for t in texts], dtype=float)
    return np.column_stack(blocks + [lengths])


def align_binary_scores(clf: Any, X: Any) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X).astype(float)
    proba = clf.predict_proba(X)[:, 1]
    return (proba - 0.5).astype(float)


def align_multiclass_scores(clf: Any, X: Any, classes: list[str]) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, classes.index(str(cls))] = scores[:, i]
    return out


def build_text_blocks(train_text: list[str], test_text: list[str]):
    train_title = [extract_title(t) for t in train_text]
    test_title = [extract_title(t) for t in test_text]

    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=22000,
        sublinear_tf=True,
        lowercase=True,
    )
    X_char = vec_char.fit_transform(train_text)
    Xt_char = vec_char.transform(test_text)

    vec_char2 = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        min_df=2,
        max_features=18000,
        sublinear_tf=True,
        lowercase=True,
    )
    X_char2 = vec_char2.fit_transform(train_text)
    Xt_char2 = vec_char2.transform(test_text)

    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=16000,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_word = vec_word.fit_transform(train_text)
    Xt_word = vec_word.transform(test_text)

    vec_title = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
        lowercase=True,
    )
    X_title = vec_title.fit_transform(train_title)
    Xt_title = vec_title.transform(test_title)

    F = rule_features(train_text)
    Ft = rule_features(test_text)
    scaler = StandardScaler()
    F = scaler.fit_transform(F)
    Ft = scaler.transform(Ft)

    X = hstack([X_char, X_char2, X_word, X_title, csr_matrix(F)], format="csr")
    Xt = hstack([Xt_char, Xt_char2, Xt_word, Xt_title, csr_matrix(Ft)], format="csr")
    return X, Xt


def cv_binary_ensemble(X, Xt, y: np.ndarray, seed: int = 42):
    models = [
        ("svc_c05", LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=6000)),
        ("svc_c025", LinearSVC(C=0.25, class_weight="balanced", dual="auto", max_iter=6000)),
        ("svc_c1", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=6000)),
        ("lr_c05", LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", max_iter=5000)),
        ("lr_c1", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000)),
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof = np.zeros((X.shape[0], len(models)))
    test = np.zeros((Xt.shape[0], len(models)))

    for j, (_, model) in enumerate(models):
        for tr, va in skf.split(np.zeros(len(y)), y):
            m = clone(model)
            m.fit(X[tr], y[tr])
            oof[va, j] = align_binary_scores(m, X[va])
            test[:, j] += align_binary_scores(m, Xt) / skf.n_splits

    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(Xt.shape[0])
    for tr, va in skf.split(oof, y):
        sc = StandardScaler()
        A = sc.fit_transform(oof[tr])
        B = sc.transform(oof[va])
        T = sc.transform(test)
        meta = LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", max_iter=5000)
        meta.fit(A, y[tr])
        meta_oof[va] = meta.decision_function(B)
        meta_test += meta.decision_function(T) / skf.n_splits

    thresholds = np.unique(np.quantile(meta_oof, np.linspace(0.03, 0.97, 600)))
    best = {"f1": -1.0, "threshold": 0.0, "positive_count": 0}
    for th in thresholds:
        pred = (meta_oof >= th).astype(int)
        f1 = f1_score(y, pred, average="macro")
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(th), "positive_count": int(pred.sum())}

    pred_test = (meta_test >= best["threshold"]).astype(int)
    return pred_test, meta_oof, meta_test, best


def cv_multiclass_ensemble(X, Xt, y: np.ndarray, classes: list[str], seed: int = 42):
    models = [
        ("svc_c05", LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=7000)),
        ("svc_c025", LinearSVC(C=0.25, class_weight="balanced", dual="auto", max_iter=7000)),
        ("svc_c1", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=7000)),
        ("lr_c05", LogisticRegression(C=0.5, class_weight="balanced", solver="lbfgs", max_iter=6000)),
        ("lr_c1", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=6000)),
        ("nb", ComplementNB(alpha=0.7)),
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # ComplementNB requires non-negative features, so it will be handled on a reduced non-negative view.
    # For simplicity, we skip NB if the matrix contains negative standardized rule features.
    safe_models = [(n, m) for n, m in models if n != "nb"]

    oof = np.zeros((X.shape[0], len(classes), len(safe_models)))
    test = np.zeros((Xt.shape[0], len(classes), len(safe_models)))

    for j, (_, model) in enumerate(safe_models):
        for tr, va in skf.split(np.zeros(len(y)), y):
            m = clone(model)
            m.fit(X[tr], y[tr])
            oof[va, :, j] = align_multiclass_scores(m, X[va], classes)
            test[:, :, j] += align_multiclass_scores(m, Xt, classes) / skf.n_splits

    oof_avg = oof.mean(axis=2)
    test_avg = test.mean(axis=2)

    pred_oof = np.array([classes[i] for i in oof_avg.argmax(axis=1)])
    pred_test = np.array([classes[i] for i in test_avg.argmax(axis=1)])
    cv = float(f1_score(y, pred_oof, average="macro"))
    return pred_test, pred_oof, oof_avg, test_avg, cv


def validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    assert list(sub.columns) == ["id", "recommended", "reviewer_segment"]
    assert len(sub) == len(test)
    assert sub["id"].astype(str).tolist() == test["id"].astype(str).tolist()
    assert set(sub["recommended"].unique()).issubset({0, 1})
    assert set(sub["reviewer_segment"].unique()).issubset(set(ALLOWED_SEGMENTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--out", default="outputs/submission_no_openai_ensemble.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X_text = train["content"].astype(str).tolist()
    Xt_text = test["content"].astype(str).tolist()
    X, Xt = build_text_blocks(X_text, Xt_text)

    y_rec = train["recommended"].astype(int).to_numpy()
    y_seg = train["reviewer_segment"].astype(str).to_numpy()

    rec_pred, rec_oof_score, rec_test_score, rec_report = cv_binary_ensemble(X, Xt, y_rec)
    seg_pred, seg_oof_pred, seg_oof_scores, seg_test_scores, seg_cv = cv_multiclass_ensemble(X, Xt, y_seg, ALLOWED_SEGMENTS)

    submission = pd.DataFrame({
        "id": test["id"].astype(str),
        "recommended": rec_pred.astype(int),
        "reviewer_segment": seg_pred.astype(str),
    })
    validate_submission(submission, test)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    audit = test[["id", "content"]].copy()
    audit["recommended"] = submission["recommended"]
    audit["reviewer_segment"] = submission["reviewer_segment"]
    audit["recommended_score"] = rec_test_score
    for i, cls in enumerate(ALLOWED_SEGMENTS):
        audit[f"segment_score_{cls}"] = seg_test_scores[:, i]
    audit.to_csv("outputs/audit_no_openai_ensemble.csv", index=False)

    report = {
        "approach": "no_openai_classical_ml_ensemble",
        "recommended_cv_macro_f1": rec_report["f1"],
        "recommended_threshold": rec_report["threshold"],
        "recommended_oof_positive_count": rec_report["positive_count"],
        "test_recommended_counts": submission["recommended"].value_counts().to_dict(),
        "segment_cv_macro_f1": seg_cv,
        "test_segment_counts": submission["reviewer_segment"].value_counts().to_dict(),
        "note": "No OpenAI/API calls. Uses all 400 train rows.",
    }
    Path("outputs/no_openai_ensemble_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
