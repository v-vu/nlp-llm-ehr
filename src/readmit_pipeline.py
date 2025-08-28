from __future__ import annotations
import os, re, json, warnings
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI  # NEW
client = OpenAI(api_key="sk-...your key...")  # for quick local testing onl

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import joblib

ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


USE_LLM = False  # turn on/off LLM extraction

LLM_SYSTEM_PROMPT = """You are a medical NLP assistant. Extract only the requested fields in strict JSON.
If evidence is absent, set booleans False and categorical fields to "unknown".
"""

def build_llm_prompt(note: str) -> str:
    schema_example = {
        "mrd_status": "positive|negative|indeterminate|unknown",
        "blasts_percent": 12.0,
        "neutropenic_fever": False,
        "sepsis": False,
        "mucositis": False,
        "central_line": True,
        "discharge_disposition": "home|home_with_services|snf|ltach|hospice|rehab|other",
        "chemo_intensity": "high|standard|low|unknown",
        "planned_readmission": False,
        "followup_within_7d": True,
        "social_support": "strong|limited|none|unknown",
        "med_nonadherence": False
    }
    return (
        f"{LLM_SYSTEM_PROMPT}\n"
        f"FIELDS_SCHEMA = {json.dumps(schema_example, indent=2)}\n\n"
        "TASK: Extract these fields from the clinical note below.\n"
        "Return ONLY JSON with keys exactly matching FIELDS_SCHEMA.\n\n"
        "NOTE:\n"
        f"{note}\n"
    )

def llm_extract(note: str) -> Dict[str, Any]:
    """
    Use OpenAI (Chat Completions) to extract fields defined in EXTRACT_FIELDS.
    Returns a dict keyed by EXTRACT_FIELDS with raw values (will be coerced later).
    Raises on transport/JSON errors so caller can fallback to regex.
    """
    # Light PHI scrub before sending (your pipeline also deidentifies upstream where used)
    def deidentify_text_local(t: str) -> str:
        t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " <DATE> ", t)
        t = re.sub(r"\b\d{7,}\b", " <ID> ", t)
        t = re.sub(r"\b[A-Z][a-z]+,?\s+[A-Z][a-z]+\b", " <NAME> ", t)
        return t

    safe_note = deidentify_text_local(note if isinstance(note, str) else "")
    # Keep payload small if needed
    safe_note = safe_note[:8000]

    # Define the schema (must match EXTRACT_FIELDS keys)
    schema_example = {
        "mrd_status": "positive|negative|indeterminate|unknown",
        "blasts_percent": 12.0,  # float
        "neutropenic_fever": False,
        "sepsis": False,
        "mucositis": False,
        "central_line": True,
        "discharge_disposition": "home|home_with_services|snf|ltach|hospice|rehab|other",
        "chemo_intensity": "high|standard|low|unknown",
        "planned_readmission": False,
        "followup_within_7d": True,
        "social_support": "strong|limited|none|unknown",
        "med_nonadherence": False
    }

    system_msg = (
        "You are a medical NLP assistant. Extract ONLY the requested fields as strict JSON. "
        "If evidence is absent, set booleans False and categorical fields to 'unknown'. "
        "Do NOT include explanations—return a single JSON object only."
    )

    user_msg = (
        "FIELDS_SCHEMA = " + json.dumps(schema_example, indent=2) + "\n\n"
        "TASK: Extract these fields from the clinical note below. "
        "Return ONLY JSON with keys exactly matching FIELDS_SCHEMA.\n\n"
        "NOTE:\n" + safe_note
    )

    # Force a JSON object back
    resp = client.chat.completions.create(
        model="gpt-4o-mini",             # or "gpt-4o" if preferred
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content
    data = json.loads(content)  # raise if not valid JSON

    # Keep only expected keys; let caller coerce + fill defaults
    cleaned = {k: data.get(k, None) for k in EXTRACT_FIELDS.keys()}
    return cleaned


EXTRACT_FIELDS = {
    "mrd_status": ["positive", "negative", "indeterminate", "unknown"],
    "blasts_percent": "float",
    "neutropenic_fever": "bool",
    "sepsis": "bool",
    "mucositis": "bool",
    "central_line": "bool",
    "discharge_disposition": ["home", "home_with_services", "snf", "ltach", "hospice", "rehab", "other"],
    "chemo_intensity": ["high", "standard", "low", "unknown"],
    "planned_readmission": "bool",
    "followup_within_7d": "bool",
    "social_support": ["strong", "limited", "none", "unknown"],
    "med_nonadherence": "bool"
}

STRUCTURED_NUMERIC = ["anc","wbc","hemoglobin","platelets","creatinine","alt","ast","los_days","prior_30d_admits","transfusion_count","infection_count","age"]
STRUCTURED_CATEGORICAL = ["sex","race","ethnicity","insurance","chemo_regimen","discharge_dayofweek"]
TEXT_COL = "note_text"
ID_COL = "patient_id"
LABEL_COL = "readmit_30d"

def deidentify_text(text: str) -> str:
    if not isinstance(text, str): return ""
    import re
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " <DATE> ", text)
    t = re.sub(r"\b\d{7,}\b", " <ID> ", t)
    t = re.sub(r"\b[A-Z][a-z]+,?\s+[A-Z][a-z]+\b", " <NAME> ", t)
    return t

def aggregate_notes(notes_df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    notes_df[text_col] = notes_df[text_col].fillna("").astype(str).apply(deidentify_text)
    agg = (notes_df.groupby(id_col)[text_col]
           .apply(lambda s: "\n\n".join(s.tolist()))
           .reset_index())
    return agg

def regex_fallback_extract(note: str) -> Dict[str, Any]:
    import re
    text = note.lower()
    def has(*terms): return any(t in text for t in terms)
    blasts = None
    m = re.search(r"blasts?\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*%", text)
    if m: blasts = float(m.group(1))
    chemo_intensity = "unknown"
    if has("hyper-cvad","cytarabine high dose","hi-dac","intensive induction","7+3"): chemo_intensity = "high"
    elif has("consolidation","maintenance","standard dose"): chemo_intensity = "standard"
    elif has("low intensity","reduced intensity"): chemo_intensity = "low"
    disposition = "unknown"
    if has("discharged home with services","home health","home infusion"): disposition = "home_with_services"
    elif has("discharged home"): disposition = "home"
    elif has("skilled nursing","snf"): disposition = "snf"
    elif has("ltach"): disposition = "ltach"
    elif has("hospice"): disposition = "hospice"
    elif has("rehab","inpatient rehab"): disposition = "rehab"
    else: disposition = "other" if has("discharged") else "unknown"
    out = {
        "mrd_status": ("positive" if has("mrd positive","mrd+") else
                       "negative" if has("mrd negative","no mrd") else
                       "indeterminate" if has("mrd indeterminate") else
                       "unknown"),
        "blasts_percent": blasts if blasts is not None else np.nan,
        "neutropenic_fever": has("neutropenic fever","febrile neutropenia"),
        "sepsis": has("sepsis","septic shock","bacteremia with hypotension"),
        "mucositis": has("mucositis","stomatitis","oral mucositis"),
        "central_line": has("port-a-cath","picc","central line","mediport"),
        "discharge_disposition": disposition,
        "chemo_intensity": chemo_intensity,
        "planned_readmission": has("planned readmission","scheduled chemo readmission"),
        "followup_within_7d": has("follow up in 7 days","follow-up within 1 week","clinic visit next week"),
        "social_support": ("strong" if has("family support","lives with spouse","caregiver available") else
                           "limited" if has("limited support","lives alone","transportation issues") else
                           "unknown"),
        "med_nonadherence": has("nonadherence","non-adherence","missed doses")
    }
    return out

"""def extract_features_from_notes(notes_df: pd.DataFrame, text_col: str, id_col: str) -> pd.DataFrame:
    rows = []
    for pid, text in zip(notes_df[id_col].tolist(), notes_df[text_col].tolist()):
        feat = regex_fallback_extract(text)
        feat[id_col] = pid
        rows.append(feat)
    feats = pd.DataFrame(rows)
    for k, v in EXTRACT_FIELDS.items():
        if v == "bool":
            feats[k] = feats[k].astype(bool).astype(int)
        elif v == "float":
            feats[k] = pd.to_numeric(feats[k], errors="coerce")
        elif isinstance(v, list):
            feats[k] = feats[k].astype(str)
    return feats"""

def extract_features_from_notes(notes_df: pd.DataFrame, text_col: str, id_col: str) -> pd.DataFrame:
    """
    If USE_LLM is True, attempt LLM-based extraction; on any error or invalid
    payload, fall back to regex_fallback_extract(). Ensures all fields in
    EXTRACT_FIELDS exist and coerces final dtypes.
    """
    rows = []

    for pid, raw_text in zip(notes_df[id_col].tolist(), notes_df[text_col].tolist()):
        text = "" if pd.isna(raw_text) else str(raw_text)

        # Try LLM if enabled; otherwise use regex extractor
        feat = None
        if USE_LLM:
            try:
                llm_out = llm_extract(text)  # must return a dict (or JSON string) matching EXTRACT_FIELDS
                if isinstance(llm_out, str):
                    # allow LLMs that return JSON string
                    llm_out = json.loads(llm_out)
                if isinstance(llm_out, dict):
                    feat = llm_out
            except Exception:
                feat = None  # fall back below

        if feat is None:
            feat = regex_fallback_extract(text)

        # Ensure all expected keys exist; fill sensible defaults
        complete = {}
        for k, spec in EXTRACT_FIELDS.items():
            if k in feat:
                complete[k] = feat[k]
            else:
                if spec == "bool":
                    complete[k] = 0
                elif spec == "float":
                    complete[k] = np.nan
                elif isinstance(spec, list):
                    # default to "unknown" if allowed, else first option
                    complete[k] = "unknown" if "unknown" in spec else spec[0]
                else:
                    complete[k] = None

        complete[id_col] = pid
        rows.append(complete)

    feats = pd.DataFrame(rows)

    # Coerce final dtypes
    for k, v in EXTRACT_FIELDS.items():
        if v == "bool":
            feats[k] = feats[k].astype(bool).astype(int)
        elif v == "float":
            feats[k] = pd.to_numeric(feats[k], errors="coerce")
        elif isinstance(v, list):
            feats[k] = feats[k].astype(str)

    return feats


def build_dataset(structured_csv: str, notes_csv: str,
                  id_col: str = "patient_id", label_col: str = "readmit_30d",
                  text_col: str = "note_text") -> Tuple[pd.DataFrame, pd.Series]:
    df_struct = pd.read_csv(structured_csv)
    df_notes = pd.read_csv(notes_csv)
    df_notes_agg = aggregate_notes(df_notes, id_col=id_col, text_col=text_col)
    df_extracted = extract_features_from_notes(df_notes_agg, text_col=text_col, id_col=id_col)
    df = (df_struct.merge(df_notes_agg, on=id_col, how="left")
                   .merge(df_extracted, on=id_col, how="left"))
    y = df[label_col].astype(int)
    return df.drop(columns=[label_col]), y

from sklearn.utils import estimator_checks  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

def build_model_pipeline(numeric_cols: List[str], categorical_cols: List[str], text_col: str = "note_text") -> Pipeline:
    extr_num = [k for k, v in EXTRACT_FIELDS.items() if v in ("float", "bool")]
    extr_cat = [k for k, v in EXTRACT_FIELDS.items() if isinstance(v, list)]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    text_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), token_pattern=r"(?u)\b[A-Za-z][A-Za-z+\-/\.%]*\b", min_df=1))])
    extr_num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler(with_mean=False))])
    extr_cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
            ("text", text_pipe, text_col),
            ("extr_num", extr_num_pipe, extr_num),
            ("extr_cat", extr_cat_pipe, extr_cat),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver="saga", penalty="l2", max_iter=2000, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
import numpy as np, json, os, joblib

def evaluate_cv(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline, n_splits: int = 2) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, auprcs, briers, thresholds = [], [], [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        model = pipeline.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:,1]
        aucs.append(roc_auc_score(yva, p))
        auprcs.append(average_precision_score(yva, p))
        briers.append(brier_score_loss(yva, p))
        fpr, tpr, thr = roc_curve(yva, p)
        j = tpr - fpr
        thresholds.append(float(thr[np.argmax(j)]))
        print(f"[Fold {fold}] AUROC={aucs[-1]:.3f} AUPRC={auprcs[-1]:.3f} Brier={briers[-1]:.3f}")
    return {
        "cv_auroc_mean": float(np.mean(aucs)),
        "cv_auroc_std": float(np.std(aucs)),
        "cv_auprc_mean": float(np.mean(auprcs)),
        "cv_auprc_std": float(np.std(auprcs)),
        "cv_brier_mean": float(np.mean(briers)),
        "cv_brier_std": float(np.std(briers)),
        "suggested_threshold": float(np.median(thresholds))
    }

def train_and_save(structured_csv: str, notes_csv: str, out_dir: str = ARTIFACT_DIR,
                   numeric_cols: List[str] = None, categorical_cols: List[str] = None,
                   text_col: str = "note_text") -> dict:
    if numeric_cols is None: numeric_cols = STRUCTURED_NUMERIC
    if categorical_cols is None: categorical_cols = STRUCTURED_CATEGORICAL
    X, y = build_dataset(structured_csv, notes_csv, text_col=text_col)
    for c in numeric_cols + categorical_cols + [text_col] + list(EXTRACT_FIELDS.keys()):
        if c not in X.columns:
            print(f"[WARN] Column missing in data: {c}")
    pipe = build_model_pipeline(numeric_cols, categorical_cols, text_col=text_col)
    print("== Cross-validation ==")
    cv_stats = evaluate_cv(X, y, pipe, n_splits=2)
    print(json.dumps(cv_stats, indent=2))
    model = pipe.fit(X, y)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(model, model_path)
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "text_col": text_col,
        "extracted_fields": list(EXTRACT_FIELDS.keys()),
        "cv_stats": cv_stats
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved model to {model_path}")
    return {"model_path": model_path, "metadata": metadata}

def load_model(artifact_dir: str = ARTIFACT_DIR):
    model = joblib.load(os.path.join(artifact_dir, "model.joblib"))
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        md = json.load(f)
    return {"model": model, "metadata": md}

def predict_one(example_struct_row: Dict[str, Any], notes_text: str, artifact_dir: str = ARTIFACT_DIR) -> dict:
    loaded = load_model(artifact_dir)
    model = loaded["model"]
    md = loaded["metadata"]
    row = {**example_struct_row}
    row["patient_id"] = "inference_1"
    row["note_text"] = deidentify_text(notes_text)
    #extr = regex_fallback_extract(row["note_text"])
    extr = llm_extract(row["note_text"]) if USE_LLM else regex_fallback_extract(row["note_text"])
    for k in md["extracted_fields"]:
        row[k] = extr.get(k, np.nan)
    X = pd.DataFrame([row])
    for c in (md["numeric_cols"] + md["categorical_cols"] + [md["text_col"]] + md["extracted_fields"]):
        if c not in X.columns:
            X[c] = np.nan
    prob = float(model.predict_proba(X)[:,1][0])
    thr = md["cv_stats"]["suggested_threshold"]
    label = int(prob >= thr)
    return {"prob_readmit_30d": prob, "label_at_thr": label, "threshold": thr}
