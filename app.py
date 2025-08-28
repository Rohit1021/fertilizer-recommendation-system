from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle, json, os, sys

with open("schema.json", "r") as f:
    SCHEMA = json.load(f)

FEATURE_COLS   = SCHEMA.get("feature_cols", [])
CAT_COLS       = SCHEMA.get("categorical_cols", [])
NUM_COLS       = SCHEMA.get("numeric_cols", [])
CAT_OPTIONS    = SCHEMA.get("categorical_options", {})
NUM_MEDIANS    = SCHEMA.get("numeric_medians", {})
TARGET_CLASSES = SCHEMA.get("target_classes")

print("[APP] feature_cols:", len(FEATURE_COLS), "example:", FEATURE_COLS[:6])
print("[APP] categorical_cols:", CAT_COLS)
print("[APP] numeric_cols:", NUM_COLS)
print("[APP] categorical_options keys:", list(CAT_OPTIONS.keys())[:10])
print("[APP] numeric_medians keys:", list(NUM_MEDIANS.keys())[:10])
print("[APP] has target_classes:", bool(TARGET_CLASSES))

if (not CAT_OPTIONS) and os.path.exists("label_encoders.pkl"):
    try:
        with open("label_encoders.pkl", "rb") as f:
            encs = pickle.load(f)
        for c in CAT_COLS:
            if c in encs:
                CAT_OPTIONS[c] = list(map(str, encs[c].classes_.tolist()))
        print("[APP] Built categorical_options from label_encoders.pkl")
    except Exception as e:
        print("[APP] Warning: could not load label_encoders.pkl:", e, file=sys.stderr)

if TARGET_CLASSES is None and os.path.exists("target_encoder.pkl"):
    try:
        with open("target_encoder.pkl", "rb") as f:
            target_le = pickle.load(f)
        TARGET_CLASSES = target_le.classes_.tolist()
        print("[APP] Loaded target_classes from target_encoder.pkl")
    except Exception as e:
        print("[APP] Warning: could not load target_encoder.pkl:", e, file=sys.stderr)

ENC_MAPS = {c: {v: i for i, v in enumerate(CAT_OPTIONS.get(c, []))} for c in CAT_COLS}

with open("fertilizer_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

print("[APP] model classes_:", getattr(MODEL, "classes_", None))

app = Flask(__name__)

def parse_float(value):
    """Friendly float parser that accepts commas/blanks."""
    s = str(value).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def preprocess_form(form):
    row = {}

    for c in CAT_COLS:
        selected = form.get(c, "")
        opts = CAT_OPTIONS.get(c, [])
        if not opts:
            row[c] = 0
        else:
            if selected not in ENC_MAPS[c]:
                selected = opts[0]
            row[c] = ENC_MAPS[c][selected]

    for n in NUM_COLS:
        raw = form.get(n, "")
        val = parse_float(raw)
        if val is None or str(raw).strip() == "":
            val = float(NUM_MEDIANS.get(n, 0.0))
        row[n] = val

    Xrow = pd.DataFrame([[row.get(col, np.nan) for col in FEATURE_COLS]], columns=FEATURE_COLS)

    for c in CAT_COLS:
        Xrow[c] = pd.to_numeric(Xrow[c], errors="coerce").fillna(0).astype("int32")
    for n in NUM_COLS:
        Xrow[n] = pd.to_numeric(Xrow[n], errors="coerce").fillna(NUM_MEDIANS.get(n, 0.0)).astype("float32")

    return Xrow

def top3_from_proba(proba_1d):
    pos = np.argsort(proba_1d)[::-1][:3]

    if hasattr(MODEL, "classes_"):
        class_ids = MODEL.classes_
    else:
        class_ids = np.arange(len(proba_1d))

    labels, scores = [], []
    for p in pos:
        cls_id = int(class_ids[p])
        if TARGET_CLASSES and 0 <= cls_id < len(TARGET_CLASSES):
            name = TARGET_CLASSES[cls_id]
        else:
            name = str(cls_id)
        labels.append(name)
        scores.append(float(proba_1d[p]))
    return list(zip(labels, scores))

@app.route("/", methods=["GET", "POST"])
def index():
    top3 = None
    inputs = {}

    if request.method == "POST":
        for c in CAT_COLS:
            inputs[c] = request.form.get(c, "")
        for n in NUM_COLS:
            inputs[n] = request.form.get(n, "")

        Xrow = preprocess_form(request.form)
        proba = MODEL.predict_proba(Xrow)[0]
        top3 = top3_from_proba(proba)

    return render_template(
        "index.html",
        feature_cols=FEATURE_COLS,
        cat_cols=CAT_COLS,
        num_cols=NUM_COLS,
        cat_options=CAT_OPTIONS,
        top3=top3,
        inputs=inputs
    )

if __name__ == "__main__":
    app.run(debug=True)
