# CSE151Project Part3
## 1.1 Load, de-duplicate, and target inference
- **Load**: `pd.read_csv('/content/data.csv')` link for downoload dataset(i miss this in p2): https://drive.google.com/file/d/1LDS1KWr3CL4DPNiDOkBB0IVSiXELVpv_/view?usp=sharing
- **De-duplicate**: `df.drop_duplicates()` to remove exact duplicate rows so they don’t bias the fit.
- **Target/label inference**: If a canonical label column isn’t provided, we scan for common names (e.g., `label`, `target`, `class`) and, failing that, pick a **low-cardinality** column (2–min(50, 20% of n)) as the label. This makes the notebook plug‑and‑play across datasets without manual edits.

## 1.2 Feature typing (categorical vs. numeric)
I derive two disjoint sets:
- **Categorical columns (`cat_cols`)**:
  1) columns with `object` dtype,
  2) *low-cardinality* numeric columns (≤5 unique values) which behave like categories (e.g., codes, bins),
  3) a small set of **name‑hints** for opcode-like columns (e.g., `asm_commands_*`).
- **Numeric columns (`num_cols`)**: everything else.

This conservative typing ensures that enumerations and codes are **one‑hot encoded** rather than treated as continuous magnitudes.

## 1.3 Imputation (handle missing values)
- **Numeric**: `SimpleImputer(strategy="median")` — robust to outliers and preserves central tendency.
- **Categorical**: `SimpleImputer(strategy="most_frequent")` — replaces missing categories with the mode, keeping the feature usable for one‑hot encoding.

All imputation happens **inside the pipeline**, so it is **fit only on training folds** (no leakage).

## 1.4 Targeted nonlinearity / feature expansion (numeric)
We create a **FeatureUnion** with two branches over numeric inputs:
- **Identity pass-through** (keeps the original numeric features).
- **Targeted log transform** on *heavy* features: we mark a numeric column as *heavy* if `max(value) > 100` **or** `nunique > 50`. Those columns get `log1p(x) = log(1 + x)`.

**Why log1p?**
- Tames extreme right-skew and large dynamic ranges.
- Keeps zeros well-defined (`log1p(0)=0`).
- Preserves rank/order while compressing very large values.

This produces an **expanded numeric set**: original + log1p‑transformed heavy columns.

## 1.5 Non-negativity for χ² scoring
Before χ² feature selection, we **clip numeric values to ≥ 0**:
> `clip_nonneg(X) = max(X, 0)`

**Rationale:** `SelectKBest(chi2)` requires **non-negative** features; clipping guarantees validity for any numeric that may be negative while retaining signal for count-like or magnitude features.

## 1.6 Scaling
Two complementary scalers are used at different stages:
- **Min–Max scaling (per numeric feature)** to `[0, 1]` *inside* the numeric pipeline. This normalizes ranges across original and log1p features and improves χ² comparability.
- **Standardization (after column-wise union & selection)**: `StandardScaler(with_mean=True, with_std=True)` centers/scales **the selected full design matrix** (numeric + one‑hot) to mean 0, unit variance — a good default for RBF SVMs.

> Notes:
> - One‑hot features (0/1) also get standardized at the end; this can help kernels treat dense and sparse blocks more uniformly.
> - The order is **(impute → clip → expand → MinMax)** within numeric, **then one‑hot** for categorical, **then χ² selection**, **then StandardScaler**.

## 1.7 Categorical encoding
`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` encodes each category into its own binary column while **safely ignoring unseen categories** at test time. We set `sparse_output=False` to work seamlessly with later dense transforms.

Low‑cardinality numeric codes are intentionally treated as categorical here to avoid imposing false numeric distances.

## 1.8 Feature selection (dimensionality & noise control)
We apply `SelectKBest(chi2)` with:
```
k = min(n_features, max(10, int(0.9 * n_features)))
```
i.e., **keep up to 90%** of features (but at least 10). χ² ranks features by their dependence with the label, removing weak/noisy columns and reducing overfitting and compute cost. Because it’s part of the pipeline, the selection is **re-fit each CV fold** (no leakage).

## 1.9 Leakage-safe orchestration
All steps above are wrapped in a `ColumnTransformer` and `Pipeline`, and hyperparameter search uses cross‑validation. That guarantees:
- **Imputation, scaling, encoding, log1p, and χ² selection are learned only from training folds**.
- The exact same transformations are applied to validation/test sets.


## 2: Preprocessing code + Train code + Result

```python
!pip -q install scikit-learn pandas numpy

import numpy as np, pandas as pd, os, joblib
from google.colab import files

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

df = pd.read_csv('/content/data.csv')
df = df.drop_duplicates()

LABEL_COL = None
label_like = {"label","labels","target","class","classes","category","y","outcome","diagnosis"}
for c in df.columns:
    if c.strip().lower() in label_like:
        LABEL_COL = c
        break
if LABEL_COL is None:
    n = len(df)
    for c in df.columns:
        nu = df[c].nunique(dropna=False)
        if 2 <= nu <= min(50, max(2, int(0.2*n))):
            LABEL_COL = c
            break
if LABEL_COL is None and "Class" in df.columns:
    LABEL_COL = "Class"

feat_cols = [c for c in df.columns if c != LABEL_COL]

cat_name_hints = {
    "asm_commands_cmc", "asm_commands_cwd", "asm_commands_faddp", "asm_commands_fchs",
    "asm_commands_fdiv", "asm_commands_fdivr", "asm_commands_fistp", "asm_commands_jno",
    "asm_commands_outs", "asm_commands_rcr", "asm_commands_sal", "asm_commands_scas",
    "asm_commands_sidt"
}

hint_cols_present = [c for c in feat_cols if c in cat_name_hints]
obj_cols = [c for c in feat_cols if df[c].dtype == "object"]
low_card_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 5]

cat_cols = sorted(set(hint_cols_present + obj_cols + low_card_cols))
num_cols = [c for c in feat_cols if c not in cat_cols]

print(f"[Info] Using {len(cat_cols)} categorical columns (encoded) and {len(num_cols)} numeric columns.")

y = df[LABEL_COL].astype(str).to_numpy()

def clip_nonneg(X):
    return np.maximum(X, 0)

if len(num_cols) > 0:
    feat_meta = df[num_cols]
    heavy_mask = (feat_meta.max() > 100) | (feat_meta.nunique() > 50)
    heavy_idx = np.where(heavy_mask.values)[0] 
else:
    heavy_idx = np.array([], dtype=int)

def log1p_selected(Z, idx):
    Z2 = np.zeros_like(Z, dtype=float)
    if len(idx):
        Z2[:, idx] = np.log1p(Z[:, idx])
    return Z2

num_expander = FeatureUnion([
    ("id",    FunctionTransformer(lambda Z: Z, accept_sparse=False)),
    ("log1p", FunctionTransformer(log1p_selected, kw_args={"idx": heavy_idx}, accept_sparse=False)),
])

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clip",   FunctionTransformer(clip_nonneg, accept_sparse=False)),
    ("expand", num_expander),
    ("minmax", MinMaxScaler()), 
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if num_cols:
    transformers.append(("num", numeric_pipe, num_cols))
if cat_cols:
    transformers.append(("cat", categorical_pipe, cat_cols))

pre = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
    sparse_threshold=0.0 
)

X = df[feat_cols].to_numpy()
n_features = X.shape[1]
k = min(n_features, max(10, int(0.9 * n_features)))  

pipe = Pipeline([
    ("pre", pre),
    ("chi2", SelectKBest(chi2, k=k)),
    ("std", StandardScaler(with_mean=True, with_std=True)),
    ("svc", SVC(kernel="rbf", class_weight="balanced"))
])

Xtr, Xte, ytr, yte = train_test_split(df[feat_cols], y, test_size=0.2, stratify=y, random_state=42)

param_dist = {
    "svc__C":     np.logspace(-1, 3, 30), 
    "svc__gamma": np.logspace(-6, 0, 30), 
}
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=25,
    scoring="f1_macro",
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)
search.fit(Xtr, ytr)

best = search.best_estimator_
yp = best.predict(Xte)
acc = accuracy_score(yte, yp)
f1m = f1_score(yte, yp, average="macro")
print("Categorical columns encoded:", cat_cols)
print("Best params:", search.best_params_)
print(f"Test Accuracy: {acc:.4f}  |  Macro-F1: {f1m:.4f}")
print("\nClassification report:\n", classification_report(yte, yp, zero_division=0))

Test Accuracy: 0.9829  |  Macro-F1: 0.9744

Classification report:
              precision    recall  f1-score   support

          1       0.97      0.97      0.97       301
          2       1.00      0.99      0.99       496
          3       0.99      0.99      0.99       589
          4       1.00      0.98      0.99        95
          5       0.89      1.00      0.94         8
          6       0.93      0.99      0.96       144
          7       0.94      0.99      0.96        79
          8       0.98      0.96      0.97       245
          9       0.99      1.00      0.99       201

   accuracy                           0.98      2158
  macro avg       0.97      0.98      0.97      2158
weighted avg       0.98      0.98      0.98      2158

from sklearn.metrics import accuracy_score
yp_tr = best.predict(Xtr)
yp_te = best.predict(Xte)

train_err = 1.0 - accuracy_score(ytr, yp_tr)
test_err  = 1.0 - accuracy_score(yte, yp_te)

print(f"Train Error: {train_err:.4f}")
print(f"Test  Error: {test_err:.4f}")

Train Error: 0.0058
Test  Error: 0.0171

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def plot_cm_log(cm_df, title):
    labels = cm_df.index.astype(str).tolist()
    cm = cm_df.values.astype(float)

    vmax = float(cm.max())
    if vmax <= 1:
        vmax = 2.0

    cm_masked = np.ma.masked_less_equal(cm, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    im = ax.imshow(cm_masked, norm=LogNorm(vmin=1, vmax=vmax))
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count (log)")

    half = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] >= half else "black")

    ax.set_xticks(np.arange(-.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.4)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig

os.makedirs("artifacts", exist_ok=True)
plot_cm_log(cm_train, "Confusion Matrix (TRAIN) — log scale").savefig(
    "artifacts/confusion_matrix_train_log.png", dpi=180, bbox_inches="tight"
)
plot_cm_log(cm_test, "Confusion Matrix (TEST) — log scale").savefig(
    "artifacts/confusion_matrix_test_log.png", dpi=180, bbox_inches="tight"
)


