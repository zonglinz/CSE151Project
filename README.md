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


## 2.1 Preprocessing code + Train code + Result

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

Train Accuracy: 0.9942  |  Macro-F1: 0.9829
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

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

yp_val = cross_val_predict(search.best_estimator_, Xtr, ytr, cv=cv, method='predict', n_jobs=-1)
yp_te  = best.predict(Xte)

val_err = 1.0 - accuracy_score(ytr, yp_val)
test_err = 1.0 - accuracy_score(yte, yp_te)

print(f"Validation Error: {val_err:.4f}")
print(f"Test Error: {test_err:.4f}")
Validation Error: 0.0239
Test Error: 0.0171

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
```
![Confusion Matrix Train](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-16.png?raw=true)
![Confusion Matrix Test](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-15.png?raw=true)

## 2.2 Result(something i want to say)
- **Generalization gap:** ~**1.13%** (0.9942 → 0.9829). This is small, indicating **low overfitting** and a well-regularized boundary for the chosen `C, γ`.  
- **Macro-F1 vs Weighted-F1:** Macro-F1 **0.9744** (treats classes equally) vs weighted avg (≈ overall **0.98**). The slightly lower macro-F1 reflects performance on **minority classes** (esp. Class 5 with only 8 samples).

### B. Class-wise Behavior (from the test classification report)
- **High-support classes (2, 3, 9):** All near-perfect (F1 ≈ 0.99–1.00), showing the model captures dominant family patterns very well.  
- **Moderate classes (1, 4, 6, 7, 8):** F1 in **0.96–0.99** range.  
- **Minor class (5, n=8):** **Recall 1.00**, **Precision 0.89**, **F1 0.94**.  
  - *Interpretation:* The model **finds every true Class-5 sample** (no false negatives), but a few **false positives** get labeled as Class-5. On an imbalanced task this is an acceptable trade-off if missing Class-5 is costly; otherwise, we can tune thresholds (see below).

### C. Error Patterns & What They Likely Mean
- **Precision-recall trade-off for Class-5:** High recall + lower precision ⇒ the decision boundary is **generous** toward Class-5. Likely some feature combinations (e.g., unusual control-flow or arithmetic counts) overlap with rare families.  
- **Class-8 slightly lower recall (0.96):** A few Class-8 samples likely sit near boundaries shared with classes 1/4/7 (families with similar instruction “texture”).  
- **Overall:** Errors are **sparse** and concentrated in **minority, edge-case** regions—consistent with high macro-F1.

### D. Bias–Variance Perspective
- **`γ ≈ 0.0053`** (post-standardization) is **not** extreme; it yields a **smooth** RBF surface rather than a wiggly boundary.  
- **`C ≈ 41.8`** enforces **moderate margin hardness**—enough to fit complex structure, but your small generalization gap suggests **variance is under control**.  
- With χ² keeping ≈90% of features and targeted `log1p`, the model has **ample capacity** without drifting into overfit; the data seems linearly separable in RBF space with a clean margin.

### E. Reliability & Robustness Checks (recommended)
- **Variance of estimates:** 2-fold CV is efficient but has **higher variance**. To report stability, repeat the search with a different `random_state` or use **RepeatedStratifiedKFold** (e.g., 2 folds × 3 repeats) just to measure spread in CV F1 (you can keep the same final model).  
- **Significance vs other models:** If you compared multiple finalists, run a **McNemar test** on the *same* test split to confirm any small differences are statistically meaningful.

### F. Thresholds, Calibration & Operating Points
- If false positives on Class-5 are costly, calibrate and adjust decision thresholds:
  - Fit `CalibratedClassifierCV(best_estimator, method="sigmoid", cv="prefit")` (or refit SVC with `probability=True`) and set **class-specific thresholds** to balance precision/recall per your costs.
- **Macro-F1** is strong already; calibration mainly helps **decision control** rather than macro-F1 itself.

### G. What Drove Performance Here
- **Preprocessing:**  
  - Median imputation prevents losing rows;  
  - **Clip + MinMax** guaranteed **non-negative** features for χ²;  
  - **Targeted `log1p`** on high-range/high-card columns stabilized scale and added nonlinear signal **without O(p²)** blow-up;  
  - χ² at ~90% preserved most discriminative features;  
  - Final **standardization** suited the SVM margin geometry.
- **Modeling:** A **balanced RBF-SVM** is a strong baseline for this domain; your search found a **smooth yet expressive** operating point.


## 3 compare with baseline model
## This is my baseline model(Has no separating Numeric and Categorical features and hot encoding) code with classification test
```python
!pip -q install scikit-learn pandas numpy

import io, numpy as np, pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib



df = pd.read_csv('/content/data.csv')

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
        if 2 <= nu <= min(50, max(2,int(0.2*n))):
            LABEL_COL = c
            break
if LABEL_COL is None:
    raise ValueError("Couldn't infer label column. Please set LABEL_COL to the correct column name.")

feat_cols = [c for c in df.columns if c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])]
assert feat_cols, "No numeric features found."
X = df[feat_cols].to_numpy()
y = df[LABEL_COL].astype(str).to_numpy()

def clip_nonneg(X):
    return np.maximum(X, 0)

n_features = X.shape[1]
k = min(n_features, max(10, int(0.9*n_features)))

pipe = Pipeline([
    ("clip", FunctionTransformer(clip_nonneg, accept_sparse=False)),
    ("minmax", MinMaxScaler()),
    ("chi2", SelectKBest(chi2, k=k)),
    ("std", StandardScaler(with_mean=True, with_std=True)),
    ("svc", SVC(kernel="rbf", class_weight="balanced"))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

param_dist = {
    "svc__C": np.logspace(-1, 3, 30),    
    "svc__gamma": np.logspace(-6, 0, 30),  
}
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=25, scoring="f1_macro",
    n_jobs=-1, cv=cv, random_state=42, verbose=1
)
search.fit(Xtr, ytr)

best = search.best_estimator_
yp = best.predict(Xte)
acc = accuracy_score(yte, yp)
f1m = f1_score(yte, yp, average="macro")
print("Best params:", search.best_params_)
print(f"Test Accuracy: {acc:.4f}  |  Macro-F1: {f1m:.4f}")
print("\nClassification report:\n", classification_report(yte, yp, zero_division=0))

Test Accuracy: 0.9825  |  Macro-F1: 0.9749

Classification report:
               precision    recall  f1-score   support

           1       0.95      0.95      0.95       308
           2       0.99      0.98      0.99       496
           3       1.00      1.00      1.00       588
           4       0.94      1.00      0.97        95
           5       1.00      0.88      0.93         8
           6       0.98      0.99      0.99       150
           7       0.99      0.99      0.99        80
           8       0.98      0.97      0.97       246
           9       0.99      0.99      0.99       203

    accuracy                           0.98      2174
   macro avg       0.98      0.97      0.97      2174
weighted avg       0.98      0.98      0.98      2174


```

## running fit graph for both model
```python
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

base = search.best_estimator_
bp   = search.best_params_.copy()

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

def sweep_param(est, X, y, pname, grid):
    tr_err, va_err = [], []
    for v in grid:
        m = clone(est).set_params(**{pname: float(v)})
        m.fit(X, y)
        tr_err.append(1.0 - accuracy_score(y, m.predict(X)))
        va_acc = cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        va_err.append(1.0 - va_acc)
    return np.array(tr_err), np.array(va_err)

def diagnose(train_e, val_e, gap=0.08, high=0.25):
    if (val_e - train_e) > gap and train_e < high: return "Overfitting"
    if train_e > high and val_e > high and abs(val_e - train_e) < gap: return "Underfitting"
    return "Good fit"

os.makedirs("artifacts", exist_ok=True)

C_fix = float(bp.get("svc__C", 1.0))
gammas = np.logspace(-6, 0, 12)
est_g = clone(base).set_params(**{"svc__C": C_fix})
tr_g, va_g = sweep_param(est_g, Xtr, ytr, "svc__gamma", gammas)
i_best_g = int(np.argmin(va_g))
g_best = float(gammas[i_best_g])

plt.figure(figsize=(7,5), dpi=140)
plt.semilogx(gammas, tr_g, marker="o", label="Train error")
plt.semilogx(gammas, va_g, marker="o", label="Validation error")
plt.axvline(g_best, ls="--", lw=1)
plt.title("Fit curve (sweep γ, C fixed)")
plt.xlabel("gamma (log)"); plt.ylabel("Error = 1 - accuracy"); plt.legend(); plt.tight_layout()
plt.savefig("artifacts/fitting_graph_gamma.png", dpi=180, bbox_inches="tight")
plt.show()

g_fix = float(bp.get("svc__gamma", g_best))
Cs = np.logspace(-2, 3, 10)
est_c = clone(base).set_params(**{"svc__gamma": g_fix})
tr_c, va_c = sweep_param(est_c, Xtr, ytr, "svc__C", Cs)
i_best_c = int(np.argmin(va_c))
C_best = float(Cs[i_best_c])

plt.figure(figsize=(7,5), dpi=140)
plt.semilogx(Cs, tr_c, marker="o", label="Train error")
plt.semilogx(Cs, va_c, marker="o", label="Validation error")
plt.axvline(C_best, ls="--", lw=1)
plt.title("Fit curve (sweep C, γ fixed)")
plt.xlabel("C (log)"); plt.ylabel("Error = 1 - accuracy"); plt.legend(); plt.tight_layout()
plt.savefig("artifacts/fitting_graph_C.png", dpi=180, bbox_inches="tight")
plt.show()

chosen = clone(base).set_params(**{"svc__C": C_best, "svc__gamma": g_best})
chosen.fit(Xtr, ytr)
train_err = 1.0 - accuracy_score(ytr, chosen.predict(Xtr))
val_acc   = cross_val_score(chosen, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1).mean()
val_err   = 1.0 - val_acc
test_err  = 1.0 - accuracy_score(yte, chosen.predict(Xte))
diag      = diagnose(train_err, val_err)

print(f"Chosen hyperparams: C={C_best:.4g}, gamma={g_best:.3g}")
print(f"Train error={train_err:.4f} | Val error={val_err:.4f} | Test error={test_err:.4f}")
print(f"Diagnosis: {diag}")
```
### Baseline
![](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-18.png?raw=true)

### My model
![](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-19.png?raw=true)

- **Underfitting (left side, low `C`)**: both **train** and **validation** errors are high → margin too soft.
- **Good fit (middle, `C ≈ 1–50`)**: both errors drop and the **gap is small** → best generalization.
- **Overfitting regime (far right, very large `C`)**: **train error → 0**, while **validation error flattens/creeps up** → boundary too hard/noisy.

Baseline plot: dashed line is at a large C → train error ~0 while validation error flattens/creeps up ⇒ possible overfitting.
my model plot: dashed line sits in the mid-C “good fit” region, near the minimum of validation error ⇒ better generalization.

### B) Model(s) compared & decision
| Model | Best `C` | Best `γ` | Test Accuracy | Test Macro-F1 | Notes |
|---|---:|---:|---:|---:|---|
| **Baseline SVM** | ~385.66 | ~0.02212 | 0.9816 | 0.9681 | Slightly higher variance (right side of curve). |
| **Final SVM (picked)** | **~41.75** | **~0.00530** | **0.9829** | **0.9744** | Smaller train–test gap; best macro-F1. |

**Pick:** the **Final SVM** because it minimizes validation/test error with a **small gap** (good generalization) and **best Macro-F1 (0.9744)**.

### C) What to try next
- **K-Nearest Neighbors (KNN):** non-parametric, can capture local structure missed by SVM; tune `k` with CV (needs scaling—already done).
- **Decision Tree:** interpretable error analysis; control depth/min_samples_split to curb variance.
- **Naive Bayes (Multinomial/Bernoulli):** fast baseline for count/indicator features; good sanity check on separability.

> Current generalization snapshot: **Train error = 0.0058**, **Test error = 0.0171** (gap ≈ **1.1%**)

## 4) Conclusion & Next Steps

**Conclusion (1st model):**  
My RBF-SVM with full preprocessing (impute → encode → targeted log1p → MinMax → χ² → standardize) delivers **Test Accuracy = 0.9829** and **Macro-F1 = 0.9744** with a small train–test gap (**Error: train 0.0058 vs test 0.0171**). Performance is strong and consistent across classes; the only notable weakness is **Class 5** (very small support, n=8) where **recall = 1.00** and **precision = 0.89** (a few false positives). Overall, the model generalizes well and is a solid baseline for this dataset.

**What could improve it:**
- **Tighten feature selection:** tune χ² `k` (e.g., 70–90% of features) to drop weak/noisy columns and sharpen the margin.
- **Minority precision for Class 5:**  
  - Calibrate probabilities (Platt/sigmoid) and set a **slightly higher class-5 threshold** to cut false positives.  
  - Or use `class_weight` with a small **boost for class 5** via a dict (keeps runtime similar).
- **Local hyperparam refine:** small search around the found best (`C≈41.8`, `γ≈0.0053`) to fine-tune precision/recall trade-offs.
- **Data balance tweaks:** light oversampling of Class 5 (e.g., simple duplication) to stabilize its decision region without heavy runtime costs.

## 5 All my code notebooks 
[notebooks/Good_verison.ipynb](Good_verison.ipynb)
[notebooks/baseline_model.ipynb](baseline_model.ipynb)
