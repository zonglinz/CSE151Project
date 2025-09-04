# MS4

## 1 Train Second Model
### Methods
Split: train_test_split(test_size=0.20, stratify=y, random_state=42).

Preprocess (train-only fit; leakage-safe):
- Categorical → SimpleImputer(most_frequent) → OneHotEncoder(ignore_unknown, sparse=True)
- Numeric → SimpleImputer(median) → MaxAbsScaler (sparse-friendly)

Dimensionality reduction: TruncatedSVD with auto dims = 67 (≤ n_features−1=68−1), L2 normalization on embeddings.

Explained variance ≈ 1.0000.

Clustering: MiniBatchKMeans, k grid = {#labels, 2×, 3×, 4×} = {9,18,27,36}.

Choose k by validation accuracy (labels only used for selection + mapping; clustering remains unsupervised).

Picked k = 36, val acc 0.7891.

Cluster → label mapping: Hungarian assignment on the train split (global optimum).
```python
!pip -q install numpy pandas scikit-learn scipy

import os, json, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, normalize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment

DATA_PATH   = "/content/data.csv"
OUT_DIR     = "artifacts_model2_short"
EMBED_DIMS  = 192      
USE_L2      = True     
SEED        = 42

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)


LABEL_COL = None
for c in df.columns:
    if c.strip().lower() in {"label","labels","target","class","classes","category","y","outcome","diagnosis"}:
        LABEL_COL = c; break
if LABEL_COL is None and "Class" in df.columns:
    LABEL_COL = "Class"
if LABEL_COL is None:
    n = len(df)
    for c in df.columns:
        nu = df[c].nunique(dropna=False)
        if 2 <= nu <= min(50, max(2, int(0.2*n))):
            LABEL_COL = c; break
if LABEL_COL is None:
    raise ValueError("Couldn't infer label column.")

y = df[LABEL_COL].astype(str).to_numpy()
feat_cols = [c for c in df.columns if c != LABEL_COL]

def is_num(s): 
    return pd.api.types.is_numeric_dtype(s)
obj_cols      = [c for c in feat_cols if df[c].dtype == "object"]
low_card_cols = [c for c in feat_cols if is_num(df[c]) and df[c].nunique() <= 5]
cat_cols      = sorted(set(obj_cols + low_card_cols))
num_cols      = [c for c in feat_cols if c not in cat_cols]


Xtr_df, Xte_df, ytr, yte = train_test_split(df[feat_cols], y, test_size=0.20, stratify=y, random_state=SEED)


def make_ohe_sparse():
    try:    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except: return OneHotEncoder(handle_unknown="ignore", sparse=True)

cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe_sparse())])
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", MaxAbsScaler())])

trs = []
if num_cols: trs.append(("num", num_pipe, num_cols))
if cat_cols: trs.append(("cat", cat_pipe, cat_cols))
pre_train = ColumnTransformer(trs, remainder="drop", sparse_threshold=1.0)

Xtr_pre = pre_train.fit_transform(Xtr_df)
Xte_pre = pre_train.transform(Xte_df)
n_features = Xtr_pre.shape[1]

svd_dims = max(2, min(EMBED_DIMS, n_features - 1))
svd = TruncatedSVD(n_components=svd_dims, random_state=SEED)

Ztr = svd.fit_transform(Xtr_pre)
Zte = svd.transform(Xte_pre)

if USE_L2:
    Ztr = normalize(Ztr); Zte = normalize(Zte)

print(f"[Info] Preprocessed features: {n_features} | SVD dims used: {svd_dims} | var≈{svd.explained_variance_ratio_.sum():.4f}")

def hungarian_map(clusters, labels):
    clus = np.unique(clusters); labs = np.unique(labels)
    M = np.zeros((len(clus), len(labs)), dtype=int)
    ci = {c:i for i,c in enumerate(clus)}; li = {l:i for i,l in enumerate(labs)}
    for c, y in zip(clusters, labels): M[ci[c], li[y]] += 1
    r, c = linear_sum_assignment(-M)
    mp = {clus[i]: labs[j] for i, j in zip(r, c)}
    for c_id in clus:
        if c_id not in mp: mp[c_id] = labs[np.argmax(M[ci[c_id]])]
    return mp


n_labels = len(np.unique(ytr))
k_grid = sorted(set([n_labels, 2*n_labels, 3*n_labels, 4*n_labels])) 

Ztr_tr, Ztr_val, ytr_tr, ytr_val = train_test_split(Ztr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)

best = {"k": None, "val_acc": -1, "model": None, "map": None}
for k in k_grid:
    mbk = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=15, batch_size=4096, max_iter=250).fit(Ztr_tr)
    mapping = hungarian_map(mbk.labels_, ytr_tr)
    yval_pred = np.array([mapping[c] for c in mbk.predict(Ztr_val)], dtype=object)
    acc = accuracy_score(ytr_val, yval_pred)
    if acc > best["val_acc"]:
        best.update({"k": k, "val_acc": acc, "model": mbk, "map": mapping})

print(f"[k-choice] k∈{k_grid} | picked k={best['k']} | val_acc={best['val_acc']:.4f}")


final = MiniBatchKMeans(n_clusters=best["k"], random_state=SEED, n_init=20, batch_size=8192, max_iter=300).fit(Ztr)
c_tr, c_te = final.labels_, final.predict(Zte)
cl2lb = hungarian_map(c_tr, ytr)
fallback = Counter(ytr).most_common(1)[0][0]
ytr_pred = np.array([cl2lb.get(c, fallback) for c in c_tr], dtype=object)
yte_pred = np.array([cl2lb.get(c, fallback) for c in c_te], dtype=object)

def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"{name} Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, f1m

print("\n=== MODEL 2 (SHORT+FIXED) RESULTS ===")
acc_tr, f1m_tr = report("TRAIN", ytr, ytr_pred)
acc_te, f1m_te = report("TEST ", yte, yte_pred)
print(f"Train Error: {1-acc_tr:.4f} | Test Error: {1-acc_te:.4f} | Gap: {(1-acc_te)-(1-acc_tr):.4f}")

TRAIN Accuracy: 0.8306 | Macro-F1: 0.7239
              precision    recall  f1-score   support

           1       0.85      0.89      0.87      1203
           2       0.83      0.94      0.88      1982
           3       0.99      0.85      0.92      2353
           4       0.43      0.89      0.58       379
           5       0.04      0.29      0.07        34
           6       0.80      0.79      0.80       577
           7       0.89      0.82      0.85       318
           8       0.99      0.71      0.83       978
           9       0.93      0.59      0.72       806

    accuracy                           0.83      8630
   macro avg       0.75      0.75      0.72      8630
weighted avg       0.88      0.83      0.84      8630

TEST  Accuracy: 0.8295 | Macro-F1: 0.7199
              precision    recall  f1-score   support

           1       0.85      0.87      0.86       301
           2       0.85      0.94      0.89       496
           3       0.99      0.82      0.89       589
           4       0.41      0.92      0.56        95
           5       0.00      0.00      0.00         8
           6       0.82      0.83      0.82       144
           7       0.89      0.78      0.83        79
           8       0.98      0.77      0.86       245
           9       0.95      0.63      0.75       201

    accuracy                           0.83      2158
   macro avg       0.75      0.73      0.72      2158
weighted avg       0.89      0.83      0.85      2158

Train Error: 0.1694 | Test Error: 0.1705 | Gap: 0.0011
```

## 2 Evualtion 
### Headline Metrics
| Split | Accuracy | Macro‑F1 | Error *(1 − Accuracy)* |
|---|---:|---:|---:|
| **Train** | **0.8306** | **0.7239** | **0.1694** |
| **Test**  | **0.8295** | **0.7199** | **0.1705** |

**Generalization gap:** **+0.0011** (test error − train error) → ~**0.11%**.  
This tiny gap indicates a **good fit**: the model generalizes essentially as well as it trains for `k=36`.

### Interpretation & Takeaways
- **Bias–variance:** The low gap suggests variance is controlled; increasing k up to 36 improved both train and validation accuracy without overfitting (see fitting graph).  
- **Why Macro‑F1 < Accuracy:** Macro‑F1 equally weights each class. An **ultra‑minor class (Class 5, n=8)** underperforms, which drags macro‑F1 below overall accuracy even though major classes score high.
- **Class‑wise behavior (Test):**
  - **Strong:** 1, 2, 3, 6, 7, 8, 9 — clusters align well with these labels.
  - **Mixed:** 4 — decent F1; some boundary confusion.
  - **Weak:** 5 — very small support causes low precision/recall under k‑means’ spherical assumption.

## Error Sources (What They Mean)
- **Spherical clusters:** k‑means partitions by Voronoi cells; rare classes that are elongated or fragmented are hard to capture with a single centroid.  
- **Imbalance:** With few examples, minority clusters either don’t form or get absorbed by nearby major classes, hurting precision/recall for those minorities.

## 3 Fitting Graph
```python
import numpy as np, matplotlib.pyplot as plt, os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs(OUT_DIR, exist_ok=True)

k_grid = sorted(set([n_labels, 2*n_labels, 3*n_labels, 4*n_labels]))  # e.g., [9,18,27,36]
train_errs, val_errs = [], []

for k in k_grid:
    mbk = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=15, batch_size=4096, max_iter=250).fit(Ztr_tr)
    mp = hungarian_map(mbk.labels_, ytr_tr)
    y_tr_hat  = np.array([mp[c] for c in mbk.predict(Ztr_tr)], dtype=object)
    y_val_hat = np.array([mp[c] for c in mbk.predict(Ztr_val)], dtype=object)
    train_errs.append(1.0 - accuracy_score(ytr_tr, y_tr_hat))
    val_errs.append(1.0 - accuracy_score(ytr_val, y_val_hat))

plt.figure(figsize=(6.5,4.5), dpi=140)
plt.plot(k_grid, train_errs, marker="o", label="Train error")
plt.plot(k_grid, val_errs, marker="o", label="Validation error")
plt.title("Model 2: fitting graph (k vs. error)")
plt.xlabel("k (clusters)"); plt.ylabel("Error = 1 - accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fitting_graph_k.png"), dpi=160)
plt.close()
```
![fitting graph](https://github.com/zonglinz/CSE151Project/blob/main/fitting_graph_k.png?raw=true)

## Where the model sits on the fitting graph

- **Underfitting (left / small k).** At **k≈9**, both **train and validation errors are high (~0.40)** → not enough capacity to capture class substructure.
- **Capacity improves fit (middle).** Moving to **k=18–27** drops error sharply (~0.27–0.28). The curves remain close, so variance is still controlled.
- **Good‑fit region (right).** At **k=36**, errors are **lowest** (≈ **train 0.22**, **val 0.21**) and nearly overlap. Combined with the tiny **generalization gap** from Section 2 (~0.11%), this places the chosen model squarely in the **good‑fit** zone (neither under‑ nor over‑fitting).

> **Chosen configuration:** SVD (auto dims = 67, ~100% var) + MiniBatchKMeans (**k=36**), Hungarian mapping on train.

## What we’ll try next — and why

1. **Gaussian Mixture Models (diag/tied) on SVD embeddings**  
   *Why:* K‑means assumes spherical clusters; several classes (esp. **5** and **7**) likely form **elliptical/aniso** groups.  
   *Plan:* Sweep `n_components` around **3–5× #labels**, pick by **BIC/AIC** and validation accuracy after Hungarian mapping.

2. **Spectral clustering (Nyström approximation) with RBF affinity**  
   *Why:* Captures **non‑convex** shapes that k‑means/GMM miss; useful if classes wrap around each other.  
   *Plan:* Build an RBF kernel on SVD features, estimate **k** via the **eigengap**, Hungarian mapping to labels.

3. **NMF (+ K‑means) on non‑negative features**  
   *Why:* Produces **parts‑based** factors; often helps tiny, sparse patterns (minority classes).  
   *Plan:* Ensure non‑negativity (clip/shift if needed), pick rank by reconstruction‑error elbow, then cluster in NMF space.

4. **Density‑based clustering (HDBSCAN/DBSCAN) after SVD**  
   *Why:* Handles **variable density** and **noise**, preventing rare classes from being absorbed.  
   *Plan:* Run on SVD embeddings; sweep `min_cluster_size` / `eps`, map clusters to labels via Hungarian.

## 4 My MS4 workspace and code
[notebooks/MS4_Model.ipynb](MS4_Model.ipynb)

# Final Readme
## Introduction 
### Why chose this?
Security teams face an asymmetric fight: adversaries continuously repack, obfuscate, and rebrand malware, while defenders must triage vast volumes of binaries quickly and accurately. This project studies multi-class malware family classification using a well-scoped static dataset: 10,868 Windows samples with 69 engineered features extracted from disassembly—counts of assembly mnemonics (e.g., control-flow, data movement, arithmetic/logic) plus file-level scale (size_asm, line_count_asm). Labels cover 9 families, with pronounced class imbalance (e.g., some families are rare), zero-inflated counts, and long-tailed feature distributions. The dataset is clean (no missing values), enabling reproducible experimentation without heavy data cleaning.
### Why it’s cool
We frame the task as a rigorous, interpretable ML problem. First, we summarize how malware “writes its story” in assembly: control-flow (jmp, call, ret) can indicate obfuscation or complex behavior; data movement (mov, push, pop, I/O) reflects program structure; scale features capture global complexity. These signals motivate a pipeline that respects feature semantics: nonnegativity clipping, scaling, χ² univariate selection on counts, and a non-linear classifier to capture overlapping class manifolds. We adopt an imbalance-aware RBF-SVM with class_weight="balanced", optimized for macro-F1 and per-class recall rather than accuracy alone. This choice keeps the evaluation honest on minority families while remaining competitive on majority classes.
### Why a strong model matters (broader impact)
- Faster triage: Route the right files to the right defenses.
- Early containment: Fewer false negatives means less dwell time and damage.
- Human-centered ops: Reduce alert fatigue and focus analysts on the hardest cases.
- Defense-in-depth: Static signals complement dynamic sandboxes for layered security.

## Figures
![Class distribution](https://github.com/zonglinz/CSE151Project/blob/main/class_distribution.png?raw=true)
Legend. Bar heights show the number of samples per malware family (labels 1–9). Pronounced imbalance is visible (e.g., Simda is rare), motivating stratified splits, class weights, and macro-F1 reporting.

![Class distribution](https://github.com/zonglinz/CSE151Project/blob/main/Unknown-13.png?raw=true)
Legend. Percentage view complements raw counts and emphasizes minority classes. Use this to justify metrics beyond accuracy and any rebalancing you apply during training.

![Scatter matrix](https://github.com/zonglinz/CSE151Project/blob/main/Unknown-12.png?raw=true)
Diagonals show right-skewed histograms with many near-zero values (zero-inflation) and long tails, especially for asm_commands_dd and asm_commands_mov. Off-diagonals show a strong positive, near-linear trend between size_asm and line_count_asm. dd/mov rise with size/lines but with wide spread and outlier.

## Top-5 Most Important Features (Permutation Importance)

Baseline macro-F1: **0.845**

| Rank | Feature           | Δ macro-F1 (shuffle drop) |
|----:|-------------------|---------------------------:|
| 1   | line_count_asm    | 0.099816 |
| 2   | size_asm          | 0.090650 |
| 3   | asm_commands_dd   | 0.073983 |
| 4   | asm_commands_jmp  | 0.070917 |
| 5   | asm_commands_push | 0.060438 |

![bar chart for top 5 column with highest weighted for prediction](https://github.com/zonglinz/CSE151Project/blob/main/Unknown-14.png?raw=true)

Bar chart of permutation importance shows the five most predictive features. Shuffling line_count_asm and size_asm causes the largest macro-F1 drops, followed by asm_commands_dd, asm_commands_jmp, and asm_commands_push. Size/scale and control-flow/data-moveme

## Method Section
### Data Exploration
#### How many observations?
- **Rows:** **10,868**
- **Columns:** **69**

#### Columns, scales, and distributions
- **Heuristic typing:** **55 continuous**, **14 categorical** (including string/ID-like or low-cardinality numeric).  
- **Per-column summary** (dtype, variable type, scale, cardinality, missing%, plus brief notes):

| name | dtype | var_type | scale | n_unique | missing_% | notes |
| --- | --- | --- | --- | --- | --- | --- |
| asm_commands_add | int64 | continuous | numeric (interval/ratio) | 2122 | 0.0 |  μ≈724, σ≈1.57e+03 |
| asm_commands_call | float64 | continuous | numeric (interval/ratio) | 1993 | 0.0 |  μ≈959, σ≈2.89e+03 |
| asm_commands_cdq | float64 | continuous | numeric (interval/ratio) | 191 | 0.0 |  μ≈10.8, σ≈39.1 |
| asm_commands_cld | float64 | continuous | numeric (interval/ratio) | 210 | 0.0 |  μ≈200, σ≈1.57e+03 |
| asm_commands_cli | float64 | continuous | numeric (interval/ratio) | 200 | 0.0 |  μ≈16.8, σ≈184 |
| asm_commands_cmc | float64 | categorical (numeric-codes likely) | nominal | 54 | 0.0 |  top: 0.0(9403); 1.0(502); 2.0(245) |
| asm_commands_cmp | float64 | continuous | numeric (interval/ratio) | 1563 | 0.0 |  μ≈480, σ≈1.38e+03 |
| asm_commands_cwd | float64 | categorical (numeric-codes likely) | nominal | 46 | 0.0 |  top: 0.0(8761); 1.0(609); 2.0(355) |
| asm_commands_daa | float64 | continuous | numeric (interval/ratio) | 1007 | 0.0 |  μ≈221, σ≈498 |
| asm_commands_dd | int64 | continuous | numeric (interval/ratio) | 6180 | 0.0 |  μ≈1.7e+04, σ≈3.29e+04 |
| asm_commands_dec | float64 | continuous | numeric (interval/ratio) | 1495 | 0.0 |  μ≈370, σ≈926 |
| asm_commands_dw | float64 | continuous | numeric (interval/ratio) | 3143 | 0.0 |  μ≈1.63e+03, σ≈3.56e+03 |
| asm_commands_endp | float64 | continuous | numeric (interval/ratio) | 975 | 0.0 |  μ≈198, σ≈639 |
| asm_commands_faddp | float64 | categorical (numeric-codes likely) | nominal | 80 | 0.0 |  top: 0.0(9483); 2.0(425); 3.0(376) |
| asm_commands_fchs | float64 | categorical (numeric-codes likely) | nominal | 56 | 0.0 |  top: 0.0(9601); 1.0(570); 4.0(251) |
| asm_commands_fdiv | float64 | categorical (numeric-codes likely) | nominal | 92 | 0.0 |  top: 0.0(5279); 2.0(1423); 8.0(1078) |
| asm_commands_fdivr | float64 | categorical (numeric-codes likely) | nominal | 52 | 0.0 |  top: 0.0(8809); 1.0(606); 5.0(353) |
| asm_commands_fistp | float64 | categorical (numeric-codes likely) | nominal | 55 | 0.0 |  top: 0.0(9466); 1.0(738); 3.0(182) |
| asm_commands_fld | float64 | continuous | numeric (interval/ratio) | 337 | 0.0 |  μ≈37.5, σ≈406 |
| asm_commands_fstp | float64 | continuous | numeric (interval/ratio) | 327 | 0.0 |  μ≈34.4, σ≈342 |
| asm_commands_fword | float64 | continuous | numeric (interval/ratio) | 142 | 0.0 |  μ≈3.86, σ≈30.2 |
| asm_commands_fxch | float64 | continuous | numeric (interval/ratio) | 116 | 0.0 |  μ≈10.7, σ≈77.4 |
| asm_commands_imul | float64 | continuous | numeric (interval/ratio) | 595 | 0.0 |  μ≈540, σ≈3.31e+03 |
| asm_commands_in | int64 | continuous | numeric (interval/ratio) | 2314 | 0.0 |  μ≈1.2e+03, σ≈5.64e+03 |
| asm_commands_inc | float64 | continuous | numeric (interval/ratio) | 795 | 0.0 |  μ≈126, σ≈532 |
| asm_commands_ins | float64 | continuous | numeric (interval/ratio) | 353 | 0.0 |  μ≈44.6, σ≈241 |
| asm_commands_jb | float64 | continuous | numeric (interval/ratio) | 526 | 0.0 |  μ≈59.6, σ≈202 |
| asm_commands_je | float64 | continuous | numeric (interval/ratio) | 354 | 0.0 |  μ≈66.7, σ≈584 |
| asm_commands_jg | float64 | continuous | numeric (interval/ratio) | 516 | 0.0 |  μ≈49, σ≈161 |
| asm_commands_jl | float64 | continuous | numeric (interval/ratio) | 629 | 0.0 |  μ≈76.5, σ≈254 |
| asm_commands_jmp | float64 | continuous | numeric (interval/ratio) | 1351 | 0.0 |  μ≈321, σ≈930 |
| asm_commands_jnb | float64 | continuous | numeric (interval/ratio) | 307 | 0.0 |  μ≈23.7, σ≈90.3 |
| asm_commands_jno | float64 | categorical (numeric-codes likely) | nominal | 51 | 0.0 |  top: 0.0(9717); 1.0(408); 2.0(196) |
| asm_commands_jo | float64 | continuous | numeric (interval/ratio) | 115 | 0.0 |  μ≈6.44, σ≈67.1 |
| asm_commands_jz | float64 | continuous | numeric (interval/ratio) | 1268 | 0.0 |  μ≈343, σ≈1.16e+03 |
| asm_commands_lea | float64 | continuous | numeric (interval/ratio) | 1479 | 0.0 |  μ≈491, σ≈1.63e+03 |
| asm_commands_mov | float64 | continuous | numeric (interval/ratio) | 3805 | 0.0 |  μ≈4.22e+03, σ≈1.16e+04 |
| asm_commands_mul | float64 | continuous | numeric (interval/ratio) | 655 | 0.0 |  μ≈571, σ≈3.42e+03 |
| asm_commands_not | float64 | continuous | numeric (interval/ratio) | 329 | 0.0 |  μ≈45.1, σ≈332 |
| asm_commands_or | int64 | continuous | numeric (interval/ratio) | 4161 | 0.0 |  μ≈3.08e+03, σ≈7.67e+03 |
| asm_commands_out | float64 | continuous | numeric (interval/ratio) | 258 | 0.0 |  μ≈23.6, σ≈138 |
| asm_commands_outs | float64 | categorical (numeric-codes likely) | nominal | 78 | 0.0 |  top: 0.0(9116); 1.0(513); 2.0(288) |
| asm_commands_pop | float64 | continuous | numeric (interval/ratio) | 1643 | 0.0 |  μ≈583, σ≈1.58e+03 |
| asm_commands_push | float64 | continuous | numeric (interval/ratio) | 2645 | 0.0 |  μ≈1.77e+03, σ≈5.34e+03 |
| asm_commands_rcl | float64 | continuous | numeric (interval/ratio) | 121 | 0.0 |  μ≈4.97, σ≈23.8 |
| asm_commands_rcr | float64 | categorical (numeric-codes likely) | nominal | 93 | 0.0 |  top: 0.0(5547); 5.0(1389); 3.0(1069) |
| asm_commands_rep | float64 | continuous | numeric (interval/ratio) | 328 | 0.0 |  μ≈32.7, σ≈128 |
| asm_commands_ret | float64 | continuous | numeric (interval/ratio) | 1238 | 0.0 |  μ≈292, σ≈805 |
| asm_commands_rol | float64 | continuous | numeric (interval/ratio) | 291 | 0.0 |  μ≈38.3, σ≈457 |
| asm_commands_ror | float64 | continuous | numeric (interval/ratio) | 338 | 0.0 |  μ≈44.7, σ≈252 |
| asm_commands_sal | float64 | categorical (numeric-codes likely) | nominal | 107 | 0.0 |  top: 0.0(5606); 8.0(995); 12.0(680) |
| asm_commands_sar | float64 | continuous | numeric (interval/ratio) | 297 | 0.0 |  μ≈29.7, σ≈206 |
| asm_commands_sbb | float64 | continuous | numeric (interval/ratio) | 318 | 0.0 |  μ≈26.7, σ≈97.6 |
| asm_commands_scas | float64 | categorical (numeric-codes likely) | nominal | 82 | 0.0 |  top: 0.0(8412); 1.0(679); 2.0(468) |
| asm_commands_shl | float64 | continuous | numeric (interval/ratio) | 442 | 0.0 |  μ≈55.8, σ≈401 |
| asm_commands_shr | float64 | continuous | numeric (interval/ratio) | 422 | 0.0 |  μ≈42.5, σ≈253 |
| asm_commands_sidt | float64 | categorical (numeric-codes likely) | nominal | 18 | 0.0 |  top: 0.0(10544); 3.0(134); 4.0(73) |
| asm_commands_stc | float64 | continuous | numeric (interval/ratio) | 115 | 0.0 |  μ≈9.15, σ≈151 |
| asm_commands_std | float64 | continuous | numeric (interval/ratio) | 855 | 0.0 |  μ≈606, σ≈3.9e+03 |
| asm_commands_sti | float64 | continuous | numeric (interval/ratio) | 148 | 0.0 |  μ≈6.15, σ≈36.8 |
| asm_commands_stos | float64 | continuous | numeric (interval/ratio) | 186 | 0.0 |  μ≈15.1, σ≈55.1 |
| asm_commands_sub | float64 | continuous | numeric (interval/ratio) | 3002 | 0.0 |  μ≈2.16e+03, σ≈6.67e+03 |
| asm_commands_test | float64 | continuous | numeric (interval/ratio) | 1311 | 0.0 |  μ≈331, σ≈1.11e+03 |
| asm_commands_wait | float64 | continuous | numeric (interval/ratio) | 125 | 0.0 |  μ≈6.23, σ≈17 |
| asm_commands_xchg | float64 | continuous | numeric (interval/ratio) | 196 | 0.0 |  μ≈71.7, σ≈581 |
| asm_commands_xor | float64 | continuous | numeric (interval/ratio) | 1309 | 0.0 |  μ≈493, σ≈2.47e+03 |
| line_count_asm | int64 | continuous | numeric (interval/ratio) | 911 | 0.0 |  μ≈8.07e+04, σ≈6.45e+04 |
| size_asm | int64 | continuous | numeric (interval/ratio) | 990 | 0.0 |  μ≈4.68e+06, σ≈3.74e+06 |
| Class | int64 | categorical (numeric-codes likely) | nominal | 9 | 0.0 |  top: 3(2942); 2(2478); 1(1541) |

#### Subgroups (what they are & why they matter)

- **Global size/scale** — (`line_count_asm`, `size_asm`)  
  Rough size of the file and the assembly listing. *Different families/packers often produce consistently larger or smaller binaries.*

- **Directives/structure** — (`asm_commands_dd`, `asm_commands_dw`, `asm_commands_endp`, `asm_commands_fword`)  
  Assembler directives for defining data/words and marking procedures. *Structural “fingerprints” can hint at certain toolchains or packers.*

- **Control-flow** — (`asm_commands_call`, `asm_commands_ret`, `asm_commands_jmp`, `asm_commands_je`/`jz`, `asm_commands_jl`/`jg`/`jb`/`jnb`, `asm_commands_jo`/`jno`, `asm_commands_rep`)  
  Branching and function transfer. *High jump/call density may indicate obfuscation, anti-analysis tricks, or family-specific control patterns.*

- **Data movement** — (`asm_commands_mov`, `asm_commands_lea`, `asm_commands_push`, `asm_commands_pop`, `asm_commands_xchg`, `asm_commands_in`/`out`, `asm_commands_ins`/`outs`, `asm_commands_stos`, `asm_commands_scas`)  
  How values move between registers, memory, stack, and I/O. *Stack-heavy vs register-heavy movement can differentiate families.*

- **Arithmetic/logic** — (`asm_commands_add`, `sub`, `mul`/`imul`, `inc`/`dec`, `cmp`, `test`, `or`/`xor`/`not`, shifts/rotates `shl`/`shr`/`sar`/`sal`/`rol`/`ror`, flags math `sbb`, sign/BCD ops `cdq`/`cwd`/`daa`)  
  Bit-twiddling and math texture. *Encryption loops, checksums, and junk code often leave characteristic arithmetic/logic patterns.*

- **Floating point (x87)** — (`asm_commands_faddp`, `fchs`, `fdiv`/`fdivr`, `fistp`, `fld`/`fstp`, `fxch`)  
  Floating-point usage is uncommon in many malware families; *spikes can signal specific packers/obfuscators or inserted junk.*

- **Flags/system** — (`asm_commands_stc`, `cld`, `cli`, `std`, `sti`, `sidt`, `cmc`, `wait`)  
  CPU flags and privileged/system-level instructions. *These can reveal low-level routines or anti-analysis behaviors.*

- **Other asm** — Any `asm_commands_*` not cleanly fitting the above buckets.  
  *(Catch-all for rarer mnemonics.)*

- **Target label** — (`Class`)  
  The ground-truth family label: **1–9 →** Ramnit, Lollipop, Kelihos_ver3, Vundo, Simda, Tracur, Kelihos_ver1, Obfuscator.ACY, Gatak. 

#### Missing and duplicate values
- **Missing cells:** **0** (rows with any missing: **0**)  
- **Duplicate rows:** **80**

#### Target column and labels
- **Target candidate:** `Class` (multi-class).  
- **Label → family mapping**
- **1**: Ramnit
- **2**: Lollipop
- **3**: Kelihos_ver3
- **4**: Vundo
- **5**: Simda
- **6**: Tracur
- **7**: Kelihos_ver1
- **8**: Obfuscator.ACY
- **9**: Gatak

#### Class distribution
| Label | Family | Count | Percent |
|------:|--------|------:|--------:|
| 1 | Ramnit | 1,541 | 14.18% |
| 2 | Lollipop | 2,478 | 22.80% |
| 3 | Kelihos_ver3 | 2,942 | 27.07% |
| 4 | Vundo | 475 | 4.37% |
| 5 | Simda | 42 | 0.39% |
| 6 | Tracur | 751 | 6.91% |
| 7 | Kelihos_ver1 | 398 | 3.66% |
| 8 | Obfuscator.ACY | 1,228 | 11.30% |
| 9 | Gatak | 1,013 | 9.32% |

### Preprocessing + Train(Model 1)
#### Load, de-duplicate, and target inference
- **Load**: `pd.read_csv('/content/data.csv')` link for downoload dataset(i miss this in p2): https://drive.google.com/file/d/1LDS1KWr3CL4DPNiDOkBB0IVSiXELVpv_/view?usp=sharing
- **De-duplicate**: `df.drop_duplicates()` to remove exact duplicate rows so they don’t bias the fit.
- **Target/label inference**: If a canonical label column isn’t provided, we scan for common names (e.g., `label`, `target`, `class`) and, failing that, pick a **low-cardinality** column (2–min(50, 20% of n)) as the label. This makes the notebook plug‑and‑play across datasets without manual edits.

#### Feature typing (categorical vs. numeric)
I derive two disjoint sets:
- **Categorical columns (`cat_cols`)**:
  1) columns with `object` dtype,
  2) *low-cardinality* numeric columns (≤5 unique values) which behave like categories (e.g., codes, bins),
  3) a small set of **name‑hints** for opcode-like columns (e.g., `asm_commands_*`).
- **Numeric columns (`num_cols`)**: everything else.

This conservative typing ensures that enumerations and codes are **one‑hot encoded** rather than treated as continuous magnitudes.

#### Imputation (handle missing values)
- **Numeric**: `SimpleImputer(strategy="median")` — robust to outliers and preserves central tendency.
- **Categorical**: `SimpleImputer(strategy="most_frequent")` — replaces missing categories with the mode, keeping the feature usable for one‑hot encoding.

All imputation happens **inside the pipeline**, so it is **fit only on training folds** (no leakage).

#### Targeted nonlinearity / feature expansion (numeric)
I create a **FeatureUnion** with two branches over numeric inputs:
- **Identity pass-through** (keeps the original numeric features).
- **Targeted log transform** on *heavy* features: I mark a numeric column as *heavy* if `max(value) > 100` **or** `nunique > 50`. Those columns get `log1p(x) = log(1 + x)`.

**Why log1p?**
- Tames extreme right-skew and large dynamic ranges.
- Keeps zeros well-defined (`log1p(0)=0`).
- Preserves rank/order while compressing very large values.

This produces an **expanded numeric set**: original + log1p‑transformed heavy columns.

#### Non-negativity for χ² scoring
Before χ² feature selection, we **clip numeric values to ≥ 0**:
> `clip_nonneg(X) = max(X, 0)`

**Rationale:** `SelectKBest(chi2)` requires **non-negative** features; clipping guarantees validity for any numeric that may be negative while retaining signal for count-like or magnitude features.

####  Scaling
Two complementary scalers are used at different stages:
- **Min–Max scaling (per numeric feature)** to `[0, 1]` *inside* the numeric pipeline. This normalizes ranges across original and log1p features and improves χ² comparability.
- **Standardization (after column-wise union & selection)**: `StandardScaler(with_mean=True, with_std=True)` centers/scales **the selected full design matrix** (numeric + one‑hot) to mean 0, unit variance — a good default for RBF SVMs.

> Notes:
> - One‑hot features (0/1) also get standardized at the end; this can help kernels treat dense and sparse blocks more uniformly.
> - The order is **(impute → clip → expand → MinMax)** within numeric, **then one‑hot** for categorical, **then χ² selection**, **then StandardScaler**.

#### Categorical encoding
`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` encodes each category into its own binary column while **safely ignoring unseen categories** at test time. We set `sparse_output=False` to work seamlessly with later dense transforms.

Low‑cardinality numeric codes are intentionally treated as categorical here to avoid imposing false numeric distances.

#### Feature selection (dimensionality & noise control)
I apply `SelectKBest(chi2)` with:
```
k = min(n_features, max(10, int(0.9 * n_features)))
```
i.e., **keep up to 90%** of features (but at least 10). χ² ranks features by their dependence with the label, removing weak/noisy columns and reducing overfitting and compute cost. Because it’s part of the pipeline, the selection is **re-fit each CV fold** (no leakage).

#### Leakage-safe orchestration
All steps above are wrapped in a `ColumnTransformer` and `Pipeline`, and hyperparameter search uses cross‑validation. That guarantees:
- **Imputation, scaling, encoding, log1p, and χ² selection are learned only from training folds**.
- The exact same transformations are applied to validation/test sets.

####  Preprocessing code + Train code + Result

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

###Preprocessing + Train(Model 2)

## Result for Model 1(Supervised and Model 2 Unsupervised)
### Model 1 result 
![Confusion Matrix Train](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-16.png?raw=true)
![Confusion Matrix Test](https://github.com/zonglinz/CSE151Project/blob/Milestone3/Unknown-15.png?raw=true)
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

- **Generalization gap:** ~**1.13%** (0.9942 → 0.9829). This is small, indicating **low overfitting** and a well-regularized boundary for the chosen `C, γ`.  
- **Macro-F1 vs Weighted-F1:** Macro-F1 **0.9744** (treats classes equally) vs weighted avg (≈ overall **0.98**). The slightly lower macro-F1 reflects performance on **minority classes** (esp. Class 5 with only 8 samples).

#### B. Class-wise Behavior (from the test classification report)
- **High-support classes (2, 3, 9):** All near-perfect (F1 ≈ 0.99–1.00), showing the model captures dominant family patterns very well.  
- **Moderate classes (1, 4, 6, 7, 8):** F1 in **0.96–0.99** range.  
- **Minor class (5, n=8):** **Recall 1.00**, **Precision 0.89**, **F1 0.94**.  
  - *Interpretation:* The model **finds every true Class-5 sample** (no false negatives), but a few **false positives** get labeled as Class-5. On an imbalanced task this is an acceptable trade-off if missing Class-5 is costly; otherwise, we can tune thresholds (see below).

#### C. Error Patterns & What They Likely Mean
- **Precision-recall trade-off for Class-5:** High recall + lower precision ⇒ the decision boundary is **generous** toward Class-5. Likely some feature combinations (e.g., unusual control-flow or arithmetic counts) overlap with rare families.  
- **Class-8 slightly lower recall (0.96):** A few Class-8 samples likely sit near boundaries shared with classes 1/4/7 (families with similar instruction “texture”).  
- **Overall:** Errors are **sparse** and concentrated in **minority, edge-case** regions—consistent with high macro-F1.

#### D. Bias–Variance Perspective
- **`γ ≈ 0.0053`** (post-standardization) is **not** extreme; it yields a **smooth** RBF surface rather than a wiggly boundary.  
- **`C ≈ 41.8`** enforces **moderate margin hardness**—enough to fit complex structure, but your small generalization gap suggests **variance is under control**.  
- With χ² keeping ≈90% of features and targeted `log1p`, the model has **ample capacity** without drifting into overfit; the data seems linearly separable in RBF space with a clean margin.

#### E. Reliability & Robustness Checks (recommended)
- **Variance of estimates:** 2-fold CV is efficient but has **higher variance**. To report stability, repeat the search with a different `random_state` or use **RepeatedStratifiedKFold** (e.g., 2 folds × 3 repeats) just to measure spread in CV F1 (you can keep the same final model).  
- **Significance vs other models:** If you compared multiple finalists, run a **McNemar test** on the *same* test split to confirm any small differences are statistically meaningful.

#### F. Thresholds, Calibration & Operating Points
- If false positives on Class-5 are costly, calibrate and adjust decision thresholds:
  - Fit `CalibratedClassifierCV(best_estimator, method="sigmoid", cv="prefit")` (or refit SVC with `probability=True`) and set **class-specific thresholds** to balance precision/recall per your costs.
- **Macro-F1** is strong already; calibration mainly helps **decision control** rather than macro-F1 itself.

#### G. What Drove Performance Here
- **Preprocessing:**  
  - Median imputation prevents losing rows;  
  - **Clip + MinMax** guaranteed **non-negative** features for χ²;  
  - **Targeted `log1p`** on high-range/high-card columns stabilized scale and added nonlinear signal **without O(p²)** blow-up;  
  - χ² at ~90% preserved most discriminative features;  
  - Final **standardization** suited the SVM margin geometry.
- **Modeling:** A **balanced RBF-SVM** is a strong baseline for this domain; your search found a **smooth yet expressive** operating point.

#### H. Model(s) compared & decision
| Model | Best `C` | Best `γ` | Test Accuracy | Test Macro-F1 | Notes |
|---|---:|---:|---:|---:|---|
| **Baseline SVM** | ~385.66 | ~0.02212 | 0.9816 | 0.9681 | Slightly higher variance (right side of curve). |
| **Final SVM (picked)** | **~41.75** | **~0.00530** | **0.9829** | **0.9744** | Smaller train–test gap; best macro-F1. |

**Pick:** the **Final SVM** because it minimizes validation/test error with a **small gap** (good generalization) and **best Macro-F1 (0.9744)**.

### Model 2 result

## Discussion
### Why & framing.
I framed this as a static malware family classification problem with zero-inflated opcode counts, heavy-tailed magnitudes, and class imbalance. Those properties shaped every design choice: keep counts non-negative, prefer feature scoring that respects count semantics, and evaluate with macro-F1 to protect minority families.
### Process & rationale (mirrors Methods).
- Data integrity. No missing values; remove duplicates to avoid leakage.
- Preprocessing. Median/most-frequent imputation, non-negativity clip, selective log1p for very large/count-like features, global scaling. This normalizes dynamic ranges so distance-based models behave sensibly.
- Feature selection. χ² (on non-negative inputs) to surface features most associated with labels; reduces noise and speeds training.
- Model 1 (supervised). RBF-SVM with class_weight="balanced" targets non-linear class boundaries and imbalance directly.
- Model 2 (unsupervised). SVD compresses correlated signals; MiniBatchKMeans/GMM cluster in the embedding; Hungarian aligns clusters to labels; a tiny k-grid favors stability/speed.
### Interpretation & believability.
Model 1 reached Test Acc ≈ 0.983, Macro-F1 ≈ 0.974 with a small gap (Train Err 0.0058 vs Test Err 0.0171). Cross-validation error (≈0.0239) roughly matched test error—evidence against overfit. Per-class metrics were strong, including minority labels (noting that tiny supports make these estimates fragile). Confusions concentrated among families likely sharing toolchains/packing, consistent with domain intuition.
Model 2 produced coherent clusters for distinctive families and offered fast embeddings/diagnostics, but it cannot tune per-class recall or leverage subtle label information, so it trails Model 1—as expected.

### Shortcomings & self-critique.
Random splits may overstate performance vs temporal/toolchain-aware splits. Size confounding (size_asm, line_count_asm) might shortcut “true” behavior. Probabilities are uncalibrated. Neither model handles open-set/OOD inputs. Model 2’s mapping can vary with seed, SVD dims, and k (over-merge/fragment).

### Balancing both models & future directions.
Use Model 2 to warm-start pseudo-labels, as a “bag-of-clusters” feature for Model 1, or for rapid triage before supervised scoring. Next: probability calibration, class-specific thresholds, size-controlled ablations, LightGBM/XGBoost baselines, SVM+GBDT ensembling, conformal/OOD abstention, and drift monitors.

### Closing thought.
There are no unicorns here—just a reproducible, interpretable static pipeline (Model 1) complemented by a compact embedding tool (Model 2). Together they form a practical, scrutinizable layer in defense-in-depth, with clear paths to make them sturdier when the future’s “donkeys” arrive.


## Conclusion 
Model 1 (χ² → scaling/standardization → RBF-SVM, class-weighted) proved to be a strong, sane baseline: Test Accuracy ≈ 0.983, Macro-F1 ≈ 0.974, with a small train–test gap, which suggests the preprocessing + imbalance handling fit this dataset’s zero-inflated counts and long tails. Importantly, minority families did not collapse: in the shown split, even Simda (support=8) reached recall ≈ 1.0—a promising, if fragile, sign. Model 2 (SVD → {MiniBatchKMeans, GMM} + Hungarian with a tiny k-grid) was compact, fast, and label-efficient. As expected, it lagged the supervised SVM, but it’s operationally attractive for rapid embeddings, diagnostics, and semi-supervised extensions.

If starting over, I would (1) calibrate probabilities (Platt/Isotonic) and tune class-specific thresholds from PR curves to explicitly trade precision vs. recall on minority classes; (2) add class-balanced/focal losses and compare to LightGBM/XGBoost, then try a simple SVM+GBDT ensemble for robustness; (3) run size-confound ablations (control or regress out size_asm / line_count_asm) to ensure performance isn’t just “big binary heuristics”; (4) evaluate with temporal/toolchain-aware splits to simulate drift and repacking patterns. Reliability-wise, I’d incorporate open-set/OOD detection and lightweight conformal prediction to enable abstention and safer escalation.

For analysis depth, I’d expand error forensics: cluster misclassifications, inspect opcode/control-flow motifs, and visualize embeddings (PCA/UMAP) to see where families overlap. I’d also stress-test robustness by perturbing opcode histograms or global size, measuring performance decay. On the engineering side: freeze the pipeline, export artifacts, add a model card, monitor feature/label drift, and wire an abstain-and-sandbox path for low-confidence cases.

## Statement of Collaboration
Name: Zonglin Zhang

Title: Project Lead (Solo) — data scientist, engineer, writer, project manager

Contribution: I completed the project independently. Responsibilities included: dataset preparation (deduping, validation), exploratory data analysis and all figures/legends, feature engineering and preprocessing pipelines, model development (search/tuning, evaluation), ablations and error analysis, write-up and references, repository organization (notebooks, figures, README), environment setup, and final packaging/submission. No other contributors participated.

Note: This was an individual project; there were no teammates.



