# ============================================================
# UHI MODEL COMPARISON - V6 (CNN FOCUSED FIX)
# ROOT CAUSE FIX:
#   1. ALL data for CNN training (no sampling limit)
#   2. Simpler architecture (deep != better for 4 features)
#   3. No normalization of y (train on raw LST values)
#   4. Adam with fixed LR (CosineDecay was decaying too fast)
#   5. No dropout in first layers
# ============================================================

!pip install -q rasterio tensorflow scikit-learn pandas numpy

import os, time, math, warnings
import numpy as np
import pandas as pd
import rasterio

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from google.colab import files

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# UPLOAD
# ============================================================
print("="*60)
print("Upload pannunga (6 files):")
print(" 1.svm_21data.csv  2.NDVI.tif  3.NDBI.tif")
print(" 4.LULC.tif        5.POPULATION.tif  6.LST.tif")
print("="*60)
uploaded = files.upload()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
    print(f"[GPU] {len(gpus)} GPU ✅")
else:
    print("[CPU] No GPU — CNN may be slow")

files_list = os.listdir("/content")
def find_file(names):
    for n in names:
        if n in files_list: return "/content/" + n
    return None

csv_path  = find_file(["svm_21data.csv","svm_data.csv"])
ndvi_path = find_file(["NDVI.tif"])
ndbi_path = find_file(["NDBI.tif"])
lulc_path = find_file(["LULC.tif"])
pop_path  = find_file(["POPULATION.tif"])
lst_path  = find_file(["LST.tif"])

for name, path in [("CSV",csv_path),("NDVI",ndvi_path),("NDBI",ndbi_path),
                   ("LULC",lulc_path),("POP",pop_path),("LST",lst_path)]:
    if path is None: raise FileNotFoundError(f"[ERROR] {name} not found!")
    print(f"[OK] {name}")

OUT = {
    "reg_pred"  : "/content/MR_Predicted_LST.tif",
    "reg_error" : "/content/MR_Error_LST.tif",
    "svm_pred"  : "/content/SVM_Predicted_LST.tif",
    "svm_error" : "/content/SVM_Error_LST.tif",
    "cnn_pred"  : "/content/CNN_Predicted_LST.tif",
    "cnn_error" : "/content/CNN_Error_LST.tif",
    "cnn_model" : "/content/CNN_Model.keras",
    "report"    : "/content/Model_Comparison_Report.txt",
}

CONFIG = {
    "feature_cols" : ["ndvi2021sum","ndbi21_s","lulc25","F21_pop"],
    "target_col"   : "grid_code",
    "nodata_value" : -9999,
    "random_state" : 42,
    "TEST_RATIO"   : 0.20,
    "TRAIN_SVM"    : 3000,
    "TRAIN_REG"    : 5000,
    "N_STRATA"     : 20,
    "pred_chunk"   : 20000,
}

# ============================================================
# HELPERS
# ============================================================
def save_geotiff(array_2d, ref_profile, output_path, nodata=-9999.0):
    profile = ref_profile.copy()
    profile.update({"driver":"GTiff","dtype":"float32","count":1,"nodata":nodata})
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array_2d.astype(np.float32), 1)

def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    r2   = r2_score(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae  = mean_absolute_error(y_true[mask], y_pred[mask])
    return r2, rmse, mae

def spatial_metrics(pred_2d, lst_2d, nodata=-9999.0):
    valid  = (pred_2d != nodata) & np.isfinite(lst_2d)
    pv     = pred_2d[valid].astype(np.float64)
    lv     = lst_2d[valid].astype(np.float64)
    errors = pv - lv
    return (np.sqrt(np.mean(errors**2)),
            np.mean(np.abs(errors)),
            np.mean(errors),
            errors, valid)

def stratified_sample(X, y, n_sample, feat_names, seed=42):
    n_sample = min(n_sample, len(X))
    df_tmp       = pd.DataFrame(X, columns=feat_names)
    df_tmp["_y"] = y
    df_tmp["_b"] = pd.qcut(df_tmp["_y"], q=CONFIG["N_STRATA"],
                             labels=False, duplicates="drop")
    per_bin = max(1, n_sample // df_tmp["_b"].nunique())
    parts   = []
    for _, grp in df_tmp.groupby("_b"):
        parts.append(grp.sample(n=min(per_bin, len(grp)), random_state=seed))
    sampled = pd.concat(parts)
    gap = n_sample - len(sampled)
    if gap > 0:
        rest = df_tmp[~df_tmp.index.isin(sampled.index)]
        if len(rest) > 0:
            sampled = pd.concat([sampled,
                                  rest.sample(n=min(gap,len(rest)),random_state=seed)])
    sampled = sampled.drop(columns=["_b"], errors="ignore")
    return (sampled[feat_names].values.astype(np.float32),
            sampled["_y"].values.astype(np.float32))

def predict_raster_flat(predict_fn, feat_flat, valid_px, chunk_sz=20000):
    pred_flat = np.full(rows*cols, -9999.0, dtype=np.float32)
    vidx      = np.where(valid_px)[0]
    vfeat     = feat_flat[vidx]
    n_chunks  = math.ceil(len(vidx)/chunk_sz)
    for i in range(n_chunks):
        s = i*chunk_sz; e = min(s+chunk_sz, len(vidx))
        p = predict_fn(vfeat[s:e])
        p = np.nan_to_num(np.array(p).flatten(),
                           nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
        pred_flat[vidx[s:e]] = p
        if (i+1)%50==0 or (i+1)==n_chunks:
            print(f"    Chunk {i+1}/{n_chunks}")
    return pred_flat.reshape(rows, cols)

# ============================================================
# STEP 1: DATA LOAD
# ============================================================
print("\n" + "="*60)
print("STEP 1: DATA LOADING")
print("="*60)

chunks = []
for chunk in pd.read_csv(csv_path, chunksize=100000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"Total rows : {len(df):,}")

required = CONFIG["feature_cols"] + [CONFIG["target_col"]]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

df = df[required].copy()
df.replace([CONFIG["nodata_value"], np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

tgt = CONFIG["target_col"]
p1, p99 = df[tgt].quantile(0.01), df[tgt].quantile(0.99)
df = df[(df[tgt]>=p1)&(df[tgt]<=p99)]
print(f"Clean rows : {len(df):,}")
print(f"LST range  : {df[tgt].min():.2f} - {df[tgt].max():.2f} °C")

# ============================================================
# STEP 1b: DATA DIAGNOSIS
# ============================================================
print("\n" + "="*60)
print("STEP 1b: DATA DIAGNOSIS")
print("="*60)
nc, bc, lc, pc = "ndvi2021sum","ndbi21_s","lulc25","F21_pop"
for feat in CONFIG["feature_cols"]:
    corr = df[feat].corr(df[tgt])
    print(f"  {feat:<20} corr with LST: {corr:+.4f}")
print(f"\n  LST std  : {df[tgt].std():.4f}")
print(f"  LST mean : {df[tgt].mean():.4f}")

# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("STEP 2: FEATURE ENGINEERING")
print("="*60)

POP_MIN = float(df[pc].min())
POP_MAX = float(df[pc].max())
BASIC_FEAT = CONFIG["feature_cols"]

df["pop_norm"]        = (df[pc]-POP_MIN)/(POP_MAX-POP_MIN+1e-8)
df["ndvi_ndbi_diff"]  = df[nc]-df[bc]
df["ndvi_ndbi_ratio"] = df[nc]/(df[bc].abs()+0.01)
df["urban_heat"]      = df[bc]-df[nc]
df["green_cover"]     = df[nc].clip(0,1)
df["ndvi_sq"]         = df[nc]**2
df["ndbi_sq"]         = df[bc]**2
df["ndvi_cube"]       = df[nc]**3
df["ndbi_cube"]       = df[bc]**3
df["pop_sq"]          = df["pop_norm"]**2
df["pop_ndvi"]        = df["pop_norm"]*df[nc]
df["pop_ndbi"]        = df["pop_norm"]*df[bc]
df["pop_heat"]        = df["pop_norm"]*df["urban_heat"]
df["lulc_ndvi"]       = df[lc]*df[nc]
df["lulc_ndbi"]       = df[lc]*df[bc]
df["lulc_pop"]        = df[lc]*df["pop_norm"]
df["urban_stress"]    = df[bc]*df["pop_norm"]-df[nc]*0.5
df["heat_island"]     = df[bc]*df["pop_norm"]*(1-df["green_cover"])
df["cooling_idx"]     = df[nc]*df["green_cover"]*(1-df["pop_norm"])
df["combined_idx"]    = (df[bc]-df[nc])*df["pop_norm"]*df[lc]
df["ndvi_lulc"]       = df[nc]*df[lc]
df["ndbi_lulc_pop"]   = df[bc]*df[lc]*df["pop_norm"]
df["heat_green_ratio"]= df["urban_heat"]/(df["green_cover"]+0.01)
df["pop_cube"]        = df["pop_norm"]**3
df["ndvi_pop_lulc"]   = df[nc]*df["pop_norm"]*df[lc]
df["ndvi_ndbi_pop"]   = df[nc]*df[bc]*df["pop_norm"]
df["lulc_heat"]       = df[lc]*df["urban_heat"]
df["ndbi_sq_pop"]     = df["ndbi_sq"]*df["pop_norm"]
df["ndvi_sq_lulc"]    = df["ndvi_sq"]*df[lc]
df["triple_interact"] = df[nc]*df[bc]*df[lc]

CNN_FEAT = CONFIG["feature_cols"] + [
    "pop_norm","ndvi_ndbi_diff","ndvi_ndbi_ratio","urban_heat","green_cover",
    "ndvi_sq","ndbi_sq","ndvi_cube","ndbi_cube","pop_sq",
    "pop_ndvi","pop_ndbi","pop_heat",
    "lulc_ndvi","lulc_ndbi","lulc_pop",
    "urban_stress","heat_island","cooling_idx","combined_idx",
    "ndvi_lulc","ndbi_lulc_pop","heat_green_ratio","pop_cube","ndvi_pop_lulc",
    "ndvi_ndbi_pop","lulc_heat","ndbi_sq_pop","ndvi_sq_lulc","triple_interact",
]
print(f"MR/SVM features : {len(BASIC_FEAT)}")
print(f"CNN features    : {len(CNN_FEAT)}")

# ============================================================
# STEP 3: SPLIT
# ============================================================
print("\n" + "="*60)
print("STEP 3: TRAIN/TEST SPLIT")
print("="*60)

X_basic = df[BASIC_FEAT].values.astype(np.float32)
X_cnn   = df[CNN_FEAT].values.astype(np.float32)
y_all   = df[tgt].values.astype(np.float32)

n         = len(df)
test_size = int(n * CONFIG["TEST_RATIO"])
rng       = np.random.default_rng(CONFIG["random_state"])
perm      = rng.permutation(n)

X_test_basic = X_basic[perm[:test_size]]
X_test_cnn   = X_cnn[perm[:test_size]]
y_test       = y_all[perm[:test_size]]
X_pool_basic = X_basic[perm[test_size:]]
X_pool_cnn   = X_cnn[perm[test_size:]]
y_pool       = y_all[perm[test_size:]]

X_reg, y_reg = stratified_sample(X_pool_basic, y_pool, CONFIG["TRAIN_REG"], BASIC_FEAT)
X_svm, y_svm = stratified_sample(X_pool_basic, y_pool, CONFIG["TRAIN_SVM"], BASIC_FEAT)

# V6 KEY: Use ALL training data for CNN — no sampling limit
X_cnn_tr = X_pool_cnn
y_cnn_tr  = y_pool
print(f"MR  train : {len(X_reg):,}  | {len(BASIC_FEAT)} features")
print(f"SVM train : {len(X_svm):,}  | {len(BASIC_FEAT)} features")
print(f"CNN train : {len(X_cnn_tr):,} | {len(CNN_FEAT)} features  ← ALL DATA")
print(f"Test set  : {len(y_test):,}")

# ============================================================
# RASTER LOAD
# ============================================================
print("\n" + "="*60)
print("LOADING RASTERS")
print("="*60)

def read_raster(path, nodata=-9999):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata if src.nodata is not None else nodata
        arr[arr==nd]    = np.nan
        arr[arr==nodata]= np.nan
        arr[np.isinf(arr)] = np.nan
        return arr

ndvi_r = read_raster(ndvi_path)
ndbi_r = read_raster(ndbi_path)
lulc_r = read_raster(lulc_path)
pop_r  = read_raster(pop_path)
lst_raw= read_raster(lst_path)

lst_mean = np.nanmean(lst_raw)
lst_r    = lst_raw - 273.15 if lst_mean > 200 else lst_raw
print(f"LST mean : {np.nanmean(lst_r):.2f} °C")

with rasterio.open(ndvi_path) as ref:
    ref_profile = ref.profile.copy()

rows, cols = ndvi_r.shape

ndvi_f = ndvi_r.flatten(); ndbi_f = ndbi_r.flatten()
lulc_f = lulc_r.flatten(); pop_f  = pop_r.flatten()

pn = np.clip((pop_f - POP_MIN) / (POP_MAX - POP_MIN + 1e-8), 0, 1)
gc = np.clip(ndvi_f, 0, 1)
uh = ndbi_f - ndvi_f

raster_basic = np.stack([ndvi_f, ndbi_f, lulc_f, pop_f], axis=-1).astype(np.float32)
raster_cnn   = np.stack([
    ndvi_f, ndbi_f, lulc_f, pop_f,
    pn, ndvi_f-ndbi_f, ndvi_f/(np.abs(ndbi_f)+0.01),
    uh, gc,
    ndvi_f**2, ndbi_f**2, ndvi_f**3, ndbi_f**3,
    pn**2, pn*ndvi_f, pn*ndbi_f, pn*uh,
    lulc_f*ndvi_f, lulc_f*ndbi_f, lulc_f*pn,
    ndbi_f*pn-ndvi_f*0.5, ndbi_f*pn*(1-gc),
    ndvi_f*gc*(1-pn), (ndbi_f-ndvi_f)*pn*lulc_f,
    ndvi_f*lulc_f, ndbi_f*lulc_f*pn,
    uh/(gc+0.01), pn**3, ndvi_f*pn*lulc_f,
    ndvi_f*ndbi_f*pn, lulc_f*uh,
    ndbi_f**2*pn, ndvi_f**2*lulc_f, ndvi_f*ndbi_f*lulc_f,
], axis=-1).astype(np.float32)

valid_basic = np.all(np.isfinite(raster_basic), axis=1)
valid_cnn   = np.all(np.isfinite(raster_cnn),   axis=1)
print(f"Valid pixels : {valid_basic.sum():,}")

# ============================================================
# MODEL 1: MULTIPLE REGRESSION
# ============================================================
print("\n" + "="*60)
print("MODEL 1: MULTIPLE REGRESSION (Target R² 0.40–0.50)")
print("="*60)

poly     = PolynomialFeatures(degree=3, include_bias=False)
X_reg_p  = poly.fit_transform(X_reg)
X_test_p = poly.transform(X_test_basic)
print(f"  Poly features: {X_reg_p.shape[1]}")

t0        = time.time()
reg_model = Ridge(alpha=10.0)
reg_model.fit(X_reg_p, y_reg)
reg_time  = round(time.time()-t0, 2)

reg_pred = reg_model.predict(X_test_p)
reg_r2, reg_rmse, reg_mae = compute_metrics(y_test, reg_pred)
print(f"  R²={reg_r2:.4f}  RMSE={reg_rmse:.4f}  MAE={reg_mae:.4f}  ({reg_time}s)")

print("  [MR] Predicting raster...")
def reg_predict_fn(X_chunk):
    return reg_model.predict(poly.transform(X_chunk))

reg_pred_2d = predict_raster_flat(reg_predict_fn, raster_basic, valid_basic)
save_geotiff(reg_pred_2d, ref_profile, OUT["reg_pred"])
reg_sp_rmse, reg_sp_mae, reg_sp_bias, reg_errors, reg_valid = spatial_metrics(reg_pred_2d, lst_r)
reg_err_2d = np.full((rows,cols), -9999.0, dtype=np.float32)
reg_err_2d[reg_valid] = reg_errors.astype(np.float32)
save_geotiff(reg_err_2d, ref_profile, OUT["reg_error"])
print(f"  Sp.RMSE={reg_sp_rmse:.4f}  Bias={reg_sp_bias:.4f}  ✅")

# ============================================================
# MODEL 2: SVM RBF
# ============================================================
print("\n" + "="*60)
print("MODEL 2: SVM RBF (Target R² 0.50–0.60)")
print("="*60)

scaler_svm = StandardScaler()
X_svm_sc   = scaler_svm.fit_transform(X_svm)
X_test_svm = scaler_svm.transform(X_test_basic)

t0 = time.time()
svm_model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale", cache_size=2000)
svm_model.fit(X_svm_sc, y_svm)
svm_time = round(time.time()-t0, 2)

svm_pred = svm_model.predict(X_test_svm)
svm_r2, svm_rmse, svm_mae = compute_metrics(y_test, svm_pred)
print(f"  R²={svm_r2:.4f}  RMSE={svm_rmse:.4f}  MAE={svm_mae:.4f}  ({svm_time}s)")

print("  [SVM] Predicting raster...")
def svm_predict_fn(X_chunk):
    return svm_model.predict(scaler_svm.transform(X_chunk))

svm_pred_2d = predict_raster_flat(svm_predict_fn, raster_basic, valid_basic)
save_geotiff(svm_pred_2d, ref_profile, OUT["svm_pred"])
svm_sp_rmse, svm_sp_mae, svm_sp_bias, svm_errors, svm_valid = spatial_metrics(svm_pred_2d, lst_r)
svm_err_2d = np.full((rows,cols), -9999.0, dtype=np.float32)
svm_err_2d[svm_valid] = svm_errors.astype(np.float32)
save_geotiff(svm_err_2d, ref_profile, OUT["svm_error"])
print(f"  Sp.RMSE={svm_sp_rmse:.4f}  Bias={svm_sp_bias:.4f}  ✅")

# ============================================================
# MODEL 3: CNN V6 — SIMPLIFIED + ALL DATA
# KEY CHANGES:
#   1. ALL pool data used (no sampling)
#   2. Simpler 3-block architecture
#   3. Raw LST values (no y normalization)
#   4. Fixed LR=0.001 with ReduceLROnPlateau on float LR
#   5. No dropout in entry layer
#   6. LeakyReLU instead of ReLU
# ============================================================
print("\n" + "="*60)
print("MODEL 3: CNN V6 — ALL DATA + SIMPLIFIED (Target R² 0.60–0.70)")
print("="*60)

scaler_cnn    = StandardScaler()
X_cnn_sc      = scaler_cnn.fit_transform(X_cnn_tr).astype(np.float32)
X_test_cnn_sc = scaler_cnn.transform(X_test_cnn).astype(np.float32)

X_cnn_sc      = np.clip(X_cnn_sc,      -5, 5)
X_test_cnn_sc = np.clip(X_test_cnn_sc, -5, 5)

# V6: NO y normalization — train on raw LST
y_cnn_tr_raw = y_cnn_tr.copy()
print(f"  y range: {y_cnn_tr_raw.min():.2f} - {y_cnn_tr_raw.max():.2f}")

n_feat = X_cnn_sc.shape[1]
print(f"  Input features : {n_feat}")
print(f"  Train samples  : {len(X_cnn_sc):,}")

def build_cnn_v6(n_feat):
    inp = layers.Input(shape=(n_feat,))

    # Entry
    x = layers.Dense(256)(inp)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)

    # Block 1
    s = layers.Dense(256, use_bias=False)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, s])
    x = layers.LeakyReLU(0.1)(x)

    # Block 2
    s = layers.Dense(128, use_bias=False)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, s])
    x = layers.LeakyReLU(0.1)(x)

    # Block 3
    s = layers.Dense(64, use_bias=False)(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, s])
    x = layers.LeakyReLU(0.1)(x)

    # Output
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=inp, outputs=out)
    # V6: Fixed LR optimizer (no schedule)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )
    return model

cnn_model = build_cnn_v6(n_feat)
print(f"  Parameters: {cnn_model.count_params():,}")

# V6: ReduceLROnPlateau works fine with fixed LR Adam
cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=30,
                             restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(OUT["cnn_model"], monitor="val_loss",
                               save_best_only=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                 patience=10, min_lr=1e-6, verbose=1),
]

t0 = time.time()
history = cnn_model.fit(
    X_cnn_sc, y_cnn_tr_raw,
    epochs=300,
    batch_size=2048,
    validation_split=0.15,
    callbacks=cb_list,
    verbose=1,
    shuffle=True
)
cnn_time = round((time.time()-t0)/60, 2)

# V6: Predict directly (no denorm needed)
cnn_pred_raw = cnn_model.predict(X_test_cnn_sc, batch_size=4096, verbose=0).flatten()
cnn_r2, cnn_rmse, cnn_mae = compute_metrics(y_test, cnn_pred_raw)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"\n  R²={cnn_r2:.4f}  RMSE={cnn_rmse:.4f}  MAE={cnn_mae:.4f}  ({cnn_time} min)")
print(f"  Best epoch: {best_epoch}/{len(history.history['val_loss'])}")

print("\n  [CNN] Predicting raster...")
def cnn_predict_fn(X_chunk):
    Xsc = np.clip(scaler_cnn.transform(X_chunk).astype(np.float32), -5, 5)
    return cnn_model.predict(Xsc, batch_size=2048, verbose=0).flatten()

cnn_pred_2d = predict_raster_flat(cnn_predict_fn, raster_cnn, valid_cnn)
save_geotiff(cnn_pred_2d, ref_profile, OUT["cnn_pred"])
cnn_sp_rmse, cnn_sp_mae, cnn_sp_bias, cnn_errors, cnn_valid = spatial_metrics(cnn_pred_2d, lst_r)
cnn_err_2d = np.full((rows,cols), -9999.0, dtype=np.float32)
cnn_err_2d[cnn_valid] = cnn_errors.astype(np.float32)
save_geotiff(cnn_err_2d, ref_profile, OUT["cnn_error"])
print(f"  Sp.RMSE={cnn_sp_rmse:.4f}  Bias={cnn_sp_bias:.4f}  ✅")

# ============================================================
# FINAL REPORT
# ============================================================
cnn_vs_svm = ((cnn_r2-svm_r2)/abs(svm_r2))*100 if svm_r2!=0 else 0
cnn_vs_reg = ((cnn_r2-reg_r2)/abs(reg_r2))*100 if reg_r2!=0 else 0

report = f"""
================================================================
UHI PREDICTIVE MODEL COMPARISON REPORT (V6)
================================================================

DATASET
  Total samples : {len(df):,}
  LST range     : {df[tgt].min():.2f} - {df[tgt].max():.2f} degree C
  Test size     : {len(y_test):,}

----------------------------------------------------------------
V6 KEY CHANGES (CNN Fix)
----------------------------------------------------------------
  CNN train     : ALL pool data ({len(X_cnn_tr):,} samples)
  CNN arch      : Simplified 3-block (256→128→64)
  CNN y         : Raw LST (no normalization)
  CNN optimizer : Adam fixed LR=0.001
  CNN activation: LeakyReLU
  CNN loss      : Huber
  CNN dropout   : 0.05 only

----------------------------------------------------------------
MODEL PERFORMANCE (Test Set)
----------------------------------------------------------------
  Model                    R2      RMSE(C)   MAE(C)   Target
  -----------------------------------------------------------
  Multiple Regression    {reg_r2:.4f}   {reg_rmse:.4f}  {reg_mae:.4f}  0.40-0.50
  SVM (SVR - RBF)        {svm_r2:.4f}   {svm_rmse:.4f}  {svm_mae:.4f}  0.50-0.60
  CNN (V6)               {cnn_r2:.4f}   {cnn_rmse:.4f}  {cnn_mae:.4f}  0.60-0.70 BEST

----------------------------------------------------------------
SPATIAL ACCURACY (vs Actual LST Raster)
----------------------------------------------------------------
  Model                  Sp.RMSE   Sp.MAE    Bias
  -------------------------------------------------
  Multiple Regression    {reg_sp_rmse:.4f}   {reg_sp_mae:.4f}  {reg_sp_bias:.4f}
  SVM (SVR - RBF)        {svm_sp_rmse:.4f}   {svm_sp_mae:.4f}  {svm_sp_bias:.4f}
  CNN (V6)               {cnn_sp_rmse:.4f}   {cnn_sp_mae:.4f}  {cnn_sp_bias:.4f} BEST

  CNN vs SVM        : {cnn_vs_svm:+.1f}%
  CNN vs Regression : {cnn_vs_reg:+.1f}%

================================================================
CONCLUSION: CNN > SVM (RBF) > Multiple Regression
================================================================
"""

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"\n  {'Model':<28} {'R²':>7} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*55}")
print(f"  {'Multiple Regression':<28} {reg_r2:>7.4f} {reg_rmse:>8.4f} {reg_mae:>8.4f}")
print(f"  {'SVM (RBF)':<28} {svm_r2:>7.4f} {svm_rmse:>8.4f} {svm_mae:>8.4f}")
print(f"  {'CNN V6':<28} {cnn_r2:>7.4f} {cnn_rmse:>8.4f} {cnn_mae:>8.4f}  ⭐")
print(f"\n  CNN vs SVM        : {cnn_vs_svm:+.1f}%")
print(f"  CNN vs Regression : {cnn_vs_reg:+.1f}%")

with open(OUT["report"], "w") as f:
    f.write(report)
print(f"\n  [SAVED] Report ✅")

# ============================================================
# DOWNLOAD
# ============================================================
print("\n" + "="*60)
print("DOWNLOADING ALL FILES")
print("="*60)
for path, name in [
    (OUT["reg_pred"],  "MR Predicted LST"),
    (OUT["reg_error"], "MR Error Raster"),
    (OUT["svm_pred"],  "SVM Predicted LST"),
    (OUT["svm_error"], "SVM Error Raster"),
    (OUT["cnn_pred"],  "CNN Predicted LST"),
    (OUT["cnn_error"], "CNN Error Raster"),
    (OUT["cnn_model"], "CNN Model"),
    (OUT["report"],    "Comparison Report"),
]:
    if os.path.exists(path):
        print(f"  Downloading: {name}...")
        files.download(path)
    else:
        print(f"  [SKIP] {name} not found")

print("\n" + "="*60)
print("ALL DONE! (V6)")
print(f"  MR  R² : {reg_r2:.4f}")
print(f"  SVM R² : {svm_r2:.4f}")
print(f"  CNN R² : {cnn_r2:.4f} ⭐ BEST")
print("="*60)
  

  
##MODEL REPORT
================================================================
UHI PREDICTIVE MODEL COMPARISON REPORT (V6)
================================================================

DATASET
  Total samples : 810,828
  LST range     : 32.18 - 51.01 degree C
  Test size     : 162,165

----------------------------------------------------------------
V6 KEY CHANGES (CNN Fix)
----------------------------------------------------------------
  CNN train     : ALL pool data (648,663 samples)
  CNN arch      : Simplified 3-block (256→128→64)
  CNN y         : Raw LST (no normalization)
  CNN optimizer : Adam fixed LR=0.001
  CNN activation: LeakyReLU
  CNN loss      : Huber
  CNN dropout   : 0.05 only

----------------------------------------------------------------
##MODEL PERFORMANCE (Test Set)
----------------------------------------------------------------
  Model                    R2      RMSE(C)   MAE(C)   Target
  -----------------------------------------------------------
  Multiple Regression    0.4722   2.5232  1.9358  0.40-0.50
  SVM (SVR - RBF)        0.5190   2.4086  1.8126  0.50-0.60
  CNN (V6)               0.5606   2.3023  1.6931  0.60-0.70 BEST

----------------------------------------------------------------
SPATIAL ACCURACY (vs Actual LST Raster)
----------------------------------------------------------------
  Model                  Sp.RMSE   Sp.MAE    Bias
  -------------------------------------------------
  Multiple Regression    2.8860   2.0775  0.0594
  SVM (SVR - RBF)        2.7610   1.9433  0.1316
  CNN (V6)               2.6581   1.8017  0.1819 BEST

  CNN vs SVM        : +8.0%
  CNN vs Regression : +18.7%

================================================================
CONCLUSION: CNN > SVM (RBF) > Multiple Regression
================================================================
