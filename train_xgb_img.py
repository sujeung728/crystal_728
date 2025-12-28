import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import xgboost as xgb

# =========================
# 설정
# =========================
CSV_PATH = Path("HAM10000_metadata.csv")
IMG_DIR  = Path("image")
OUT_DIR  = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32

MALIGNANT = {"mel", "bcc", "akiec"}

# =========================
# 데이터 로드
# =========================
df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image_id"].astype(str).apply(
    lambda x: str(IMG_DIR / f"{x}.jpg")
)

df = df[df["image_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

y = df["dx"].astype(str).isin(MALIGNANT).astype(int).values
paths = df["image_path"].values

print("N:", len(df))
print("Pos:", y.sum(), "Neg:", (y == 0).sum())

# =========================
# CNN Feature Extractor (저장 ❌)
# =========================
fe = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(224, 224, 3),
)
fe.trainable = False

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img

ds = (
    tf.data.Dataset.from_tensor_slices(paths)
    .map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# =========================
# 임베딩 추출
# =========================
embeds = fe.predict(ds, verbose=1)
print("Embedding shape:", embeds.shape)

# =========================
# train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    embeds, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# XGBoost
# =========================
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 4,
    "eta": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=50,
)

# =========================
# 평가
# =========================
proba = bst.predict(dvalid)
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("PR-AUC :", average_precision_score(y_test, proba))

# =========================
# 저장 (XGBoost만!)
# =========================
bst.save_model(str(OUT_DIR / "xgb_img_embed.json"))
print("✅ Saved:", OUT_DIR / "xgb_img_embed.json")
