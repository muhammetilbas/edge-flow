"""
models/trainer.py — Model Eğitim Pipeline
XGBoost + LightGBM eğitimi, walk-forward CV, model kaydetme.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from features.technical import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/saved")

# Hedef değişken değerleri
TARGET_BUY = 1
TARGET_HOLD = 0
TARGET_SELL = -1

# XGBoost için etiket dönüşümü (0, 1, 2)
LABEL_MAP = {TARGET_SELL: 0, TARGET_HOLD: 1, TARGET_BUY: 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def prepare_target(
    df: pd.DataFrame,
    forward_days: int = 3,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Target değişkeni oluşturur.

    Args:
        df:           OHLCV + feature DataFrame'i
        forward_days: İleri dönük kontrol süresi (gün)
        threshold:    Hedef yüzde eşiği (%2 = 0.02)

    Returns:
        'target' sütunu eklenmiş DataFrame (NaN'lar temizlenmiş)
    """
    df = df.copy()
    future_return = df["close"].shift(-forward_days) / df["close"] - 1

    df["target"] = TARGET_HOLD
    df.loc[future_return > threshold, "target"] = TARGET_BUY
    df.loc[future_return < -threshold, "target"] = TARGET_SELL

    logger.info(
        f"Target dağılımı: "
        f"BUY={int((df['target'] == TARGET_BUY).sum())}, "
        f"HOLD={int((df['target'] == TARGET_HOLD).sum())}, "
        f"SELL={int((df['target'] == TARGET_SELL).sum())}"
    )
    return df.dropna()


class ModelTrainer:
    """XGBoost ve LightGBM model eğitim pipeline.

    Walk-forward time-series cross-validation ile eğitilir,
    ardından en iyi model models/saved/ altına kaydedilir.
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _build_xgb(self) -> "xgb.XGBClassifier":
        """XGBoost sınıflandırıcısını oluşturur."""
        if not XGB_AVAILABLE:
            raise ImportError("xgboost kurulu değil.")
        return xgb.XGBClassifier(
            n_estimators=self._cfg.get("n_estimators", 300),
            max_depth=self._cfg.get("max_depth", 4),
            learning_rate=self._cfg.get("learning_rate", 0.05),
            subsample=self._cfg.get("subsample", 0.8),
            colsample_bytree=self._cfg.get("colsample_bytree", 0.8),
            random_state=self._cfg.get("random_state", 42),
            n_jobs=-1,
            eval_metric="mlogloss",
            verbosity=0,
        )

    def _build_lgb(self) -> "lgb.LGBMClassifier":
        """LightGBM sınıflandırıcısını oluşturur."""
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm kurulu değil.")
        return lgb.LGBMClassifier(
            n_estimators=self._cfg.get("n_estimators", 300),
            max_depth=self._cfg.get("max_depth", 4),
            learning_rate=self._cfg.get("learning_rate", 0.05),
            subsample=self._cfg.get("subsample", 0.8),
            colsample_bytree=self._cfg.get("colsample_bytree", 0.8),
            random_state=self._cfg.get("random_state", 42),
            n_jobs=-1,
            verbose=-1,
        )

    def train(
        self,
        df: pd.DataFrame,
        ticker: str,
        model_type: str = "xgb",
        forward_days: int = 3,
        threshold: float = 0.02,
        n_splits: int = 5,
    ) -> tuple:
        """Modeli eğitir ve kaydeder.

        Args:
            df:           Feature + close sütunlu DataFrame
            ticker:       Hisse sembolü (model adı için)
            model_type:   'xgb' veya 'lgb'
            forward_days: Hedef gün sayısı
            threshold:    Hareket eşiği
            n_splits:     Walk-forward CV split sayısı

        Returns:
            (model, metrics_dict) tuple
        """
        # Target oluştur
        df = prepare_target(df, forward_days=forward_days, threshold=threshold)

        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(available_features) < 5:
            raise ValueError(f"Yetersiz feature ({len(available_features)}). Min 5 gerekli.")

        X = df[available_features]
        y = df["target"]

        # XGBoost için etiket dönüşümü (0,1,2)
        y_encoded = y.map(LABEL_MAP)

        # Walk-forward CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        logger.info(f"{ticker} için {model_type.upper()} eğitimi başlıyor...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

            model_fold = self._build_xgb() if model_type == "xgb" else self._build_lgb()
            model_fold.fit(X_train, y_train)

            preds = model_fold.predict(X_val)
            score = accuracy_score(y_val, preds)
            cv_scores.append(score)
            logger.debug(f"  Fold {fold + 1}/{n_splits}: Accuracy={score:.4f}")

        avg_cv_score = float(np.mean(cv_scores))
        logger.info(f"Walk-forward CV ortalama accuracy: {avg_cv_score:.4f}")

        # Tüm veriyle son model
        final_model = self._build_xgb() if model_type == "xgb" else self._build_lgb()
        final_model.fit(X, y_encoded)

        # Test son %20 veriyle değerlendirme
        test_size = max(int(len(X) * 0.2), 10)
        X_test = X.iloc[-test_size:]
        y_test = y_encoded.iloc[-test_size:]
        test_preds = final_model.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, test_preds))

        metrics = {
            "ticker": ticker,
            "model_type": model_type,
            "cv_accuracy": round(avg_cv_score, 4),
            "test_accuracy": round(test_accuracy, 4),
            "cv_scores": [round(s, 4) for s in cv_scores],
            "features_used": available_features,
            "n_samples": len(df),
            "trained_at": datetime.now().isoformat(),
        }
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Model kaydet
        model_path = self._save_model(final_model, ticker, model_type, metrics)
        metrics["model_path"] = str(model_path)

        return final_model, metrics

    def _save_model(
        self, model, ticker: str, model_type: str, metrics: dict
    ) -> Path:
        """Modeli ve metriklerini pickle ile kaydeder."""
        filename = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
        path = MODEL_DIR / filename

        payload = {"model": model, "metrics": metrics, "label_map": LABEL_MAP_INV}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Model kaydedildi: {path}")
        return path

    @staticmethod
    def load_model(path: str) -> tuple:
        """Kaydedilmiş modeli yükler.

        Returns:
            (model, metrics, label_map_inv) tuple
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        logger.info(f"Model yüklendi: {path}")
        return payload["model"], payload["metrics"], payload["label_map"]

    def get_latest_model_path(self, ticker: str, model_type: str = "xgb") -> Optional[str]:
        """En son eğitilmiş model dosyasını bulur."""
        pattern = f"{ticker}_{model_type}_*.pkl"
        files = sorted(MODEL_DIR.glob(pattern))
        if not files:
            return None
        return str(files[-1])
