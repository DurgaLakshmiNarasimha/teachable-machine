"""Teachable Machine core: learn image classes from user-provided examples."""
from __future__ import annotations

import io
from collections import Counter
from typing import Dict, List

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIZE = 32
BINS = 8


def to_image(src) -> Image.Image:
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    return Image.open(src).convert("RGB")  # file path / file-like


def features(src) -> np.ndarray:
    """Downscaled pixels + per-channel colour histogram + edge-strength histogram."""
    img = to_image(src)
    small = np.asarray(img.resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    pix = np.asarray(img.resize((16, 16)), dtype=np.float32).ravel() / 255.0
    hist = np.concatenate([np.histogram(small[..., c], bins=BINS, range=(0, 1))[0] for c in range(3)])
    hist = hist / hist.sum() * 3
    gray = small.mean(axis=2)
    gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    edge = np.histogram(np.concatenate([gx.ravel(), gy.ravel()]), bins=BINS, range=(0, 1))[0]
    edge = edge / max(edge.sum(), 1)
    return np.concatenate([pix, hist, edge])


class TeachableMachine:
    def __init__(self):
        self.model = None
        self.classes: List[str] = []

    def train(self, data: Dict[str, list]) -> dict:
        """data: {class_name: [image sources]} -> returns training summary."""
        X, y = [], []
        for label, items in data.items():
            for it in items:
                X.append(features(it)); y.append(label)
        counts = Counter(y)
        if len(counts) < 2:
            raise ValueError("Add at least 2 classes with examples.")
        if min(counts.values()) < 2:
            raise ValueError("Each class needs at least 2 examples.")
        X, y = np.array(X), np.array(y)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        self.model.fit(X, y)
        self.classes = list(self.model.classes_)
        cv = min(3, min(counts.values()))
        acc = float(cross_val_score(self.model, X, y, cv=cv).mean()) if cv >= 2 else None
        return {"samples": len(y), "classes": dict(counts), "cv_accuracy": acc}

    def predict(self, src) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Train the model first.")
        p = self.model.predict_proba([features(src)])[0]
        return dict(sorted(zip(self.classes, map(float, p)), key=lambda kv: -kv[1]))

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str) -> "TeachableMachine":
        blob = joblib.load(path)
        tm = cls(); tm.model, tm.classes = blob["model"], blob["classes"]
        return tm


def load_folder(root: str) -> Dict[str, list]:
    """Dataset layout: root/<class_name>/*.jpg|png"""
    import os
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return {d: [os.path.join(root, d, f) for f in sorted(os.listdir(os.path.join(root, d))) if f.lower().endswith(exts)]
            for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))}
