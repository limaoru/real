"""高级客流预测（sklearn 回归，无依赖时回退均值）"""
from datetime import datetime
from typing import Optional

import numpy as np

from retail.config.settings import ENABLE_ADVANCED_FORECAST


class AdvancedForecaster:
    def __init__(self):
        self._model = None
        self._fitted = False

    def _fit_if_needed(self, hourly: dict):
        if not ENABLE_ADVANCED_FORECAST or len(hourly) < 4:
            return
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            return
        hours = sorted(hourly.keys())
        y = np.array([hourly[h] for h in hours], dtype=np.float32)
        x = np.array([int(h.split(":")[0]) for h in hours], dtype=np.float32).reshape(-1, 1)
        self._model = Ridge(alpha=1.0)
        self._model.fit(x, y)
        self._fitted = True

    def predict_next_hour(self, hourly: dict) -> int:
        if not hourly:
            return 0
        self._fit_if_needed(hourly)
        next_h = (datetime.now().hour + 1) % 24
        if self._fitted and self._model is not None:
            pred = float(self._model.predict(np.array([[next_h]], dtype=np.float32))[0])
            return max(0, int(round(pred * 1.05)))
        vals = list(hourly.values())
        return max(0, int(sum(vals) / len(vals) * 1.1))

    def predict_rest_of_day(self, hourly: dict) -> int:
        if not hourly:
            return 0
        cur = datetime.now().hour
        remaining_hours = max(1, 20 - cur)
        return self.predict_next_hour(hourly) * remaining_hours
