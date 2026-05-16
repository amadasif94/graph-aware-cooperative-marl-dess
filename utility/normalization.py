"""
Normalization utilities for MARL-DESS observations.

These functions normalize ONLY neural-network observations.

Physical simulation quantities inside:
    - power flow,
    - battery dynamics,
    - feasibility checks,
    - rewards,
remain in real physical units.
"""

import numpy as np


class ObservationNormalizer:
    """
    Static normalization helper for graph node features and agent observations.

    Important:
        Normalized features are scaled but NOT clipped.
        This preserves information when values exceed the expected range.
    """

    def __init__(
        self,
        max_time_index=95.0,
        max_price=300.0,
        max_load_kw=4000.0,
        max_pv_kw=2500.0,
        max_grid_import_kw=4000.0,
        max_voltage_deviation=1.0,
        max_line_current_pu=5.0,
    ):
        self.max_time_index = float(max_time_index)
        self.max_price = float(max_price)
        self.max_load_kw = float(max_load_kw)
        self.max_pv_kw = float(max_pv_kw)
        self.max_grid_import_kw = float(max_grid_import_kw)
        self.max_voltage_deviation = float(max_voltage_deviation)
        self.max_line_current_pu = float(max_line_current_pu)

    @staticmethod
    def safe_divide(x, scale):
        scale = max(float(scale), 1e-12)
        return np.asarray(x, dtype=np.float32) / scale

    def normalize_time_index(self, t):
        # Time is naturally bounded inside one episode, so clipping is acceptable here.
        return np.clip(float(t) / self.max_time_index, 0.0, 1.0)

    def normalize_price(self, price):
        return self.safe_divide(price, self.max_price)

    def normalize_load_kw(self, load_kw):
        return self.safe_divide(load_kw, self.max_load_kw)

    def normalize_pv_kw(self, pv_kw):
        return self.safe_divide(pv_kw, self.max_pv_kw)

    def normalize_grid_import_kw(self, grid_import_kw):
        return self.safe_divide(grid_import_kw, self.max_grid_import_kw)

    @staticmethod
    def normalize_voltage_pu(voltage_pu):
        return np.asarray(voltage_pu, dtype=np.float32)

    @staticmethod
    def normalize_soc(soc):
        return np.asarray(soc, dtype=np.float32)

    def normalize_voltage_deviation(self, voltage_deviation):
        return self.safe_divide(voltage_deviation, self.max_voltage_deviation)

    def normalize_line_current_pu(self, current_pu):
        return self.safe_divide(current_pu, self.max_line_current_pu)