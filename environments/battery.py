"""
Battery model for distributed energy storage systems (DESS).

Action convention:
    action > 0  -> discharge
    action < 0  -> charge
    action = 0  -> idle

Power sign convention:
    power_kw > 0  -> DESS injects active power into the grid
    power_kw < 0  -> DESS absorbs active power from the grid

State-of-charge convention:
    SOC is stored as a fraction in [0, 1].

The model enforces:
    - charging power limit,
    - discharging power limit,
    - minimum SOC,
    - maximum SOC.

The default efficiency is 0.95.
"""


import numpy as np


class BatteryConfig:
    """
    Configuration container for a single DESS unit.
    """

    def __init__(
        self,
        capacity_kwh,
        max_charge_kw,
        max_discharge_kw,
        efficiency=0.95,
        charge_efficiency=None,
        discharge_efficiency=None,
        soc_min=0.20,
        soc_max=0.80,
        soc_init=0.40,
    ):
        self.capacity_kwh = float(capacity_kwh)
        self.max_charge_kw = float(max_charge_kw)
        self.max_discharge_kw = float(max_discharge_kw)

        if charge_efficiency is None:
            charge_efficiency = efficiency

        if discharge_efficiency is None:
            discharge_efficiency = efficiency

        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)

        # Backward compatibility with older code.
        self.efficiency = float(efficiency)

        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)
        self.soc_init = float(soc_init)

        self._validate()

    def _validate(self):
        if self.capacity_kwh <= 0.0:
            raise ValueError("capacity_kwh must be positive.")

        if self.max_charge_kw < 0.0:
            raise ValueError("max_charge_kw must be nonnegative.")

        if self.max_discharge_kw < 0.0:
            raise ValueError("max_discharge_kw must be nonnegative.")

        if not (0.0 < self.charge_efficiency <= 1.0):
            raise ValueError("charge_efficiency must be in the interval (0, 1].")

        if not (0.0 < self.discharge_efficiency <= 1.0):
            raise ValueError("discharge_efficiency must be in the interval (0, 1].")

        if not (0.0 <= self.soc_min <= self.soc_max <= 1.0):
            raise ValueError("SOC limits must satisfy 0 <= soc_min <= soc_max <= 1.")

        if not (self.soc_min <= self.soc_init <= self.soc_max):
            raise ValueError("soc_init must be between soc_min and soc_max.")


class Battery:
    """
    DESS battery model with hard SOC and power-limit enforcement.
    """

    def __init__(self, config, delta_t_hours):
        self.config = config
        self.delta_t_hours = float(delta_t_hours)

        if self.delta_t_hours <= 0.0:
            raise ValueError("delta_t_hours must be positive.")

        self.reset()

    def reset(self):
        """
        Reset the battery to its initial SOC.

        Returns
        -------
        float
            Initial state of charge.
        """

        self.soc = float(self.config.soc_init)
        self.last_power_kw = 0.0
        self.last_energy_kwh = 0.0
        self.last_requested_power_kw = 0.0
        self.last_action = 0.0
        self.last_was_limited = False

        return self.soc

    def step(self, action):
        """
        Apply one normalized battery action.

        Parameters
        ----------
        action : float
            Normalized action in [-1, 1].

        Returns
        -------
        dict
            Battery transition information.
        """

        action = float(np.clip(action, -1.0, 1.0))
        self.last_action = action

        if action > 0.0:
            requested_power_kw = action * self.config.max_discharge_kw
            actual_power_kw = self._apply_discharge(requested_power_kw)

        elif action < 0.0:
            requested_power_kw = action * self.config.max_charge_kw
            actual_power_kw = self._apply_charge(requested_power_kw)

        else:
            requested_power_kw = 0.0
            actual_power_kw = 0.0

        self.last_requested_power_kw = float(requested_power_kw)
        self.last_power_kw = float(actual_power_kw)
        self.last_energy_kwh = float(actual_power_kw * self.delta_t_hours)

        self.last_was_limited = not np.isclose(
            abs(self.last_power_kw),
            abs(self.last_requested_power_kw),
            rtol=1e-6,
            atol=1e-9,
        )

        return {
            "soc": self.soc,
            "power_kw": self.last_power_kw,
            "energy_kwh": self.last_energy_kwh,
            "requested_power_kw": self.last_requested_power_kw,
            "was_limited": self.last_was_limited,
        }

    def _apply_discharge(self, requested_power_kw):
        """
        Apply discharging action.

        Positive DESS power is active-power injection into the grid.

        SOC update:
            E_removed = P_dis * delta_t / eta_dis
            SOC_next = SOC_current - E_removed / E_capacity
        """

        requested_power_kw = max(0.0, float(requested_power_kw))

        available_energy_kwh = (
            self.soc - self.config.soc_min
        ) * self.config.capacity_kwh

        max_feasible_discharge_kw = (
            available_energy_kwh * self.config.discharge_efficiency
        ) / self.delta_t_hours

        actual_power_kw = min(
            requested_power_kw,
            self.config.max_discharge_kw,
            max_feasible_discharge_kw,
        )

        energy_removed_kwh = (
            actual_power_kw * self.delta_t_hours
        ) / self.config.discharge_efficiency

        self.soc -= energy_removed_kwh / self.config.capacity_kwh
        self.soc = float(np.clip(self.soc, self.config.soc_min, self.config.soc_max))

        return actual_power_kw

    def _apply_charge(self, requested_power_kw):
        """
        Apply charging action.

        Negative DESS power is active-power absorption from the grid.

        SOC update:
            E_added = P_ch * delta_t * eta_ch
            SOC_next = SOC_current + E_added / E_capacity
        """

        requested_charge_kw = abs(float(requested_power_kw))

        available_capacity_kwh = (
            self.config.soc_max - self.soc
        ) * self.config.capacity_kwh

        max_feasible_charge_kw = (
            available_capacity_kwh / self.config.charge_efficiency
        ) / self.delta_t_hours

        actual_charge_kw = min(
            requested_charge_kw,
            self.config.max_charge_kw,
            max_feasible_charge_kw,
        )

        energy_added_kwh = (
            actual_charge_kw
            * self.delta_t_hours
            * self.config.charge_efficiency
        )

        self.soc += energy_added_kwh / self.config.capacity_kwh
        self.soc = float(np.clip(self.soc, self.config.soc_min, self.config.soc_max))

        return -actual_charge_kw

    def get_soc(self):
        return self.soc

    def get_power_kw(self):
        return self.last_power_kw

    def get_energy_kwh(self):
        return self.last_energy_kwh

    def get_available_discharge_kw(self):
        available_energy_kwh = (
            self.soc - self.config.soc_min
        ) * self.config.capacity_kwh

        return min(
            self.config.max_discharge_kw,
            (available_energy_kwh * self.config.discharge_efficiency)
            / self.delta_t_hours,
        )

    def get_available_charge_kw(self):
        available_capacity_kwh = (
            self.config.soc_max - self.soc
        ) * self.config.capacity_kwh

        return min(
            self.config.max_charge_kw,
            (available_capacity_kwh / self.config.charge_efficiency)
            / self.delta_t_hours,
        )