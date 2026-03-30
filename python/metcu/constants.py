"""Physical constants for meteorological calculations."""

# Fundamental constants
Rd = 287.04  # J/(kg·K) — dry air gas constant
Rv = 461.5   # J/(kg·K) — water vapor gas constant
Cp_d = 1004.0  # J/(kg·K) — specific heat of dry air at constant pressure
Cv_d = 717.0   # J/(kg·K) — specific heat of dry air at constant volume
Lv = 2.501e6   # J/kg — latent heat of vaporization at 0°C
Ls = 2.834e6   # J/kg — latent heat of sublimation
Lf = 3.34e5    # J/kg — latent heat of fusion
g = 9.80665    # m/s² — gravitational acceleration
epsilon = Rd / Rv  # 0.6219569...
kappa = Rd / Cp_d  # 0.28571...
ZEROCNK = 273.15   # Kelvin offset

# Earth
EARTH_RADIUS = 6371229.0  # meters
OMEGA = 7.2921159e-5  # rad/s — Earth's angular velocity
