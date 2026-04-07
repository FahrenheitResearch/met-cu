from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cupy as cp

from metpy.cbook import get_test_data
from metpy.units import units


def load_oun_sounding():
    col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]
    sounding_data = pd.read_fwf(
        get_test_data("20110522_OUN_12Z.txt", as_file_obj=False),
        skiprows=7,
        usecols=[0, 1, 2, 3, 6, 7],
        names=col_names,
    )
    sounding_data = sounding_data.dropna(
        subset=("temperature", "dewpoint", "direction", "speed"),
        how="all",
    ).reset_index(drop=True)

    pressure = sounding_data["pressure"].values * units.hPa
    temperature = sounding_data["temperature"].values * units.degC
    dewpoint = sounding_data["dewpoint"].values * units.degC
    height = sounding_data["height"].values * units.meter
    speed = sounding_data["speed"].values * units.knots
    direction = sounding_data["direction"].values * units.degrees

    return {
        "pressure": pressure,
        "temperature": temperature,
        "dewpoint": dewpoint,
        "height": height,
        "speed": speed,
        "direction": direction,
    }


def compare_quantity(actual, expected):
    actual_units = str(actual.units)
    expected_units = str(expected.units)
    if (
        "degree_Celsius" in actual_units
        and "delta_degree_Celsius" in expected_units
    ) or (
        "delta_degree_Celsius" in actual_units
        and "degree_Celsius" in expected_units
    ):
        return (
            np.asarray(actual.magnitude, dtype=np.float64),
            np.asarray(expected.magnitude, dtype=np.float64),
        )

    try:
        return (
            np.asarray(actual.to(expected.units).magnitude, dtype=np.float64),
            np.asarray(expected.magnitude, dtype=np.float64),
        )
    except Exception:
        return (
            np.asarray(actual.to_base_units().magnitude, dtype=np.float64),
            np.asarray(expected.to_base_units().magnitude, dtype=np.float64),
        )


def to_numpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if hasattr(value, "magnitude"):
        value = value.magnitude
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray(value)


def assert_runtime_close(actual, expected, atol):
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert isinstance(expected, (list, tuple))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_runtime_close(actual_item, expected_item, atol)
        return

    if isinstance(actual, xr.DataArray) and isinstance(expected, xr.DataArray):
        assert actual.dims == expected.dims
        assert_runtime_close(actual.data, expected.data, atol)
        return

    if hasattr(actual, "to") and hasattr(expected, "to"):
        actual_arr, expected_arr = compare_quantity(actual, expected)
    else:
        actual_arr = to_numpy(actual)
        if hasattr(expected, "magnitude"):
            expected_arr = np.asarray(expected.magnitude, dtype=np.float64)
        else:
            try:
                expected_arr = np.asarray(expected, dtype=np.float64)
            except (TypeError, ValueError):
                expected_arr = np.asarray(expected)

    actual_arr = np.atleast_1d(actual_arr)
    expected_arr = np.atleast_1d(expected_arr)
    assert actual_arr.shape == expected_arr.shape
    if actual_arr.dtype.kind not in "fcui" or expected_arr.dtype.kind not in "fcui":
        np.testing.assert_array_equal(actual_arr, expected_arr)
        return
    np.testing.assert_allclose(
        actual_arr,
        expected_arr,
        atol=atol,
        rtol=atol,
        equal_nan=True,
    )


def assert_dataset_close(actual, expected, atol):
    assert dict(actual.sizes) == dict(expected.sizes)
    assert set(actual.coords) == set(expected.coords)
    assert set(actual.data_vars) == set(expected.data_vars)

    for coord_name in expected.coords:
        np.testing.assert_allclose(
            np.asarray(actual[coord_name].values, dtype=np.float64),
            np.asarray(expected[coord_name].values, dtype=np.float64),
            atol=atol,
            rtol=atol,
            equal_nan=True,
        )
        for attr_name, attr_value in expected[coord_name].attrs.items():
            assert actual[coord_name].attrs.get(attr_name) == attr_value

    for var_name in expected.data_vars:
        assert_runtime_close(actual[var_name].data, expected[var_name].data, atol)
        for attr_name, attr_value in expected[var_name].attrs.items():
            assert actual[var_name].attrs.get(attr_name) == attr_value


@pytest.fixture(scope="module")
def sounding_profile():
    return load_oun_sounding()
