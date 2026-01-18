import numpy as np
import pytest
from pkoffee.metrics import (
    SizeMismatchError,
    check_size_match,
    compute_mae,
    compute_rmse,
)


def test_compute_rmse() -> None:
    """Test the compute_rmse function."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])
    rmse = compute_rmse(y_true, y_pred)
    expected_rmse = 0.1
    assert np.isclose(rmse, expected_rmse), f"Expected RMSE {expected_rmse}, got {rmse}"


def test_check_size_match() -> None:
    """Test the check_size_match function."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8])

    check_size_match(a, b)

    try:
        check_size_match(a, c)
    except SizeMismatchError as e:
        assert str(e) == "Arrays must have same length, got 3 and 2"
    else:
        assert False, "SizeMismatchError was not raised"
