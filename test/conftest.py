import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import empca_auto  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_dispatch_state(tmp_path, monkeypatch):
    """Never let a real calibration file or EMPCA_BACKEND from the environment leak into tests."""
    monkeypatch.setenv('EMPCA_CALIBRATION_FILE', str(tmp_path / 'calibration.json'))
    monkeypatch.delenv('EMPCA_BACKEND', raising=False)
    empca_auto.clear_calibration_cache()
    yield
    empca_auto.clear_calibration_cache()
    empca_auto.refresh_environment()


def make_problem(nobs=60, nvar=80, nvec=3, seed=0, mask_frac=0.2, noise=0.1):
    """Low-rank signal + noise with heteroscedastic weights, masked pixels and some special rows."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 2 * np.pi, nvar)
    data = sum(rng.normal(size=(nobs, 1)) * np.sin(x * (k + 1)) for k in range(nvec))
    data = data + noise * rng.normal(size=(nobs, nvar))
    weights = rng.uniform(0.5, 2.0, size=(nobs, nvar))
    weights[rng.uniform(size=(nobs, nvar)) < mask_frac] = 0.0
    if nobs >= 6:
        weights[0] = 1.0            # uniform row  -> plain projection path
        weights[1] = 0.37           # uniform row with value != 1
        weights[2] = 0.0            # fully masked row
        weights[3] = 0.0
        weights[3, [4, 9]] = 1.0    # rank-deficient row (2 pixels, nvec > 2)
    return data, weights
