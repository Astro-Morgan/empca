"""
Mean (and mean-derivative) as fixed leading basis vectors, fitted jointly with
the learned vectors (empca(..., subtract_mean=True), the default mean_as_vector=True).

The reference is a plain-loop implementation built on the serial Model, so the
vectorized backends are checked against independent code.
"""
import json

import numpy as np
import pytest

import empca as serial
import empca_auto
import empca_numpy

from test_equivalence import IMPLS, assert_models_equal
from test_preprocessing import spectra

MODES = [
    pytest.param(dict(subtract_mean=True), id='mean'),
    pytest.param(dict(subtract_mean=True, subtract_mean_deriv=True), id='mean+deriv'),
]


class LoopFixedModel(serial.Model):
    """Reference: serial loops, ``nfixed`` leading rows frozen."""

    def __init__(self, eigvec, data, weights, nfixed):
        self.nfixed = nfixed
        serial.Model.__init__(self, eigvec, data, weights)
        self.dof += nfixed * self.nvar        # fixed vectors are not fitted parameters

    def solve_eigenvectors(self, smooth=None):
        data = self.data.copy()
        for k in range(self.nvec):
            c = self.coeff[:, k]
            if k >= self.nfixed:
                for j in range(self.nvar):
                    w, x = self.weights[:, j], data[:, j]
                    cw = c * w
                    den = c.dot(cw)
                    self.eigvec[k, j] = x.dot(cw) / den if den > 0 else 0.0
                if smooth is not None:
                    self.eigvec[k] = smooth(self.eigvec[k])
            data -= np.outer(c, self.eigvec[k])
        for k in range(self.nfixed, self.nvec):
            if k == 0:
                self.eigvec[0] /= np.sqrt(np.dot(self.eigvec[0], self.eigvec[0]))
                continue
            for kx in range(k):
                self.eigvec[k] -= np.dot(self.eigvec[k], self.eigvec[kx]) * self.eigvec[kx]
            self.eigvec[k] /= np.sqrt(np.dot(self.eigvec[k], self.eigvec[k]))
        self.solve_model()


def reference(data, w, nvec, niter, opts, include_fixed=True, smooth=0, seed=1):
    setup = empca_numpy._Setup(data, w, nvec, opts['subtract_mean'], opts.get('subtract_mean_deriv', False),
                               None, True, include_fixed)
    sm = serial.SavitzkyGolay(width=smooth) if smooth > 0 else None
    m = LoopFixedModel(setup.initial_eigvec(data.shape[1], seed), data, w, setup.nfixed)
    for _ in range(niter):
        m.solve_coeffs()
        m.solve_eigenvectors(smooth=sm)
    m.solve_coeffs()
    return m, setup


# --------------------------------------------------------------------------- correctness
@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('opts', MODES)
@pytest.mark.parametrize('include_fixed,nvec', [(True, 5), (False, 3)], ids=['nvec-total', 'nvec-extra'])
def test_matches_loop_reference(impl, opts, include_fixed, nvec):
    _, fn, extra = impl
    data, w = spectra()
    ref, setup = reference(data, w, nvec, 12, opts, include_fixed)
    got = fn(data, w, nvec=nvec, niter=12, silent=True, include_fixed_in_nvec=include_fixed, **opts, **extra)
    assert got.nfixed == ref.nfixed == setup.nfixed
    assert got.eigvec.shape == ref.eigvec.shape
    assert_models_equal(ref, got)


@pytest.mark.parametrize('impl', IMPLS)
def test_matches_loop_reference_with_smoothing(impl):
    _, fn, extra = impl
    data, w = spectra()
    opts = dict(subtract_mean=True, subtract_mean_deriv=True)
    ref, _ = reference(data, w, 4, 8, opts, smooth=9)
    got = fn(data, w, nvec=4, niter=8, smooth=9, silent=True, **opts, **extra)
    assert_models_equal(ref, got)


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('opts', MODES)
def test_fixed_rows_are_frozen_and_basis_is_orthonormal(impl, opts):
    _, fn, extra = impl
    data, w = spectra()
    F, mean, dmean = empca_numpy.fixed_basis(data, w, opts.get('subtract_mean_deriv', False))
    m = fn(data, w, nvec=5, niter=10, silent=True, **opts, **extra)
    np.testing.assert_array_equal(m.eigvec[:len(F)], F)                       # untouched, exactly
    np.testing.assert_allclose(m.eigvec @ m.eigvec.T, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(m.eigvec[0], mean / np.linalg.norm(mean))
    np.testing.assert_allclose(m.mean, mean)
    if dmean is not None:
        np.testing.assert_allclose(m.mean_deriv, dmean)
        assert abs(np.dot(m.eigvec[1], m.eigvec[0])) < 1e-12


@pytest.mark.parametrize('impl', IMPLS)
def test_nvec_semantics(impl):
    _, fn, extra = impl
    data, w = spectra()
    kw = dict(niter=3, silent=True, subtract_mean=True, **extra)
    assert fn(data, w, nvec=4, **kw).eigvec.shape[0] == 4                                   # mean + 3 learned
    assert fn(data, w, nvec=4, include_fixed_in_nvec=False, **kw).eigvec.shape[0] == 5      # mean + 4 learned
    m = fn(data, w, nvec=4, subtract_mean_deriv=True, **kw)
    assert m.nfixed == 2 and m.eigvec.shape[0] == 4
    m = fn(data, w, nvec=4, subtract_mean_deriv=True, include_fixed_in_nvec=False, **kw)
    assert m.eigvec.shape[0] == 6
    only_fixed = fn(data, w, nvec=1, **kw)                                                  # mean alone: coefficients only
    assert only_fixed.eigvec.shape[0] == 1 and np.all(np.isfinite(only_fixed.coeff))
    with pytest.raises(ValueError):
        fn(data, w, nvec=1, subtract_mean_deriv=True, **kw)                                 # 2 fixed > nvec=1


@pytest.mark.parametrize('impl', IMPLS)
def test_dof_counts_only_learned_parameters(impl):
    _, fn, extra = impl
    data, w = spectra()
    m = fn(data, w, nvec=5, niter=2, silent=True, subtract_mean=True, subtract_mean_deriv=True, **extra)
    n_learn, n_tot = 3, 5
    assert m.dof == int(np.sum(w > 0)) - n_learn * data.shape[1] - n_tot * data.shape[0]


@pytest.mark.parametrize('impl', IMPLS)
def test_model_is_of_raw_data_and_set_data_takes_raw_data(impl):
    """No preprocessing step is needed for new data: the mean is part of the basis."""
    _, fn, extra = impl
    data, w = spectra(nobs=80)
    new, new_w = spectra(nobs=30, seed=5)
    m = fn(data, w, nvec=4, niter=6, silent=True, subtract_mean=True, **extra)
    np.testing.assert_array_equal(m.data, data)
    np.testing.assert_array_equal(m.full_model(), m.model)
    m.set_data(new, new_w)
    assert m.coeff.shape == (30, 4) and m.model.shape == new.shape
    ref = LoopFixedModel(m.eigvec.copy(), new, new_w, 1)
    np.testing.assert_allclose(m.coeff, ref.coeff, rtol=1e-9, atol=1e-9)


def test_free_amplitude_mean_alone_explains_scaled_copies():
    """Objects are scaled copies of the mean: the fixed mean vector alone (nothing learned) absorbs that."""
    rng = np.random.RandomState(3)
    x = np.linspace(0, 1, 100)
    mean = 2 + np.sin(6 * x)
    amp = rng.uniform(0.3, 2.0, size=(60, 1))
    sigma = 0.01
    data = amp * mean + sigma * rng.normal(size=(60, 100))
    w = np.ones_like(data)
    m = empca_numpy.empca(data, w, nvec=1, niter=3, silent=True, subtract_mean=True)
    assert m.nfixed == 1 and m.eigvec.shape[0] == 1
    assert np.sum((m.model - data) ** 2) < 1.5 * sigma ** 2 * data.size
    np.testing.assert_allclose(m.coeff[:, 0] / np.linalg.norm(mean), amp[:, 0], rtol=0.05)


def test_default_is_vector_mode_and_residual_mode_still_available():
    data, w = spectra()
    vec = empca_numpy.empca(data, w, nvec=3, niter=3, silent=True, subtract_mean=True)
    res = empca_numpy.empca(data, w, nvec=3, niter=3, silent=True, subtract_mean=True, mean_as_vector=False)
    assert vec.nfixed == 1 and getattr(res, 'nfixed', 0) == 0
    assert not np.array_equal(res.data, data) and np.array_equal(vec.data, data)


def test_fixed_basis_validation():
    data, w = spectra()
    with pytest.raises(ValueError):
        empca_numpy.fixed_basis(np.zeros((5, 10)), np.ones((5, 10)))               # zero mean
    flat = np.tile(np.arange(10.0), (5, 1))                                         # d(mean) is constant, fine
    F, _, _ = empca_numpy.fixed_basis(flat, np.ones((5, 10)), True)
    np.testing.assert_allclose(F @ F.T, np.eye(2), atol=1e-12)
    parallel = np.tile(np.arange(1.0, 11.0), (5, 1))                                # mean = pixel index: gradient = ones
    ok, _, _ = empca_numpy.fixed_basis(parallel, np.ones((5, 10)), True)
    assert ok.shape == (2, 10)
    scaled = np.tile(np.ones(10), (5, 1))                                           # gradient of a constant is 0
    with pytest.raises(ValueError):
        empca_numpy.fixed_basis(scaled, np.ones((5, 10)), True)
    with pytest.raises(ValueError):
        empca_numpy.empca(data, w, nvec=3, subtract_mean=False, subtract_mean_deriv=True)


def test_random_orthonormal_with_fixed_rows():
    F = empca_numpy.random_orthonormal(2, 40, seed=9)
    A = empca_numpy.random_orthonormal(4, 40, seed=1, fixed=F)
    full = np.vstack([F, A])
    np.testing.assert_allclose(full @ full.T, np.eye(6), atol=1e-12)
    with pytest.raises(ValueError):
        empca_numpy.random_orthonormal(40, 40, fixed=F)


# --------------------------------------------------------------------------- dispatcher
def test_auto_never_runs_serial_in_vector_mode(monkeypatch):
    avail = empca_auto.available_backends()
    data, w = spectra(nobs=40, nvar=60)
    cost = {b: [0.0, 1.0] for b in avail}
    cost['serial'] = [0.0, 1e-6]                                  # table says serial is by far the fastest
    key = empca_auto._fingerprint_key()
    with open(empca_auto.calibration_path(), 'w') as f:
        json.dump({key: dict(fingerprint={}, key=key, records=[dict(nobs=40, nvar=60, nvec=3, cost=cost)])}, f)
    empca_auto.clear_calibration_cache()
    assert empca_auto.empca(data, w, nvec=3, niter=2, silent=True).backend == 'serial'            # no mean: allowed
    m = empca_auto.empca(data, w, nvec=3, niter=2, silent=True, subtract_mean=True)
    assert m.backend != 'serial' and m.nfixed == 1
    m = empca_auto.empca(data, w, nvec=3, niter=2, silent=True, subtract_mean=True, mean_as_vector=False)
    assert m.backend == 'serial'                                                                  # residual mode is fine


def test_auto_serial_explicit_in_vector_mode_raises():
    data, w = spectra(nobs=40, nvar=60)
    with pytest.raises(ValueError, match='fixed'):
        empca_auto.empca(data, w, nvec=3, silent=True, subtract_mean=True, backend='serial')


@pytest.mark.parametrize('opts', MODES)
def test_auto_backends_agree_in_vector_mode(opts):
    data, w = spectra()
    backends = [b for b in empca_auto.available_backends() if b != 'serial']
    ref = empca_auto.empca(data, w, nvec=4, niter=8, silent=True, backend=backends[0], **opts)
    for b in backends[1:]:
        got = empca_auto.empca(data, w, nvec=4, niter=8, silent=True, backend=b, **opts)
        assert_models_equal(ref, got)
