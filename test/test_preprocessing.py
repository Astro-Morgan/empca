"""Residual-mode (mean_as_vector=False) mean / mean-derivative subtraction and the protected random start vectors."""
import numpy as np
import pytest

import empca as serial
import empca_auto
import empca_numpy

from conftest import make_problem
from test_equivalence import IMPLS, assert_models_equal


def spectra(nobs=80, nvar=120, seed=0, dead_pixel=False):
    """Spectra = template + redshift-like shifts of it + low-rank structure + noise."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, nvar)
    template = 3 + np.exp(-((x - 0.5) / 0.08) ** 2) + 0.5 * np.sin(12 * x)
    shift = rng.normal(scale=0.004, size=nobs)
    data = np.array([np.interp(x + s, x, template) for s in shift])
    data += rng.normal(size=(nobs, 1)) * np.sin(5 * x) + 0.05 * rng.normal(size=(nobs, nvar))
    w = rng.uniform(0.5, 2, size=(nobs, nvar))
    w[rng.uniform(size=w.shape) < 0.15] = 0
    if dead_pixel:
        w[:, 7] = 0                               # a pixel nobody measured
    return data, w


# --------------------------------------------------------------------------- preprocess()
def test_mean_subtraction_is_weighted_and_leaves_zero_weighted_mean():
    data, w = spectra(dead_pixel=True)
    pre = empca_numpy.preprocess(data, w, subtract_mean=True)
    expect = np.array([(w[:, j] * data[:, j]).sum() / w[:, j].sum() if w[:, j].sum() > 0 else 0.0
                       for j in range(data.shape[1])])
    np.testing.assert_allclose(pre.mean, expect, rtol=1e-12)
    assert pre.mean[7] == 0.0 and np.all(np.isfinite(pre.residual))
    np.testing.assert_allclose((w * pre.residual).sum(0), 0, atol=1e-9)
    assert pre.mean_deriv is None and pre.deriv_coeff is None


def test_derivative_subtraction_orthogonal_to_mean_derivative():
    data, w = spectra()
    pre = empca_numpy.preprocess(data, w, subtract_mean=True, subtract_mean_deriv=True)
    np.testing.assert_allclose(pre.mean_deriv, np.gradient(pre.mean))
    # per-object weighted projection onto d(mean) is gone from the residual
    np.testing.assert_allclose((w * pre.residual) @ pre.mean_deriv, 0, atol=1e-9)
    # and it removed something: coefficients track the injected shifts, variance drops
    assert np.std(pre.deriv_coeff) > 0
    assert np.sum(w * pre.residual ** 2) < np.sum(w * (data - pre.mean) ** 2)


def test_decomposition_reconstructs_original():
    data, w = spectra()
    pre = empca_numpy.preprocess(data, w, subtract_mean=True, subtract_mean_deriv=True)
    np.testing.assert_allclose(pre.residual + pre.baseline(), data, atol=1e-12)


def test_derivative_with_explicit_grid():
    data, w = spectra()
    grid = np.log(np.linspace(3000, 9000, data.shape[1]))
    pre = empca_numpy.preprocess(data, w, True, True, deriv_grid=grid)
    np.testing.assert_allclose(pre.mean_deriv, np.gradient(pre.mean, grid))


def test_derivative_requires_mean():
    data, w = spectra()
    with pytest.raises(ValueError):
        empca_numpy.preprocess(data, w, subtract_mean=False, subtract_mean_deriv=True)


def test_no_preprocessing_is_identity():
    data, w = spectra()
    pre = empca_numpy.preprocess(data, w)
    assert pre.residual is data or np.array_equal(pre.residual, data)
    np.testing.assert_array_equal(pre.baseline(), 0)


# --------------------------------------------------------------------------- all backends
OPTIONS = [dict(subtract_mean=True, mean_as_vector=False),
           dict(subtract_mean=True, subtract_mean_deriv=True, mean_as_vector=False)]


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('opts', OPTIONS, ids=['mean', 'mean+deriv'])
def test_backends_match_serial_on_preprocessed_data(impl, opts):
    _, fn, extra = impl
    data, w = spectra()
    pre = empca_numpy.preprocess(data, w, opts['subtract_mean'], opts.get('subtract_mean_deriv', False))
    ref = serial.empca(pre.residual, w, nvec=3, niter=12, silent=True)       # serial has no native option
    got = fn(data, w, nvec=3, niter=12, silent=True, **opts, **extra)
    assert_models_equal(ref, got)
    np.testing.assert_allclose(got.mean, pre.mean)
    np.testing.assert_allclose(got.original_data, data)
    np.testing.assert_allclose(got.data, pre.residual)
    # full model = baseline + EMPCA model of the residuals
    np.testing.assert_allclose(got.full_model(), pre.baseline() + ref.model, rtol=1e-10, atol=1e-10)
    if 'subtract_mean_deriv' in opts:
        np.testing.assert_allclose(got.deriv_coeff, pre.deriv_coeff)


@pytest.mark.parametrize('opts', OPTIONS, ids=['mean', 'mean+deriv'])
def test_auto_all_backends_agree_with_preprocessing(opts):
    data, w = spectra()
    ref = empca_auto.empca(data, w, nvec=3, niter=8, silent=True, backend='serial', **opts)
    for b in empca_auto.available_backends():
        got = empca_auto.empca(data, w, nvec=3, niter=8, silent=True, backend=b, **opts)
        assert_models_equal(ref, got)
        np.testing.assert_allclose(got.full_model(), ref.full_model(), rtol=1e-10, atol=1e-10)


def test_mean_subtraction_improves_fit_quality_in_few_vectors():
    """Without mean removal the first vector is spent on the mean; with it the same nvec fits better."""
    data, w = spectra()
    plain = empca_numpy.empca(data, w, nvec=2, niter=15, silent=True)
    sub = empca_numpy.empca(data, w, nvec=2, niter=15, silent=True, subtract_mean=True,
                              mean_as_vector=False)
    chi2_plain = np.sum(w * (plain.model - data) ** 2)
    chi2_sub = np.sum(w * (sub.full_model() - data) ** 2)
    assert chi2_sub < chi2_plain


def test_defaults_have_no_preprocessing_attributes():
    data, w = make_problem(nobs=30, nvar=40)
    m = empca_numpy.empca(data, w, nvec=2, niter=2, silent=True)
    assert not hasattr(m, 'full_model')


# --------------------------------------------------------------------------- random_orthonormal
@pytest.mark.parametrize('nvec,nvar,seed', [(1, 10, 1), (5, 50, 1), (8, 9, 3), (4, 200, None)])
def test_random_orthonormal_matches_serial_when_serial_works(nvec, nvar, seed):
    if seed is None:
        np.random.seed(11)
    a = serial._random_orthonormal(nvec, nvar, seed=seed)
    if seed is None:
        np.random.seed(11)
    b = empca_numpy.random_orthonormal(nvec, nvar, seed=seed)
    np.testing.assert_array_equal(a, b)                       # exactly the same numbers
    np.testing.assert_allclose(b @ b.T, np.eye(nvec), atol=1e-12)


def _duplicate_first_draw(monkeypatch, nvec, nvar, dup_of=0, row=1, zero=False):
    """Make the first np.random.normal call return a degenerate matrix, later calls behave normally."""
    real = np.random.normal
    state = {'n': 0}

    def fake(*a, **k):
        out = real(*a, **k)
        if state['n'] == 0:
            out = out.copy()
            out[row] = 0.0 if zero else out[dup_of]
        state['n'] += 1
        return out
    monkeypatch.setattr(np.random, 'normal', fake)


@pytest.mark.parametrize('zero', [False, True], ids=['duplicate', 'zero'])
def test_random_orthonormal_survives_degenerate_draw(monkeypatch, zero):
    _duplicate_first_draw(monkeypatch, 4, 30, zero=zero)
    bad = serial._random_orthonormal(4, 30, seed=2)
    # serial breaks: either NaNs (zero vector) or a roundoff-noise direction that is not orthogonal
    with np.errstate(all='ignore'):
        defect = np.max(np.abs(bad @ bad.T - np.eye(4)))
    assert not np.isfinite(defect) or defect > 1e-6

    monkeypatch.undo()
    _duplicate_first_draw(monkeypatch, 4, 30, zero=zero)
    good = empca_numpy.random_orthonormal(4, 30, seed=2)
    assert np.all(np.isfinite(good))
    np.testing.assert_allclose(good @ good.T, np.eye(4), atol=1e-12)


def test_random_orthonormal_rejects_impossible_request():
    with pytest.raises(ValueError):
        empca_numpy.random_orthonormal(6, 5)


def test_full_basis_is_possible():
    q = empca_numpy.random_orthonormal(12, 12, seed=4)
    np.testing.assert_allclose(q @ q.T, np.eye(12), atol=1e-10)


@pytest.mark.parametrize('impl', IMPLS)
def test_pixel_with_zero_total_weight_is_finite(impl):
    """The serial code returns NaN everywhere here; vectorized backends leave that pixel at 0."""
    _, fn, extra = impl
    data, w = spectra(dead_pixel=True)
    m = fn(data, w, nvec=3, niter=10, silent=True, subtract_mean=True, subtract_mean_deriv=True, **extra)
    assert np.all(np.isfinite(m.eigvec)) and np.all(np.isfinite(m.coeff)) and np.all(np.isfinite(m.model))
    np.testing.assert_allclose(m.eigvec[m.nfixed:, 7], 0, atol=1e-9)   # learned rows: 0 up to re-orthogonalisation against the fixed vectors
    assert m.R2() > 0.5
