"""
All implementations must give the same answer as the original serial code,
to floating-point round-off (float64), for the same data and initialization.

Summation order necessarily differs between a Python loop, BLAS, and a GPU
reduction, so literal bit equality of every float is not achievable; we assert
agreement to ~1e-10 relative, whereas the observed differences are ~1e-15
(eigenvectors) / ~1e-13 (coefficients).  Observed maxima are checked tightly in
``test_observed_error_is_roundoff``.
"""
import numpy as np
import pytest

import empca as serial
import empca_auto
import empca_numpy

try:
    import empca_torch
    import torch
    HAS_TORCH = True
except ImportError:                                  # pragma: no cover
    empca_torch = None
    HAS_TORCH = False

from conftest import make_problem

RTOL, ATOL = 1e-10, 1e-11
ROUNDOFF = 1e-12      # tight bound on the eigenvectors (unit norm); observed ~4e-15


def _impls():
    out = [pytest.param(('numpy', empca_numpy.empca, {}), id='numpy')]
    if HAS_TORCH:
        out.append(pytest.param(('torch-cpu', empca_torch.empca, dict(device='cpu')), id='torch-cpu'))
        out.append(pytest.param(
            ('torch-cuda', empca_torch.empca, dict(device='cuda')), id='torch-cuda',
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')))
    return out


IMPLS = _impls()


def assert_models_equal(ref, got, rtol=RTOL, atol=ATOL):
    for name in ('eigvec', 'coeff', 'model'):
        r = getattr(ref, name)
        # absolute tolerance scaled to the array's magnitude: rank-deficient rows amplify round-off
        np.testing.assert_allclose(getattr(got, name), r, rtol=rtol, atol=max(atol, 1e-11 * np.abs(r).max()),
                                   err_msg=name)
    assert got.dof == ref.dof
    assert got.nobs == ref.nobs and got.nvar == ref.nvar and got.nvec == ref.nvec
    np.testing.assert_allclose(got.chi2(), ref.chi2(), rtol=rtol)
    np.testing.assert_allclose(got.rchi2(), ref.rchi2(), rtol=rtol)
    np.testing.assert_allclose(got.R2(), ref.R2(), rtol=rtol, atol=atol)
    for i in range(ref.nvec):
        np.testing.assert_allclose(got.R2vec(i), ref.R2vec(i), rtol=rtol, atol=atol)
    np.testing.assert_allclose(got.R2(min(2, ref.nvec)), ref.R2(min(2, ref.nvec)), rtol=rtol, atol=atol)


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('kw', [
    dict(nvec=3, niter=15),
    dict(nvec=1, niter=10),
    dict(nvec=5, niter=25, randseed=7),
    dict(nvec=3, niter=10, smooth=7),
], ids=['nvec3', 'nvec1', 'nvec5-seed7', 'smoothed'])
def test_empca_matches_serial(impl, kw):
    _, fn, extra = impl
    data, weights = make_problem(nobs=60, nvar=80)
    ref = serial.empca(data, weights, silent=True, **kw)
    got = fn(data, weights, silent=True, **dict(kw, **extra))
    assert_models_equal(ref, got)


@pytest.mark.parametrize('impl', IMPLS)
def test_default_weights_none(impl):
    _, fn, extra = impl
    data, _ = make_problem(nobs=40, nvar=50)
    assert_models_equal(serial.empca(data, None, nvec=3, niter=8, silent=True),
                        fn(data, None, nvec=3, niter=8, silent=True, **extra))


@pytest.mark.parametrize('impl', IMPLS)
def test_all_rows_uniform_weights(impl):
    """Every row takes the unweighted-projection branch of the serial solver."""
    _, fn, extra = impl
    data, _ = make_problem(nobs=30, nvar=40)
    w = np.tile(np.linspace(1, 3, 30)[:, None], (1, 40))
    assert_models_equal(serial.empca(data, w, nvec=3, niter=8, silent=True),
                        fn(data, w, nvec=3, niter=8, silent=True, **extra))


@pytest.mark.parametrize('impl', IMPLS)
def test_crosses_internal_block_boundaries(impl):
    """Large enough that the implementations process several internal row blocks."""
    _, fn, extra = impl
    data, weights = make_problem(nobs=523, nvar=701, nvec=4, seed=3)
    ref = serial.empca(data, weights, nvec=4, niter=4, silent=True)
    assert_models_equal(ref, fn(data, weights, nvec=4, niter=4, silent=True, **extra))


@pytest.mark.parametrize('impl', IMPLS)
def test_observed_error_is_roundoff(impl):
    _, fn, extra = impl
    data, weights = make_problem(nobs=120, nvar=150, nvec=4, seed=5)
    ref = serial.empca(data, weights, nvec=4, niter=25, silent=True)
    got = fn(data, weights, nvec=4, niter=25, silent=True, **extra)
    assert np.max(np.abs(got.eigvec - ref.eigvec)) < ROUNDOFF
    assert np.max(np.abs(got.coeff - ref.coeff)) / np.max(np.abs(ref.coeff)) < 1e-11
    assert np.max(np.abs(got.model - ref.model)) / np.max(np.abs(ref.model)) < 1e-11


@pytest.mark.parametrize('impl', IMPLS)
def test_modelwise_steps_match(impl):
    """Individual Model methods (not just the full loop) agree from the same state."""
    name, fn, extra = impl
    data, weights = make_problem(nobs=50, nvar=64)
    V0 = serial._random_orthonormal(4, 64, seed=2)
    if name == 'numpy':
        ModelCls, kw = empca_numpy.Model, {}
    else:
        ModelCls, kw = empca_torch.Model, extra
    ref = serial.Model(V0.copy(), data, weights)
    got = ModelCls(V0.copy(), data, weights, **kw)
    np.testing.assert_allclose(got.coeff, ref.coeff, rtol=RTOL, atol=ATOL)     # set_data -> solve_coeffs
    for _ in range(3):
        ref.solve_coeffs()
        got.solve_coeffs()
        np.testing.assert_allclose(got.coeff, ref.coeff, rtol=RTOL, atol=ATOL)
        ref.solve_eigenvectors()
        got.solve_eigenvectors()
        np.testing.assert_allclose(got.eigvec, ref.eigvec, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(got.model, ref.model, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('impl', IMPLS)
def test_set_data_applies_model_to_new_data(impl):
    _, fn, extra = impl
    data, weights = make_problem(nobs=50, nvar=64, seed=1)
    new_data, new_weights = make_problem(nobs=23, nvar=64, seed=9)
    ref = serial.empca(data, weights, nvec=3, niter=6, silent=True)
    got = fn(data, weights, nvec=3, niter=6, silent=True, **extra)
    ref.set_data(new_data, new_weights)
    got.set_data(new_data, new_weights)
    assert got.coeff.shape == (23, 3)
    assert_models_equal(ref, got)


@pytest.mark.parametrize('impl', IMPLS)
def test_api_surface(impl):
    _, fn, extra = impl
    data, weights = make_problem(nobs=30, nvar=40)
    m = fn(data, weights, nvec=2, niter=3, silent=True, **extra)
    for attr in ('eigvec', 'data', 'weights', 'coeff', 'model', 'dof', 'nobs', 'nvar', 'nvec'):
        assert hasattr(m, attr), attr
    for meth in ('solve_coeffs', 'solve_eigenvectors', 'solve_model', 'chi2', 'rchi2', 'R2', 'R2vec', 'set_data'):
        assert callable(getattr(m, meth)), meth
    assert isinstance(m.eigvec, np.ndarray) and isinstance(m.coeff, np.ndarray) and isinstance(m.model, np.ndarray)
    assert m.eigvec.shape == (2, 40) and m.coeff.shape == (30, 2) and m.model.shape == (30, 40)
    np.testing.assert_allclose(m.eigvec @ m.eigvec.T, np.eye(2), atol=1e-12)


@pytest.mark.parametrize('impl', IMPLS)
def test_verbose_output_matches(impl, capsys):
    _, fn, extra = impl
    data, weights = make_problem(nobs=40, nvar=50)
    serial.empca(data, weights, nvec=3, niter=4)
    ref_out = capsys.readouterr().out
    fn(data, weights, nvec=3, niter=4, **extra)
    got_out = capsys.readouterr().out
    # R2 / rchi2 are printed to 8 decimals; they must agree line by line
    assert got_out.splitlines() == ref_out.splitlines()


def test_empca_auto_every_backend_matches_serial():
    data, weights = make_problem(nobs=50, nvar=60)
    ref = serial.empca(data, weights, nvec=3, niter=8, silent=True)
    for b in empca_auto.available_backends():
        got = empca_auto.empca(data, weights, nvec=3, niter=8, silent=True, backend=b)
        assert got.backend == b
        assert_models_equal(ref, got)
