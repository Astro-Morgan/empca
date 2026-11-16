"""
Validation of the context-based implementation switching in empca_auto.

Selection logic is tested deterministically (injected calibration tables and
fake environments, no timing); a single test also runs a real, tiny
calibration end to end on this machine.
"""
import json

import numpy as np
import pytest

import empca as serial
import empca_auto as auto

from conftest import make_problem
from test_equivalence import assert_models_equal

ALL = ('serial', 'numpy', 'torch-cpu', 'torch-cuda')


def fake_env(monkeypatch, torch=True, cuda=True):
    monkeypatch.setitem(auto._env_cache, 'probe', (torch, cuda))


def rec(nobs, nvar, nvec, **cost):
    """cost: backend=(fixed, per_iter)"""
    return dict(nobs=nobs, nvar=nvar, nvec=nvec, cost={k.replace('_', '-'): list(v) for k, v in cost.items()})


def table(*records):
    return dict(fingerprint={}, key='x', records=list(records))


# --------------------------------------------------------------------------- availability
def test_available_backends_reflect_environment(monkeypatch):
    fake_env(monkeypatch, torch=False, cuda=False)
    assert auto.available_backends() == ('serial', 'numpy')
    fake_env(monkeypatch, torch=True, cuda=False)
    assert auto.available_backends() == ('serial', 'numpy', 'torch-cpu')
    fake_env(monkeypatch, torch=True, cuda=True)
    assert auto.available_backends() == ALL


def test_requesting_unavailable_backend_raises(monkeypatch):
    fake_env(monkeypatch, torch=False, cuda=False)
    with pytest.raises(RuntimeError, match='not available'):
        auto.resolve_backend(10, 10, 2, backend='torch-cuda')


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        auto.resolve_backend(10, 10, 2, backend='fortran')
    with pytest.raises(ValueError):
        with auto.use_backend('fortran'):
            pass


# --------------------------------------------------------------------------- calibrated selection
def test_table_nearest_point_decides(monkeypatch):
    fake_env(monkeypatch)
    t = table(
        rec(50, 50, 2, serial=(0, 1e-3), numpy=(0, 2e-3), torch_cpu=(0, 3e-3), torch_cuda=(0, 9e-3)),
        rec(1000, 1000, 5, serial=(0, 1.0), numpy=(0, 0.3), torch_cpu=(0, 0.1), torch_cuda=(0, 0.2)),
        rec(100000, 5000, 5, numpy=(0, 9.0), torch_cpu=(0, 4.0), torch_cuda=(0, 0.1)),
    )
    assert auto.choose_backend(60, 40, 2, table=t) == 'serial'
    assert auto.choose_backend(800, 1200, 5, table=t) == 'torch-cpu'
    assert auto.choose_backend(80000, 4000, 4, table=t) == 'torch-cuda'
    # exact grid points
    assert auto.choose_backend(50, 50, 2, table=t) == 'serial'
    assert auto.choose_backend(100000, 5000, 5, table=t) == 'torch-cuda'


def test_table_prediction_uses_iteration_count(monkeypatch):
    """High fixed (upload/launch) cost but low per-iteration cost: GPU only pays off for long runs."""
    fake_env(monkeypatch)
    t = table(rec(1000, 1000, 5, numpy=(0.0, 0.05), torch_cuda=(2.0, 0.01)))
    assert auto.choose_backend(1000, 1000, 5, niter=5, table=t) == 'numpy'
    assert auto.choose_backend(1000, 1000, 5, niter=500, table=t) == 'torch-cuda'


def test_table_respects_availability(monkeypatch):
    t = table(rec(1000, 1000, 5, numpy=(0, 0.5), torch_cpu=(0, 0.4), torch_cuda=(0, 0.01)))
    fake_env(monkeypatch, torch=True, cuda=True)
    assert auto.choose_backend(1000, 1000, 5, table=t) == 'torch-cuda'
    fake_env(monkeypatch, torch=True, cuda=False)
    assert auto.choose_backend(1000, 1000, 5, table=t) == 'torch-cpu'
    fake_env(monkeypatch, torch=False, cuda=False)
    assert auto.choose_backend(1000, 1000, 5, table=t) == 'numpy'


def test_candidates_restrict_choice(monkeypatch):
    fake_env(monkeypatch)
    t = table(rec(1000, 1000, 5, numpy=(0, 0.5), torch_cpu=(0, 0.4), torch_cuda=(0, 0.01)))
    assert auto.choose_backend(1000, 1000, 5, table=t, candidates=('numpy', 'torch-cpu')) == 'torch-cpu'
    with pytest.raises(RuntimeError):
        auto.choose_backend(1000, 1000, 5, table=t, candidates=())


def test_record_missing_a_backend_is_not_chosen(monkeypatch):
    """Serial is omitted from large grid points (too slow to time): it must never be picked there."""
    fake_env(monkeypatch)
    t = table(rec(10000, 4000, 20, numpy=(0, 5.0), torch_cpu=(0, 2.0), torch_cuda=(0, 0.2)))
    assert auto.choose_backend(9000, 3000, 20, table=t) == 'torch-cuda'


# --------------------------------------------------------------------------- heuristic fallback
def test_heuristic_small_problem_uses_numpy(monkeypatch):
    fake_env(monkeypatch)
    assert auto.choose_backend(30, 40, 3, table=False) == 'numpy'


def test_heuristic_large_problem_uses_gpu_when_present(monkeypatch):
    fake_env(monkeypatch, torch=True, cuda=True)
    assert auto.choose_backend(20000, 2000, 5, table=False) == 'torch-cuda'


def test_heuristic_large_problem_without_gpu(monkeypatch):
    fake_env(monkeypatch, torch=True, cuda=False)
    assert auto.choose_backend(20000, 2000, 5, table=False) in ('torch-cpu', 'numpy')
    fake_env(monkeypatch, torch=False, cuda=False)
    assert auto.choose_backend(20000, 2000, 5, table=False) == 'numpy'


def test_heuristic_wide_basis_still_uses_gpu(monkeypatch):
    """The Cholesky fast path keeps the GPU competitive for nvec > 32 (the old pinv path was not)."""
    fake_env(monkeypatch, torch=True, cuda=True)
    assert auto.choose_backend(20000, 2000, 64, table=False) == 'torch-cuda'


# --------------------------------------------------------------------------- override priority
def test_priority_argument_over_context_over_env_over_auto(monkeypatch):
    fake_env(monkeypatch)
    t = table(rec(100, 100, 3, numpy=(0, 0.01), torch_cpu=(0, 1), torch_cuda=(0, 1), serial=(0, 1)))
    json.dump({auto._fingerprint_key(): dict(t, key=auto._fingerprint_key())}, open(auto.calibration_path(), 'w'))
    auto.clear_calibration_cache()

    assert auto.resolve_backend(100, 100, 3) == 'numpy'                    # automatic (calibration)
    monkeypatch.setenv('EMPCA_BACKEND', 'torch-cpu')
    assert auto.resolve_backend(100, 100, 3) == 'torch-cpu'                # environment
    with auto.use_backend('serial'):
        assert auto.resolve_backend(100, 100, 3) == 'serial'               # context beats env
        assert auto.resolve_backend(100, 100, 3, backend='torch-cuda') == 'torch-cuda'   # argument beats context
        with auto.use_backend('auto'):
            assert auto.resolve_backend(100, 100, 3) == 'numpy'            # nested 'auto' re-enables selection
        assert auto.resolve_backend(100, 100, 3) == 'serial'               # restored after inner block
    assert auto.resolve_backend(100, 100, 3) == 'torch-cpu'                # context exited


# --------------------------------------------------------------------------- calibration storage
def test_calibration_ignored_for_other_machine(monkeypatch):
    fake_env(monkeypatch)
    other = table(rec(100, 100, 3, numpy=(0, 1.0), torch_cuda=(0, 0.001)))
    json.dump({'some-other-machine': dict(other, key='some-other-machine')}, open(auto.calibration_path(), 'w'))
    auto.clear_calibration_cache()
    assert auto.load_calibration() is None
    assert auto.choose_backend(100, 100, 3) == 'numpy'     # fell back to the heuristic


def test_corrupt_calibration_file_is_ignored():
    with open(auto.calibration_path(), 'w') as f:
        f.write('{not json')
    auto.clear_calibration_cache()
    assert auto.load_calibration() is None


def test_real_calibration_roundtrip_and_selection():
    """Time every available backend on a tiny grid, persist it, and check auto() follows it."""
    grid = dict(nobs=(20, 200), nvar=(20, 100), nvec=(2, 4))
    tab = auto.calibrate(grid=grid, repeats=1, serial_max_work=1e9)
    assert len(tab['records']) == 8
    for r in tab['records']:
        assert set(r['cost']) == set(auto.available_backends())
        assert all(f >= 0 and p >= 0 for f, p in r['cost'].values())

    auto.clear_calibration_cache()
    loaded = auto.load_calibration()
    assert loaded is not None and loaded['records'] == tab['records']

    for r in tab['records']:      # at each measured point, the choice is that point's argmin
        niter = 25
        pred = {b: f + p * niter for b, (f, p) in r['cost'].items()}
        assert auto.choose_backend(r['nobs'], r['nvar'], r['nvec'], niter=niter) == min(pred, key=pred.get)

    data, weights = make_problem(nobs=200, nvar=100, nvec=4)
    m = auto.empca(data, weights, nvec=4, niter=5, silent=True)
    ref = serial.empca(data, weights, nvec=4, niter=5, silent=True)
    assert m.backend == auto.choose_backend(200, 100, 4, niter=5)
    assert_models_equal(ref, m)


# --------------------------------------------------------------------------- end to end
@pytest.mark.parametrize('backend', ALL)
def test_empca_dispatches_to_requested_backend(backend, monkeypatch):
    if backend not in auto.available_backends():
        pytest.skip('%s not available here' % backend)
    calls = []
    real = auto._run
    monkeypatch.setattr(auto, '_run', lambda b, *a, **k: (calls.append(b), real(b, *a, **k))[1])
    data, weights = make_problem(nobs=30, nvar=40)
    m = auto.empca(data, weights, nvec=2, niter=3, silent=True, backend=backend)
    assert calls == [backend] and m.backend == backend
    assert_models_equal(serial.empca(data, weights, nvec=2, niter=3, silent=True), m)


def test_empca_auto_follows_injected_table(monkeypatch):
    """The backend actually executed is the one the table says is fastest for the problem size."""
    avail = auto.available_backends()
    data, weights = make_problem(nobs=30, nvar=40)
    for target in avail:
        costs = {b: (0.0, 1.0) for b in avail}
        costs[target] = (0.0, 0.001)
        t = dict(fingerprint={}, key=auto._fingerprint_key(),
                 records=[dict(nobs=30, nvar=40, nvec=2, cost={k: list(v) for k, v in costs.items()})])
        with open(auto.calibration_path(), 'w') as f:
            json.dump({t['key']: t}, f)
        auto.clear_calibration_cache()
        m = auto.empca(data, weights, nvec=2, niter=3, silent=True)
        assert m.backend == target
        assert_models_equal(serial.empca(data, weights, nvec=2, niter=3, silent=True), m)


def test_env_var_forces_backend(monkeypatch):
    monkeypatch.setenv('EMPCA_BACKEND', 'numpy')
    data, weights = make_problem(nobs=30, nvar=40)
    assert auto.empca(data, weights, nvec=2, niter=2, silent=True).backend == 'numpy'


def test_default_weights_and_signature():
    data, _ = make_problem(nobs=30, nvar=40)
    m = auto.empca(data, nvec=2, niter=2, silent=True)
    assert m.weights.shape == data.shape
    # positional use, as with the original API: empca(data, weights, niter, nvec, smooth, randseed, silent)
    m2 = auto.empca(data, None, 2, 2, 0, 1, True)
    np.testing.assert_allclose(m.eigvec, m2.eigvec)
