"""
Context-aware EMPCA: pick the fastest implementation for this machine and problem.

    import empca_auto
    m = empca_auto.empca(data, weights, nvec=5)            # backend='auto'
    m = empca_auto.empca(data, weights, backend='numpy')   # force one
    with empca_auto.use_backend('torch-cuda'):             # force within a block
        m = empca_auto.empca(data, weights)

Backends (all return a Model with identical numpy attributes, equal to
round-off):

    'serial'      the original loop implementation in empca.py
    'numpy'       vectorized numpy (empca_numpy.py)
    'torch-cpu'   vectorized torch on CPU (empca_torch.py)
    'torch-cuda'  vectorized torch on a CUDA GPU (empca_torch.py)

Selection priority:  backend= argument  >  use_backend() context  >
EMPCA_BACKEND environment variable  >  automatic.

Automatic selection uses, in order:
  1. a calibration table measured on *this* machine (``calibrate()``; stored as
     JSON, see ``calibration_path()``): the nearest measured (nobs, nvar, nvec)
     in log-space decides, using predicted time = fixed + per_iter * niter;
  2. otherwise built-in size thresholds (``_heuristic``).
Only backends that are actually available (torch installed, CUDA present) are
considered.
"""
from __future__ import division, print_function

import contextlib
import contextvars
import hashlib
import json
import os
import platform
import time

import numpy as np

BACKENDS = ('serial', 'numpy', 'torch-cpu', 'torch-cuda')

# ---- built-in fallback thresholds (used only when there is no calibration) ----
# work = nobs * nvar * nvec : roughly proportional to the cost of one EM iteration.
# Measured on an i-class laptop CPU + RTX 4090 laptop GPU (see benchmarks/).
HEURISTIC_SERIAL_MAX_WORK = 0          # serial is never faster than vectorized numpy
HEURISTIC_CUDA_MIN_WORK = 1.0e6        # below this GPU launch/transfer overhead dominates
HEURISTIC_TORCH_CPU_MIN_WORK = 1.0e6   # multithreaded torch beats single-threaded numpy above this

_backend_override = contextvars.ContextVar('empca_backend', default=None)
_env_cache = {}


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------
def _probe_environment():
    """(has_torch, has_cuda); cached.  Call refresh_environment() to re-probe."""
    if 'probe' not in _env_cache:
        has_torch = has_cuda = False
        try:
            import torch
            has_torch = True
            has_cuda = bool(torch.cuda.is_available())
        except Exception:               # ImportError, or broken CUDA/DLL setups
            pass
        _env_cache['probe'] = (has_torch, has_cuda)
    return _env_cache['probe']


def refresh_environment():
    _env_cache.clear()


def available_backends():
    has_torch, has_cuda = _probe_environment()
    out = ['serial', 'numpy']
    if has_torch:
        out.append('torch-cpu')
        if has_cuda:
            out.append('torch-cuda')
    return tuple(out)


def environment_fingerprint():
    """Dict identifying the hardware/software the timings are valid for."""
    info = {
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'numpy': np.__version__,
    }
    has_torch, has_cuda = _probe_environment()
    if has_torch:
        import torch
        info['torch'] = torch.__version__
        if has_cuda:
            info['gpu'] = torch.cuda.get_device_name(0)
    return info


def _fingerprint_key(info=None):
    info = environment_fingerprint() if info is None else info
    return hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# running a backend
# ----------------------------------------------------------------------------
def _run(backend, data, weights, niter, nvec, smooth, randseed, silent, **pre):
    if backend == 'serial':
        import empca as impl
        import empca_numpy
        p = None
        if pre.get('subtract_mean') or pre.get('subtract_mean_deriv'):
            p = empca_numpy.preprocess(data, weights, pre.get('subtract_mean', False),
                                       pre.get('subtract_mean_deriv', False), pre.get('deriv_grid'))
            data = p.residual
        m = impl.empca(data, weights, niter=niter, nvec=nvec, smooth=smooth,
                       randseed=randseed, silent=silent)
        return p.attach(m) if p is not None else m
    if backend == 'numpy':
        import empca_numpy as impl
        return impl.empca(data, weights, niter=niter, nvec=nvec, smooth=smooth,
                          randseed=randseed, silent=silent, **pre)
    if backend in ('torch-cpu', 'torch-cuda'):
        import empca_torch as impl
        return impl.empca(data, weights, niter=niter, nvec=nvec, smooth=smooth,
                          randseed=randseed, silent=silent, **pre,
                          device='cuda' if backend == 'torch-cuda' else 'cpu')
    raise ValueError("unknown backend %r; choose from %s" % (backend, BACKENDS + ('auto',)))


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
def calibration_path():
    p = os.environ.get('EMPCA_CALIBRATION_FILE')
    if p:
        return p
    return os.path.join(os.path.expanduser('~'), '.cache', 'empca', 'calibration.json')


def _synthetic(nobs, nvar, nvec, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 2 * np.pi, nvar)
    coeff = rng.normal(size=(nobs, nvec))
    basis = np.sin(np.outer(np.arange(1, nvec + 1), x))
    data = coeff @ basis + 0.1 * rng.normal(size=(nobs, nvar))
    weights = rng.uniform(0.5, 2.0, size=(nobs, nvar))
    weights[rng.uniform(size=(nobs, nvar)) < 0.1] = 0.0
    return data, weights


def time_backend(backend, nobs, nvar, nvec, niter=3, repeats=2, data=None, weights=None):
    """Best-of-`repeats` wall time (s) of a full empca() call, silent, incl. transfers."""
    if data is None:
        data, weights = _synthetic(nobs, nvar, nvec)
    best = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run(backend, data, weights, niter, nvec, 0, 1, True)
        best = min(best, time.perf_counter() - t0)
    return best


def _fit_cost(backend, nobs, nvar, nvec, repeats, n_lo=1, n_hi=6, slow_s=None, data=None, weights=None):
    """
    Return (fixed_seconds, per_iteration_seconds) from two niter measurements.
    If one iteration already takes longer than ``slow_s`` the second measurement
    is skipped and the whole time is attributed to the iteration (fixed = 0).
    """
    if data is None:
        data, weights = _synthetic(nobs, nvar, nvec)
    t_lo = time_backend(backend, nobs, nvar, nvec, n_lo, 1 if slow_s else repeats, data, weights)
    if slow_s and t_lo > slow_s:
        return 0.0, t_lo
    if slow_s and repeats > 1:
        t_lo = min(t_lo, time_backend(backend, nobs, nvar, nvec, n_lo, repeats - 1, data, weights))
    t_hi = time_backend(backend, nobs, nvar, nvec, n_hi, repeats, data, weights)
    per = max((t_hi - t_lo) / (n_hi - n_lo), 0.0)
    return max(t_lo - per * n_lo, 0.0), per


# Concentrated on sizes where the choice costs real time (seconds to minutes per fit).
DEFAULT_GRID = dict(nobs=(1000, 3000, 10000, 30000), nvar=(500, 1000, 2000, 4000, 8000),
                    nvec=(3, 5, 10, 20, 40))


def calibrate(grid=None, backends=None, repeats=2, serial_max_work=2.0e6, path=None,
              save=True, verbose=False, max_elements=1.6e8, slow_s=10.0):
    """
    Time every available backend over a grid of (nobs, nvar, nvec) on this machine.

    Returns the calibration table (dict) and, if `save`, writes it to `path`
    (default ``calibration_path()``), merging with tables for other machines.
    Serial is skipped where nobs*nvar*nvec > serial_max_work (it is hopelessly slow there).
    Grid points with nobs*nvar > max_elements (memory) are skipped; a backend whose single
    iteration takes longer than slow_s seconds is timed once only.
    """
    grid = DEFAULT_GRID if grid is None else grid
    backends = tuple(backends) if backends else available_backends()
    if 'torch-cuda' in backends:              # absorb one-off CUDA context creation
        _run('torch-cuda', *_synthetic(50, 20, 2), 1, 2, 0, 1, True)
    records = []
    for nobs in grid['nobs']:
        for nvar in grid['nvar']:
            for nvec in grid['nvec']:
                if nobs * nvar > max_elements:
                    continue
                rec = dict(nobs=nobs, nvar=nvar, nvec=nvec, cost={})
                data, weights = _synthetic(nobs, nvar, nvec)
                for b in backends:
                    if b == 'serial' and nobs * nvar * nvec > serial_max_work:
                        continue
                    fixed, per = _fit_cost(b, nobs, nvar, nvec, repeats, slow_s=slow_s, data=data, weights=weights)
                    rec['cost'][b] = [fixed, per]
                if verbose:
                    print(nobs, nvar, nvec, {b: round(sum(c), 4) for b, c in rec['cost'].items()})
                records.append(rec)
    info = environment_fingerprint()
    table = dict(fingerprint=info, key=_fingerprint_key(info), records=records)
    if save:
        save_calibration(table, path)
    return table


def save_calibration(table, path=None):
    path = path or calibration_path()
    store = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                store = json.load(f)
        except (OSError, ValueError):
            store = {}
    store[table['key']] = table
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(store, f, indent=1)
    clear_calibration_cache()
    return path


def load_calibration(path=None):
    """Calibration table for this machine, or None."""
    cache_key = ('calibration', path)
    if cache_key in _env_cache:
        return _env_cache[cache_key]
    table = None
    p = path or calibration_path()
    try:
        with open(p) as f:
            table = json.load(f).get(_fingerprint_key())
    except (OSError, ValueError):
        table = None
    _env_cache[cache_key] = table
    return table


def clear_calibration_cache():
    for k in [k for k in _env_cache if isinstance(k, tuple)]:
        del _env_cache[k]


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------
def _heuristic(nobs, nvar, nvec, candidates):
    work = float(nobs) * nvar * nvec
    if 'torch-cuda' in candidates and work >= HEURISTIC_CUDA_MIN_WORK:
        return 'torch-cuda'
    if 'torch-cpu' in candidates and work >= HEURISTIC_TORCH_CPU_MIN_WORK:
        return 'torch-cpu'
    if 'numpy' in candidates:
        return 'numpy'
    return candidates[0]


def _from_table(table, nobs, nvar, nvec, niter, candidates):
    """Nearest calibration record (log-space); lowest predicted time among candidates."""
    q = np.log2([max(nobs, 1), max(nvar, 1), max(nvec, 1)])
    best_rec, best_d = None, float('inf')
    for rec in table['records']:
        if not any(c in rec['cost'] for c in candidates):
            continue
        d = float(np.sum((np.log2([rec['nobs'], rec['nvar'], rec['nvec']]) - q) ** 2))
        if d < best_d:
            best_rec, best_d = rec, d
    if best_rec is None:
        return None
    pred = {b: f + p * niter for b, (f, p) in best_rec['cost'].items() if b in candidates}
    return min(pred, key=pred.get)


def choose_backend(nobs, nvar, nvec, niter=25, candidates=None, table=None):
    """
    Pick a backend for a problem of this size.  `candidates` limits the choice
    (default: all available); `table` overrides the loaded calibration (pass
    ``False`` to force the built-in heuristic).
    """
    avail = available_backends()
    candidates = tuple(c for c in (avail if candidates is None else candidates) if c in avail)
    if not candidates:
        raise RuntimeError("no usable EMPCA backend among the requested candidates")
    if table is None:
        table = load_calibration()
    if table:
        pick = _from_table(table, nobs, nvar, nvec, niter, candidates)
        if pick is not None:
            return pick
    return _heuristic(nobs, nvar, nvec, candidates)


@contextlib.contextmanager
def use_backend(name):
    """Force a backend (or 'auto') for empca() calls inside the ``with`` block."""
    _validate(name)
    token = _backend_override.set(name)
    try:
        yield
    finally:
        _backend_override.reset(token)


def _validate(name):
    if name != 'auto' and name not in BACKENDS:
        raise ValueError("unknown backend %r; choose from %s" % (name, BACKENDS + ('auto',)))


def resolve_backend(nobs, nvar, nvec, niter=25, backend=None, candidates=None):
    """The backend empca() would use, following the priority order in the module doc."""
    if backend is None:
        backend = _backend_override.get() or os.environ.get('EMPCA_BACKEND') or 'auto'
    _validate(backend)
    if backend == 'auto':
        return choose_backend(nobs, nvar, nvec, niter, candidates=candidates)
    if backend not in available_backends():
        raise RuntimeError("backend %r requested but not available here (available: %s)"
                           % (backend, ', '.join(available_backends())))
    return backend


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def empca(data, weights=None, niter=25, nvec=5, smooth=0, randseed=1, silent=False,
          backend=None, subtract_mean=False, subtract_mean_deriv=False, deriv_grid=None,
          mean_as_vector=True, include_fixed_in_nvec=True):
    """
    Same signature and result as ``empca.empca`` plus ``backend``:
    None/'auto' (default), 'serial', 'numpy', 'torch-cpu', 'torch-cuda'.
    The chosen backend name is stored in ``model.backend``.
    """
    data = np.asarray(data)
    if weights is None:
        weights = np.ones(data.shape)
    nobs, nvar = data.shape
    vector_mode = bool(subtract_mean and mean_as_vector)
    name = resolve_backend(nobs, nvar, nvec, niter, backend,
                           candidates=tuple(b for b in BACKENDS if b != 'serial') if vector_mode else None)
    if vector_mode and name == 'serial':
        raise ValueError("the serial implementation cannot hold fixed leading vectors; use another backend "
                         "or mean_as_vector=False")
    pre = {}
    if subtract_mean or subtract_mean_deriv:
        pre = dict(subtract_mean=subtract_mean, subtract_mean_deriv=subtract_mean_deriv, deriv_grid=deriv_grid)
        if vector_mode:
            pre.update(mean_as_vector=True, include_fixed_in_nvec=include_fixed_in_nvec)
        else:
            pre['mean_as_vector'] = False
    model = _run(name, data, weights, niter, nvec, smooth, randseed, silent, **pre)
    model.backend = name
    return model
