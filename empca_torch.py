"""
Vectorized EMPCA in PyTorch (CPU or CUDA), float64.

Same interface as ``empca.py`` / ``empca_numpy.py``: ``Model`` and ``empca()``
take and return numpy arrays (``model.eigvec``, ``model.coeff``, ``model.model``
... are numpy), so results can be used interchangeably.  The device is chosen
with the ``device`` keyword ('cpu', 'cuda', a ``torch.device``, or ``None`` for
CUDA if available else CPU).

Arithmetic is float64 throughout so results agree with the serial and numpy
implementations to round-off.  The data and weights are uploaded once and the
whole EM loop runs on the device; only the small [nvec, nvar] eigenvector
block is moved to the host each iteration (Gram-Schmidt and optional smoothing
run there, with the same code as the numpy implementation).
"""
from __future__ import division, print_function

import sys

import numpy as np
import torch

import empca as _serial
import empca_numpy as _np_impl
from empca_numpy import SavitzkyGolay, classic_pca, lower_rank, random_orthonormal, preprocess, fixed_basis  # noqa: F401

_EPS = float(np.finfo(np.float64).eps)
_BLOCK_ELEMENTS = 1 << 24


def cuda_available():
    return torch.cuda.is_available()


def resolve_device(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _to_dev(a, device):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), device=device)


def _blocks(n, ncols):
    step = max(1, _BLOCK_ELEMENTS // max(1, ncols))
    for s in range(0, n, step):
        yield s, min(s + step, n)


class _State(object):
    """Device-resident copies of data/weights plus the row-uniformity split."""

    def __init__(self, data, weights, device):
        self.device = device
        self.D = _to_dev(data, device)
        self.W = _to_dev(weights, device)
        uniform = _np_impl._uniform_rows(np.asarray(weights))
        self.uniform_idx = torch.as_tensor(np.flatnonzero(uniform), device=device)
        self.nonuniform_idx = torch.as_tensor(np.flatnonzero(~uniform), device=device)
        self.all_nonuniform = self.uniform_idx.numel() == 0


@torch.no_grad()
def _solve_normal(M, b):
    """
    Solve M[i] x[i] = b[i] (symmetric PSD).  Well-conditioned rows use a batched
    Cholesky solve; the rest use the minimum-norm pseudo-inverse (lstsq semantics).
    Same decision rule as empca_numpy._solve_normal.  The pseudo-inverse is
    avoided on the fast path because batched eigh/svd on the GPU is very slow
    for matrices wider than 32.
    """
    L, info = torch.linalg.cholesky_ex(M)
    d = torch.diagonal(L, dim1=-2, dim2=-1)
    good = (info == 0) & (d.amin(-1) > _np_impl._PIVOT_RATIO_MIN * d.amax(-1))
    x = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    bad = torch.nonzero(~good).squeeze(-1)
    if bad.numel():
        Mb = M[bad]
        Minv = torch.linalg.pinv(Mb, rtol=_EPS * M.shape[-1], hermitian=True)
        x[bad] = (Minv @ b[bad].unsqueeze(-1)).squeeze(-1)
    return x


@torch.no_grad()
def _coeffs(st, V):
    """c[nobs, nvec] for eigenvectors V [nvec, nvar] (device tensor)."""
    nvec, nvar = V.shape
    nobs = st.D.shape[0]
    C = torch.empty(nobs, nvec, dtype=torch.float64, device=st.device)

    if st.uniform_idx.numel():
        ui = st.uniform_idx
        C[ui] = st.D[ui] @ V.T

    P = (V[:, None, :] * V[None, :, :]).reshape(nvec * nvec, nvar)
    idx = st.nonuniform_idx
    for s, e in _blocks(idx.numel(), nvar):
        rows = idx[s:e]
        if st.all_nonuniform:                      # avoid a gather copy
            w, d = st.W[s:e], st.D[s:e]
        else:
            w, d = st.W[rows], st.D[rows]
        M = (w @ P.T).reshape(-1, nvec, nvec)
        b = (w * d) @ V.T
        C[rows] = _solve_normal(M, b)
    return C


@torch.no_grad()
def _eigvecs(st, C, V, smooth=None, nfixed=0):
    """Sequential eigenvector update; returns new (un-orthonormalised) V clone."""
    V = V.clone()
    X = st.D.clone()
    if nfixed:
        X -= C[:, :nfixed] @ V[:nfixed]
    for k in range(nfixed, V.shape[0]):
        c = C[:, k]
        cw = c[:, None] * st.W
        num = (X * cw).sum(0)
        den = c @ cw
        V[k] = torch.where(den > 0, num / den, torch.zeros_like(num))   # unconstrained pixel -> 0 (serial: NaN)
        if smooth is not None:
            v = smooth(V[k].cpu().numpy())
            V[k] = torch.as_tensor(np.ascontiguousarray(v, dtype=np.float64), device=V.device)
        X -= c[:, None] * V[k][None, :]
    return V


def _orthonormalize_dev(V, start=0):
    """Gram-Schmidt on the host (same code as numpy impl); returns device tensor."""
    host = V.cpu().numpy().copy()
    _np_impl._orthonormalize(host, start)
    return torch.as_tensor(host, device=V.device)


class Model(_serial.Model):
    """
    Same interface as ``empca.Model``; computation runs in torch on ``device``.
    All public attributes (eigvec, data, weights, coeff, model) are numpy arrays.
    """

    def __init__(self, eigvec, data, weights, device=None, nfixed=0):
        self.nfixed = nfixed
        self.device = resolve_device(device)
        super(Model, self).__init__(eigvec, data, weights)

    def set_data(self, data, weights):
        self._st = _State(data, weights, self.device)
        super(Model, self).set_data(data, weights)
        self.dof += self.nfixed * self.nvar          # fixed vectors are not fitted parameters

    def solve_coeffs(self):
        V = _to_dev(self.eigvec, self.device)
        C = _coeffs(self._st, V)
        self.coeff[...] = C.cpu().numpy()
        self.solve_model()

    def solve_eigenvectors(self, smooth=None):
        V = _to_dev(self.eigvec, self.device)
        C = _to_dev(self.coeff, self.device)
        V = _eigvecs(self._st, C, V, smooth, self.nfixed)
        V = _orthonormalize_dev(V, self.nfixed)
        self.eigvec[...] = V.cpu().numpy()
        self.solve_model()

    def solve_model(self):
        V = _to_dev(self.eigvec, self.device)
        C = _to_dev(self.coeff, self.device)
        self.model[...] = (C @ V).cpu().numpy()


def empca(data, weights=None, niter=25, nvec=5, smooth=0, randseed=1, silent=False,
          device=None, subtract_mean=False, subtract_mean_deriv=False, deriv_grid=None,
          mean_as_vector=True, include_fixed_in_nvec=True):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights (torch).

    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]

    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize
      - silent   : suppress per-iteration output
      - device   : 'cpu', 'cuda', torch.device, or None (cuda if available)
      - subtract_mean, subtract_mean_deriv, deriv_grid, mean_as_vector,
        include_fixed_in_nvec : mean/derivative handling, as in empca_numpy.empca

    Returns Model object (numpy attributes)
    """
    device = resolve_device(device)
    data = np.asarray(data)
    if weights is None:
        weights = np.ones(data.shape)

    smooth = SavitzkyGolay(width=smooth) if smooth > 0 else None

    setup = _np_impl._Setup(data, weights, nvec, subtract_mean, subtract_mean_deriv, deriv_grid,
                            mean_as_vector, include_fixed_in_nvec)
    data = setup.data
    nfixed, ntotal = setup.nfixed, setup.total

    nobs, nvar = data.shape
    assert data.shape == weights.shape

    eigvec = setup.initial_eigvec(nvar, randseed)

    st = _State(data, weights, device)
    V = _to_dev(eigvec, device)

    if not silent:
        print("       iter        R2             rchi2")
        mask = st.W > 0
        dof = int(mask.sum()) - (ntotal - nfixed) * nvar - ntotal * nobs
        sel_data = st.D[mask]
        data_var = torch.var(sel_data, unbiased=False)
        sqrtw = torch.sqrt(st.W)

    for k in range(niter):
        C = _coeffs(st, V)
        V = _orthonormalize_dev(_eigvecs(st, C, V, smooth, nfixed), nfixed)
        if not silent:
            # as in the serial code: model from this iteration's coeffs and new eigvec
            mdl = C @ V
            R2 = 1.0 - torch.var((mdl - st.D)[mask], unbiased=False) / data_var
            chi2 = torch.sum(((mdl - st.D) * sqrtw) ** 2)
            print('EMPCA %2d/%2d  %15.8f %15.8f' %
                  (k + 1, niter, float(R2), float(chi2) / dof))
            sys.stdout.flush()

    # Final coefficients/model for the latest eigenvectors (done by Model.set_data)
    model = Model(V.cpu().numpy().copy(), data, weights, device=device, nfixed=nfixed)

    if not silent:
        print("R2:", model.R2())

    return setup.finish(model)
