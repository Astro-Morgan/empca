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


def _upload64(a, device):
    """Upload a 2-D array and return a C-contiguous float64 device tensor.  The layout fix-up and the dtype conversion
    run on the device: np.ascontiguousarray on a Fortran-ordered or strided host array (e.g. the result of ``F[:, keep]``)
    is a ~1 s host copy per 700 MB, while the same transpose on the GPU takes milliseconds."""
    a = np.asarray(a)
    if a.ndim == 2 and not a.flags.c_contiguous and a.flags.f_contiguous:
        return torch.as_tensor(a.T, device=device).T.contiguous().to(torch.float64)
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(torch.float64)


def _blocks(n, ncols):
    step = max(1, _BLOCK_ELEMENTS // max(1, ncols))
    for s in range(0, n, step):
        yield s, min(s + step, n)


class _State(object):
    """Device-resident copies of data/weights plus the row-uniformity split.

    Speed-ups vs the original: the arrays are uploaded in their native dtype and converted to float64 on the
    device (half the PCIe traffic for float32 input), the 'uniform row' test runs on the device (the host test
    was a full N x L numpy pass), and W*D is formed once (it is constant) instead of in every iteration.
    """

    def __init__(self, data, weights, device):
        self.device = device
        self.D = _upload64(data, device)
        self.W = _upload64(weights, device)
        self.WD = self.W * self.D
        uniform = ~torch.any(self.W != self.W[:, :1], dim=1)
        self.uniform_idx = torch.nonzero(uniform).squeeze(1)
        self.nonuniform_idx = torch.nonzero(~uniform).squeeze(1)
        self.all_nonuniform = self.uniform_idx.numel() == 0
        self.data_var = None                       # lazily: variance of the unmasked data


@torch.no_grad()
def _masked_var(st, X=None):
    """Population variance of X[W>0] (X defaults to the data) without a boolean gather: two blockwise passes."""
    X = st.D if X is None else X
    n = 0
    s = torch.zeros((), dtype=torch.float64, device=st.device)
    blocks = list(_blocks(X.shape[0], X.shape[1]))
    for a, b in blocks:
        m = st.W[a:b] > 0
        n += int(m.sum())
        s += torch.where(m, X[a:b], 0.0).sum()
    mean = s / max(n, 1)
    ss = torch.zeros((), dtype=torch.float64, device=st.device)
    for a, b in blocks:
        m = st.W[a:b] > 0
        ss += (torch.where(m, X[a:b] - mean, 0.0) ** 2).sum()
    return float(ss / max(n, 1))


@torch.no_grad()
def _fit_stats(st, C, V):
    """(R2, chi2) of the model C @ V, computed blockwise on the device (no N x L host or gather passes)."""
    if st.data_var is None:
        st.data_var = _masked_var(st)
    n = 0
    s = torch.zeros((), dtype=torch.float64, device=st.device)
    chi2 = torch.zeros((), dtype=torch.float64, device=st.device)
    blocks = list(_blocks(st.D.shape[0], st.D.shape[1]))
    for a, b in blocks:
        d = C[a:b] @ V - st.D[a:b]
        w = st.W[a:b]
        m = w > 0
        n += int(m.sum())
        s += torch.where(m, d, 0.0).sum()
        chi2 += (w * d * d).sum()
    mean = s / max(n, 1)
    ss = torch.zeros((), dtype=torch.float64, device=st.device)
    for a, b in blocks:
        d = C[a:b] @ V - st.D[a:b]
        m = st.W[a:b] > 0
        ss += (torch.where(m, d - mean, 0.0) ** 2).sum()
    var_d = float(ss) / max(n, 1)
    return 1.0 - var_d / st.data_var, float(chi2)


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
    n = idx.numel()
    if n:
        # the two skinny GEMMs run block-wise over the constant matrices (bounded temporaries when rows must be
        # gathered); the tiny [n, nvec, nvec] normal equations are then solved ONCE (one Cholesky, one sync) instead of
        # once per block
        M = torch.empty(n, nvec * nvec, dtype=torch.float64, device=st.device)
        b = torch.empty(n, nvec, dtype=torch.float64, device=st.device)
        for s, e in _blocks(n, nvar):
            if st.all_nonuniform:                      # avoid a gather copy
                w, wd = st.W[s:e], st.WD[s:e]
            else:
                rows = idx[s:e]
                w, wd = st.W[rows], st.WD[rows]
            M[s:e] = w @ P.T
            b[s:e] = wd @ V.T
        C[idx] = _solve_normal(M.reshape(n, nvec, nvec), b)
    return C


@torch.no_grad()
def _eigvecs(st, C, V, smooth=None, nfixed=0):
    """Sequential eigenvector update; returns new (un-orthonormalised) V clone.

    Same Gauss-Seidel update as before, V_k = sum_i c_ik w_ij (d_ij - sum_{l<k} c_il V_lj) / sum_i c_ik^2 w_ij, but the
    sums over observations are all done up front with two GEMMs over the constant matrices (the numerators need
    G1[j,k] = sum_i c_ik w_ij d_ij and the pair sums G2[j,k,l] = sum_i c_ik c_il w_ij); the sequential part is then
    a handful of length-nvar vector operations.  The original cloned D and made ~6 full N x L passes per vector.
    """
    nobs, nvec = C.shape
    nvar = V.shape[1]
    V = V.clone()
    G1 = st.WD.T @ C                                                    # [nvar, nvec]
    Q = (C[:, :, None] * C[:, None, :]).reshape(nobs, nvec * nvec)       # [nobs, nvec^2]
    G2 = (st.W.T @ Q).reshape(nvar, nvec, nvec)                          # [nvar, nvec, nvec]
    for k in range(nfixed, nvec):
        num = G1[:, k].clone()
        for l in range(k):
            num -= V[l] * G2[:, k, l]
        den = G2[:, k, k]
        V[k] = torch.where(den > 0, num / den, torch.zeros_like(num))   # unconstrained pixel -> 0 (serial: NaN)
        if smooth is not None:
            v = smooth(V[k].cpu().numpy())
            V[k] = torch.as_tensor(np.ascontiguousarray(v, dtype=np.float64), device=V.device)
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

    Speed-ups vs the original: ``state=`` lets empca() hand over its device-resident data instead of uploading
    everything a second time, the degrees of freedom are counted on the device, and the bookkeeping that needs a
    full host pass (``model``, ``_unmasked``, ``_unmasked_data_var``) is computed lazily on first use, so building the
    returned Model no longer costs several seconds at large N.  All values are unchanged.
    """

    def __init__(self, eigvec, data, weights, device=None, nfixed=0, state=None):
        self.nfixed = nfixed
        self.device = resolve_device(device)
        self._given_state = state
        self.eigvec = eigvec
        self.nvec = eigvec.shape[0]
        self.set_data(data, weights)

    # -- lazily built host quantities --------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            self._model = (self._C_dev @ _to_dev(self.eigvec, self.device)).cpu().numpy()
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def _unmasked(self):
        if self.__dict__.get('_unmasked_cache') is None:
            self._unmasked_cache = np.where(self.weights > 0)
        return self._unmasked_cache

    @property
    def _unmasked_data_var(self):
        if self.__dict__.get('_uvar') is None:
            if self._st.data_var is None:
                self._st.data_var = _masked_var(self._st)
            self._uvar = self._st.data_var
        return self._uvar

    def _stats(self):
        return _fit_stats(self._st, self._C_dev, _to_dev(self.eigvec, self.device))

    def chi2(self):
        """Returns sum( (model-data)^2 * weights ), computed on the device."""
        return self._stats()[1]

    def R2(self, nvec=None):
        """Fraction of unmasked data variance explained (device-side for the full model)."""
        if nvec is not None:
            return super(Model, self).R2(nvec)
        return self._stats()[0]

    def set_data(self, data, weights):
        self._st = self._given_state if self._given_state is not None else _State(data, weights, self.device)
        self._given_state = None
        self.data = data
        self.weights = weights
        self.nobs, self.nvar = data.shape
        self.coeff = np.zeros((self.nobs, self.nvec))
        self._model = None
        self._unmasked_cache = None
        self._uvar = None
        self.dof = int(torch.count_nonzero(self._st.W > 0)) - self.eigvec.size - self.nvec * self.nobs
        self.dof += self.nfixed * self.nvar          # fixed vectors are not fitted parameters
        self.solve_coeffs()

    def solve_coeffs(self):
        V = _to_dev(self.eigvec, self.device)
        self._C_dev = _coeffs(self._st, V)
        self.coeff[...] = self._C_dev.cpu().numpy()
        self._model = None                            # rebuilt on demand

    def solve_eigenvectors(self, smooth=None):
        V = _to_dev(self.eigvec, self.device)
        C = _to_dev(self.coeff, self.device)
        V = _eigvecs(self._st, C, V, smooth, self.nfixed)
        V = _orthonormalize_dev(V, self.nfixed)
        self.eigvec[...] = V.cpu().numpy()
        self._C_dev = C
        self._model = None

    def solve_model(self):
        self._C_dev = _to_dev(self.coeff, self.device)
        self._model = None


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
        dof = int(torch.count_nonzero(st.W > 0)) - (ntotal - nfixed) * nvar - ntotal * nobs

    for k in range(niter):
        C = _coeffs(st, V)
        V = _orthonormalize_dev(_eigvecs(st, C, V, smooth, nfixed), nfixed)
        if not silent:
            # as in the serial code: model from this iteration's coeffs and new eigvec
            R2, chi2 = _fit_stats(st, C, V)
            print('EMPCA %2d/%2d  %15.8f %15.8f' %
                  (k + 1, niter, R2, chi2 / dof))
            sys.stdout.flush()

    # Final coefficients/model for the latest eigenvectors (done by Model.set_data)
    model = Model(V.cpu().numpy().copy(), data, weights, device=device, nfixed=nfixed, state=st)

    if not silent:
        print("R2:", model.R2())

    return setup.finish(model)
