"""
Vectorized (pure numpy) EMPCA.

Drop-in replacement for the serial implementation in ``empca.py``: the same
``Model`` attributes/methods and the same ``empca()`` signature, but the
per-observation / per-pixel Python loops are replaced by batched linear algebra.

The mathematics (and therefore the result, to floating-point round-off) is
identical to the serial code:

  * coefficients: rows whose weights are all equal use the plain projection
    ``eigvec @ data[i]``; every other row solves the weighted normal equations
    ``(V W V^T) c = V W d`` with the same minimum-norm (pseudo-inverse) solution
    that ``np.linalg.lstsq`` gives in the serial code;
  * eigenvectors: updated one at a time on the residual left by the previous
    ones, then Gram-Schmidt re-orthonormalised, exactly as before.

Differences from the serial output are limited to summation-order round-off
(~1e-15 relative per operation, float64).
"""
from __future__ import division, print_function

import sys

import numpy as np

import empca as _serial
# Re-exported unchanged so this module can be used as a drop-in replacement.
from empca import SavitzkyGolay, classic_pca, lower_rank  # noqa: F401

_EPS = np.finfo(np.float64).eps

def random_orthonormal(nvec, nvar, seed=1, tol=1e-8, max_tries=100, fixed=None):
    """
    Random orthonormal start vectors A[nvec, nvar].

    Identical to ``empca._random_orthonormal`` (same RNG stream, same operations)
    whenever that function works, but protects against its rare failure: a vector
    that is (numerically) a combination of the earlier ones is left with ~zero norm
    after Gram-Schmidt and the serial code then divides by zero (NaN eigenvectors).
    Here such a vector is redrawn from the same RNG stream.

    ``fixed`` [nfixed, nvar] (orthonormal rows) makes the new vectors orthogonal to
    those as well (used when leading basis vectors, e.g. the mean, are held fixed).
    """
    nfixed = 0 if fixed is None else len(fixed)
    if nvec + nfixed > nvar:
        raise ValueError("cannot make %d orthonormal vectors in %d dimensions" % (nvec + nfixed, nvar))
    if seed is not None:
        np.random.seed(seed)
    if nfixed:
        return _random_orthonormal_given(nvec, nvar, np.asarray(fixed), tol, max_tries)

    def degenerate(v):
        n = np.linalg.norm(v)
        return not (np.isfinite(n) and n > tol)

    A = np.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        for _ in range(max_tries):
            if not degenerate(A[i]):
                break
            A[i] = np.random.normal(size=nvar)
        else:
            raise RuntimeError("could not draw a non-degenerate random vector")
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, nvec):
        for _ in range(max_tries):
            ok = True
            for j in range(0, i):
                A[i] -= np.dot(A[j], A[i]) * A[j]
                if degenerate(A[i]):
                    ok = False
                    break
                A[i] /= np.linalg.norm(A[i])
            if ok:
                break
            v = np.random.normal(size=nvar)           # degenerate: redraw and redo the projections
            A[i] = v / np.linalg.norm(v)
        else:
            raise RuntimeError("could not draw a non-degenerate random vector")
    return A


def _random_orthonormal_given(nvec, nvar, fixed, tol, max_tries):
    """nvec random unit vectors orthogonal to each other and to the rows of ``fixed``."""
    A = np.random.normal(size=(nvec, nvar))
    out = np.empty((nvec, nvar))
    for i in range(nvec):
        v = A[i]
        for attempt in range(max_tries):
            if attempt:
                v = np.random.normal(size=nvar)
            n = np.linalg.norm(v)
            if not (np.isfinite(n) and n > 0):
                continue
            v = v / n
            for _ in range(2):                       # two passes: re-orthogonalise for stability
                for b in list(fixed) + list(out[:i]):
                    v = v - np.dot(b, v) * b
            n = np.linalg.norm(v)
            if np.isfinite(n) and n > tol:
                out[i] = v / n
                break
        else:
            raise RuntimeError("could not draw a non-degenerate random vector")
    return out


_random_orthonormal = random_orthonormal   # protected drop-in for the serial helper


# ---------------------------------------------------------------------------
# Optional preprocessing for template building
# ---------------------------------------------------------------------------
class Preprocessing(object):
    """
    Result of ``preprocess``.  For each object i the data are decomposed as

        original[i] = mean + deriv_coeff[i] * mean_deriv + residual[i]

    EMPCA is run on ``residual``.  Attributes (None when not requested):
      - mean        [nvar] weighted mean spectrum  sum_i w_ij d_ij / sum_i w_ij
      - mean_deriv  [nvar] derivative of the mean along the variable axis
      - deriv_coeff [nobs] weighted-least-squares amount of mean_deriv in each
                           (data - mean), i.e. the response to a small shift of
                           the mean template (such as a redshift error)
      - residual    [nobs, nvar]
    """
    def __init__(self, original, residual, mean=None, mean_deriv=None, deriv_coeff=None):
        self.original = original
        self.residual = residual
        self.mean = mean
        self.mean_deriv = mean_deriv
        self.deriv_coeff = deriv_coeff

    def baseline(self):
        """[nobs, nvar] part of the data removed before EMPCA (zeros if nothing removed)."""
        base = np.zeros(self.original.shape)
        if self.mean is not None:
            base += self.mean
        if self.deriv_coeff is not None:
            base += np.outer(self.deriv_coeff, self.mean_deriv)
        return base

    def attach(self, model):
        """Store the decomposition on a Model; ``model.full_model()`` rebuilds the original data."""
        model.preprocessing = self
        model.mean = self.mean
        model.mean_deriv = self.mean_deriv
        model.deriv_coeff = self.deriv_coeff
        model.original_data = self.original
        model.full_model = lambda: model.model + self.baseline()
        return model


def _weighted_mean(data, weights):
    sw = weights.sum(axis=0)
    return np.divide(np.einsum('ij,ij->j', weights, data), sw, out=np.zeros(data.shape[1]), where=sw > 0)


def fixed_basis(data, weights=None, subtract_mean_deriv=False, deriv_grid=None):
    """
    Leading basis vectors held fixed while EMPCA learns the rest.

    Returns (F, mean, mean_deriv): F[nfixed, nvar] orthonormal rows -- the
    normalised weighted mean, then (optionally) the derivative of the mean
    along the variable axis with its component along the mean removed and
    normalised; plus the raw mean and raw derivative (None if not requested).
    """
    data = np.asarray(data)
    if weights is None:
        weights = np.ones(data.shape)
    mean = _weighted_mean(data, weights)
    nm = np.linalg.norm(mean)
    if not nm > 0:
        raise ValueError("weighted mean is zero; cannot use it as a basis vector")
    rows, dmean = [mean / nm], None
    if subtract_mean_deriv:
        dmean = np.gradient(mean) if deriv_grid is None else np.gradient(mean, np.asarray(deriv_grid))
        d = dmean - np.dot(dmean, rows[0]) * rows[0]
        nd = np.linalg.norm(d)
        if not nd > 1e-12 * max(np.linalg.norm(dmean), 1e-300):
            raise ValueError("derivative of the mean is parallel to the mean; cannot use it as a basis vector")
        rows.append(d / nd)
    return np.array(rows), mean, dmean


class _Setup(object):
    """Resolves the mean/derivative options of empca() into data, fixed basis and vector counts."""

    def __init__(self, data, weights, nvec, subtract_mean, subtract_mean_deriv, deriv_grid,
                 mean_as_vector, include_fixed_in_nvec):
        if subtract_mean_deriv and not subtract_mean:
            raise ValueError("subtract_mean_deriv requires subtract_mean=True")
        self.data, self.pre, self.fixed = data, None, None
        self.mean = self.mean_deriv = None
        self.nfixed = 0
        self.nlearn = nvec
        if subtract_mean and not mean_as_vector:
            self.pre = preprocess(data, weights, subtract_mean, subtract_mean_deriv, deriv_grid)
            self.data = self.pre.residual
        elif subtract_mean:
            self.fixed, self.mean, self.mean_deriv = fixed_basis(data, weights, subtract_mean_deriv, deriv_grid)
            self.nfixed = len(self.fixed)
            self.nlearn = nvec - self.nfixed if include_fixed_in_nvec else nvec
            if self.nlearn < 0:
                raise ValueError("nvec=%d is smaller than the %d fixed vectors" % (nvec, self.nfixed))
        self.total = self.nfixed + self.nlearn

    def initial_eigvec(self, nvar, seed):
        if not self.nfixed:
            return random_orthonormal(self.nlearn, nvar, seed=seed)
        learned = random_orthonormal(self.nlearn, nvar, seed=seed, fixed=self.fixed)
        return np.vstack([self.fixed, learned])

    def finish(self, model):
        if self.pre is not None:
            self.pre.attach(model)
        elif self.nfixed:
            model.mean, model.mean_deriv = self.mean, self.mean_deriv
            model.fixed_vectors = self.fixed
            model.deriv_coeff = None
            model.original_data = model.data
            model.full_model = lambda: model.model.copy()
        return model


def preprocess(data, weights=None, subtract_mean=False, subtract_mean_deriv=False, deriv_grid=None):
    """
    Remove the weighted mean and/or the (per-object fitted) derivative of the
    mean from data[nobs, nvar] before EMPCA.  ``deriv_grid`` optionally gives the
    coordinates of the variable axis (e.g. log wavelength) for the derivative
    (default: unit pixel spacing).  Pixels with zero total weight get mean 0.
    """
    data = np.asarray(data)
    if weights is None:
        weights = np.ones(data.shape)
    if subtract_mean_deriv and not subtract_mean:
        raise ValueError("subtract_mean_deriv requires subtract_mean=True")
    resid, mean, dmean, coef = data, None, None, None
    if subtract_mean:
        mean = _weighted_mean(data, weights)
        resid = data - mean
    if subtract_mean_deriv:
        dmean = np.gradient(mean) if deriv_grid is None else np.gradient(mean, np.asarray(deriv_grid))
        den = weights @ dmean ** 2
        coef = np.divide(np.einsum('ij,ij,j->i', weights, resid, dmean), den,
                         out=np.zeros(data.shape[0]), where=den > 0)
        resid = resid - np.outer(coef, dmean)
    return Preprocessing(data, resid, mean, dmean, coef)


# Rows processed per block when building the normal equations; bounds the size
# of the [block, nvar] temporaries.
_BLOCK_ELEMENTS = 1 << 18


_EIG_BLOCK_ELEMENTS = 1 << 17


def _row_blocks(nrows, ncols, elements=_BLOCK_ELEMENTS):
    step = max(1, elements // max(1, ncols))
    for s in range(0, nrows, step):
        yield slice(s, min(s + step, nrows))


# Rows whose normal matrix is well conditioned (Cholesky pivot ratio above this,
# i.e. cond(M) <~ 1e4) are solved directly; the rest use the minimum-norm
# pseudo-inverse, which is what np.linalg.lstsq does in the serial code.  The
# direct solve differs from lstsq by ~cond*eps (~1e-12) and avoids the very
# slow batched eigendecomposition for large nvec.
_PIVOT_RATIO_MIN = 1e-2


def _pinv_solve(M, b):
    nvec = M.shape[-1]
    # Same cutoff as np.linalg.lstsq (eps * max(m, n))
    Minv = np.linalg.pinv(M, rcond=_EPS * nvec, hermitian=True)
    return np.matmul(Minv, b[..., None])[..., 0]


def _solve_normal(M, b):
    """Solve M[i] x[i] = b[i] for symmetric PSD M [n,K,K], b [n,K]; lstsq semantics."""
    out = np.zeros(b.shape)
    empty = ~np.any(M != 0, axis=(1, 2))          # fully masked rows: lstsq gives 0
    idx = np.flatnonzero(~empty)
    if idx.size:
        out[idx] = _solve_rows(M[idx], b[idx])
    return out


def _solve_rows(M, b):
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:                  # some matrix is not numerically PD
        if M.shape[0] == 1:
            return _pinv_solve(M, b)
        h = M.shape[0] // 2
        return np.concatenate([_solve_rows(M[:h], b[:h]), _solve_rows(M[h:], b[h:])])
    d = np.diagonal(L, axis1=1, axis2=2)
    good = d.min(axis=1) > _PIVOT_RATIO_MIN * d.max(axis=1)
    if good.all():
        return np.linalg.solve(M, b[..., None])[..., 0]
    x = np.empty(b.shape)
    x[good] = np.linalg.solve(M[good], b[good][..., None])[..., 0] if good.any() else x[good]
    x[~good] = _pinv_solve(M[~good], b[~good])
    return x


def _uniform_rows(weights):
    """Boolean [nobs]: True where all weights of the row equal its first weight."""
    return ~np.any(weights != weights[:, :1], axis=1)


def _wls_coeffs(eigvec, data, weights):
    """
    Batched weighted least squares.  Solve data[i] ~= sum_k c[i,k] eigvec[k]
    for every row i with weights[i].  Returns c [n, nvec].
    """
    nvec, nvar = eigvec.shape
    # P[k*nvec+l, j] = V[k,j] V[l,j]  ->  normal matrix M[i,k,l] = sum_j V[k,j] w[i,j] V[l,j]
    P = (eigvec[:, None, :] * eigvec[None, :, :]).reshape(nvec * nvec, nvar)
    out = np.empty((data.shape[0], nvec))
    blocks = list(_row_blocks(data.shape[0], nvar))
    wd = np.empty((blocks[0].stop - blocks[0].start, nvar)) if blocks else None
    for sl in blocks:
        w, d = weights[sl], data[sl]
        M = (w @ P.T).reshape(-1, nvec, nvec)
        wd_blk = wd[:sl.stop - sl.start]
        np.multiply(w, d, out=wd_blk)
        b = wd_blk @ eigvec.T
        out[sl] = _solve_normal(M, b)
    return out


def _orthonormalize(eigvec, start=0):
    """
    In-place Gram-Schmidt, identical operation order to the serial code.
    Rows before ``start`` are fixed (assumed orthonormal) and left untouched.
    """
    def norm(x):
        return np.sqrt(np.dot(x, x))

    if start == 0 and eigvec.shape[0]:
        eigvec[0] /= norm(eigvec[0])
        start = 1
    for k in range(start, eigvec.shape[0]):
        for kx in range(0, k):
            c = np.dot(eigvec[k], eigvec[kx])
            eigvec[k] -= c * eigvec[kx]
        eigvec[k] /= norm(eigvec[k])


class Model(_serial.Model):
    """
    Same interface as ``empca.Model`` with vectorized solvers.

      Inputs:  eigvec [nvec, nvar], data [nobs, nvar], weights [nobs, nvar]
      Derived: coeff  [nobs, nvec], model [nobs, nvar]

    ``nfixed`` leading rows of eigvec (e.g. the mean) are held fixed: they take
    part in every coefficient fit but are never updated, and the learned rows
    are kept orthogonal to them.

    Everything not overridden here (chi2, rchi2, R2, R2vec, set_data, ...) is
    inherited from the serial Model; set_data/solve_coeffs dispatch to the
    vectorized versions below.
    """

    def __init__(self, eigvec, data, weights, nfixed=0):
        self.nfixed = nfixed
        super(Model, self).__init__(eigvec, data, weights)

    def set_data(self, data, weights):
        super(Model, self).set_data(data, weights)
        self.dof += self.nfixed * self.nvar          # fixed vectors are not fitted parameters

    def solve_coeffs(self):
        """Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]"""
        uniform = _uniform_rows(self.weights)
        if uniform.all():
            self.coeff[...] = self.data @ self.eigvec.T
        else:
            if uniform.any():
                self.coeff[uniform] = self.data[uniform] @ self.eigvec.T
            nu = np.flatnonzero(~uniform)
            if nu.size == self.nobs:
                self.coeff[...] = _wls_coeffs(self.eigvec, self.data, self.weights)
            else:
                self.coeff[nu] = _wls_coeffs(self.eigvec, self.data[nu], self.weights[nu])
        self.solve_model()

    def solve_eigenvectors(self, smooth=None):
        """Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]"""
        D, W, C = self.data, self.weights, self.coeff
        blocks = list(_row_blocks(self.nobs, self.nvar, _EIG_BLOCK_ELEMENTS))
        step = blocks[0].stop - blocks[0].start
        xbuf = np.empty((step, self.nvar))       # reused scratch: avoids per-block allocation
        cwbuf = np.empty((step, self.nvar))
        for k in range(self.nfixed, self.nvec):
            # x = data with the (already updated) vectors 0..k-1 removed, built
            # block-by-block so the temporaries stay cache resident.
            num = np.zeros(self.nvar)
            den = np.zeros(self.nvar)
            for sl in blocks:
                n = sl.stop - sl.start
                c = C[sl, k]
                if k == 0:
                    x = D[sl]
                else:
                    x = xbuf[:n]
                    np.matmul(C[sl, :k], self.eigvec[:k], out=x)
                    np.subtract(D[sl], x, out=x)
                cw = cwbuf[:n]
                np.multiply(c[:, None], W[sl], out=cw)
                num += np.einsum('ij,ij->j', x, cw)    # sum_i x_ij c_i w_ij
                den += c @ cw                          # sum_i c_i^2 w_ij
            # A pixel with zero total weight is unconstrained (serial code: 0/0 = NaN
            # that then poisons every coefficient); give it a zero component.
            self.eigvec[k] = np.divide(num, den, out=np.zeros(self.nvar), where=den > 0)

            if smooth is not None:
                self.eigvec[k] = smooth(self.eigvec[k])

        _orthonormalize(self.eigvec, self.nfixed)
        self.solve_model()

    def solve_model(self):
        """Uses eigenvectors and coefficients to model data"""
        self.model[...] = self.coeff @ self.eigvec


def empca(data, weights=None, niter=25, nvec=5, smooth=0, randseed=1, silent=False,
          subtract_mean=False, subtract_mean_deriv=False, deriv_grid=None,
          mean_as_vector=True, include_fixed_in_nvec=True):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights (vectorized).

    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]

    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize
      - subtract_mean       : use the weighted mean spectrum (see below)
      - subtract_mean_deriv : also use the derivative of the mean (needs subtract_mean)
      - deriv_grid          : coordinates of the variable axis for that derivative
      - mean_as_vector      : True (default): the mean (and derivative) become fixed,
            leading basis vectors, normalised and orthogonal to each other, fitted
            jointly with the learned vectors with a free amplitude per object.
            ``model.eigvec[:model.nfixed]`` are those vectors, ``model.data`` is the
            raw data.  False: the mean is removed at amplitude 1 and each object's
            fitted amount of the mean's derivative is removed too; EMPCA is fitted
            to the residuals (``model.data``) and ``model.full_model()`` rebuilds
            the original data (see ``preprocess``).
      - include_fixed_in_nvec : (mean_as_vector only) True (default): ``nvec`` counts
            the fixed vectors, so eigvec has nvec rows and nvec-nfixed are learned.
            False: nvec vectors are learned in addition to the fixed ones.

    Returns Model object
    """
    if weights is None:
        weights = np.ones(data.shape)
    setup = _Setup(data, weights, nvec, subtract_mean, subtract_mean_deriv, deriv_grid,
                   mean_as_vector, include_fixed_in_nvec)
    data = setup.data

    smooth = SavitzkyGolay(width=smooth) if smooth > 0 else None

    nobs, nvar = data.shape
    assert data.shape == weights.shape

    eigvec = setup.initial_eigvec(nvar, randseed)
    model = Model(eigvec, data, weights, nfixed=setup.nfixed)

    if not silent:
        print("       iter        R2             rchi2")

    for k in range(niter):
        model.solve_coeffs()
        model.solve_eigenvectors(smooth=smooth)
        if not silent:
            print('EMPCA %2d/%2d  %15.8f %15.8f' %
                  (k + 1, niter, model.R2(), model.rchi2()))
            sys.stdout.flush()

    # One last time with latest coefficients
    model.solve_coeffs()

    if not silent:
        print("R2:", model.R2())

    return setup.finish(model)
