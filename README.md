### empca: Weighted Expectation Maximization Principal Component Analysis ###

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

That returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
If you want to apply the model to new data:

    m.set_data(new_data, new_weights)
    
and then it will recalculate m.coeff, m.model, m.rchi2, etc. for the new data.

m.R2() is the fraction of data variance explained by the model, while
m.R2vec(i) is the amount of variance explained by eigenvector i.

This implementation of EMPCA does *not* subtract the mean from the data.
If you don't subtract the mean yourself, it will still work, with
the first eigenvector likely being something similar to the mean.
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...

Everything is self contained in empca.py .  Just put that into your
PYTHONPATH and "pydoc empca" for more details.  For a quick test
on toy example data, run

    python empca.py

This requires numpy and scipy; it will make plots if you have pylab installed.

#### Fast implementations ####

`empca.py` is the original serial reference implementation. Three drop-in
vectorized versions with the same `Model`/`empca()` API give the same results
to floating-point round-off (float64):

- `empca_numpy.py` : vectorized numpy
- `empca_torch.py` : PyTorch on CPU or CUDA (`device='cpu'|'cuda'`), requires torch
- `empca_auto.py`  : picks the fastest backend for your machine and (nobs, nvar, nvec)

    import empca_auto
    m = empca_auto.empca(data, weights, nvec=5)             # automatic
    m = empca_auto.empca(data, weights, backend='numpy')    # or force one
    with empca_auto.use_backend('torch-cuda'): ...          # or force within a block

For template building, all vectorized backends accept `subtract_mean=True`.
By default (`mean_as_vector=True`) the weighted mean spectrum becomes a **fixed
first basis vector** (normalised): it is fitted jointly with the learned vectors
with a free amplitude per object, never updated, and the learned vectors are kept
orthogonal to it. `subtract_mean_deriv=True` adds a second fixed vector, d(mean)/d(pixel)
(optional `deriv_grid=`, e.g. log wavelength) with its component along the mean
removed, the signature of a small redshift error. `nvec` counts the fixed vectors
(`include_fixed_in_nvec=True`, default: nvec=5 gives mean + 4 learned); with
`include_fixed_in_nvec=False` nvec vectors are learned in addition. `m.nfixed`,
`m.mean`, `m.mean_deriv` describe them and `m.data` stays the raw data, so
`m.set_data(new_raw, new_w)` works directly. The serial implementation cannot do this.
`mean_as_vector=False` instead removes the mean at amplitude 1 plus each
object's fitted amount of d(mean), fits EMPCA to the residuals (`m.data`), and
`m.full_model()` rebuilds the originals (new data must then be preprocessed
by hand). Pixels with zero total weight are left at 0 in learned vectors
(the serial code gives NaN).
Start vectors come from `empca_numpy.random_orthonormal`, which matches the
serial helper but redraws degenerate vectors.

`EMPCA_BACKEND=numpy` in the environment also forces a backend. Run
`empca_auto.calibrate()` once to time all backends on your hardware (stored in
`~/.cache/empca/calibration.json`, override with `EMPCA_CALIBRATION_FILE`);
without a calibration, built-in size thresholds are used.
Tests: `pytest test/`.  Scaling benchmarks: `python benchmarks/bench_scaling.py`.

The paper S. Bailey 2012, PASP, 124, 1015 describes the underlying math
and is available as a pre-print at:

http://arxiv.org/abs/1208.4122

If you use this code in an academic paper, please include a citation
as described in CITATION.txt, and optionally an acknowledgement such as:

> This work uses the Weighted EMPCA code by Stephen Bailey,
> available at https://github.com/sbailey/empca/

The examples in the paper were prepared with version v0.2 of the code.

Stephen Bailey, Summer 2012

