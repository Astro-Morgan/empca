"""
Check the automatic backend choice against measured end-to-end times at problem
sizes that are NOT on the calibration grid.

    python benchmarks/validate_auto.py            # uses the saved calibration
    python benchmarks/validate_auto.py --niter 25

For each shape every available backend is timed on a full empca() call (including
uploads/downloads); a backend that needs more than --skip seconds for a single
iteration is not timed further.  Reports the picked backend, the best one, and
how much slower the pick was (1.00 = optimal).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import empca_auto as auto  # noqa: E402

SHAPES = [  # (nobs, nvar, nvec): off-grid, mostly large
    (2000, 1500, 4), (5000, 3000, 7), (7000, 1200, 15), (15000, 1500, 6),
    (20000, 3000, 12), (4000, 6000, 8), (2500, 3000, 30), (12000, 2500, 25),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--niter', type=int, default=10)
    ap.add_argument('--skip', type=float, default=10.0)
    args = ap.parse_args(argv)
    avail = auto.available_backends()
    if 'torch-cuda' in avail:
        auto._run('torch-cuda', *auto._synthetic(50, 20, 2), 1, 2, 0, 1, True)
    print('calibration:', 'found' if auto.load_calibration() else 'NONE (built-in heuristic)')
    worst = 1.0
    for nobs, nvar, nvec in SHAPES:
        data, w = auto._synthetic(nobs, nvar, nvec, seed=1)
        times = {}
        for b in avail:
            if b == 'serial':
                continue
            t1 = auto.time_backend(b, nobs, nvar, nvec, 1, 1, data, w)
            if t1 > args.skip:
                times[b] = t1 * args.niter            # extrapolated lower bound
                continue
            times[b] = auto.time_backend(b, nobs, nvar, nvec, args.niter, 1, data, w)
        pick = auto.choose_backend(nobs, nvar, nvec, niter=args.niter, candidates=tuple(times))
        best = min(times, key=times.get)
        ratio = times[pick] / times[best]
        worst = max(worst, ratio)
        print('%6d x %5d x %3d  pick=%-10s best=%-10s slowdown=%5.2f   %s' % (
            nobs, nvar, nvec, pick, best, ratio,
            '  '.join('%s=%.2fs' % (b, t) for b, t in times.items())))
    print('worst slowdown vs best: %.2fx' % worst)


if __name__ == '__main__':
    main()
