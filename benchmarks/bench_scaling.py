"""
Scaling benchmark for every EMPCA backend.

Sweeps nobs, nvar and nvec one at a time (others held at the base values) and
prints seconds per EM iteration for each available backend, plus the backend
the automatic selector picks.  Results are also written to CSV, and plotted if
matplotlib is installed.

    python benchmarks/bench_scaling.py                 # default sweeps
    python benchmarks/bench_scaling.py --quick         # small/fast
    python benchmarks/bench_scaling.py --calibrate     # also store a calibration for empca_auto
    python benchmarks/bench_scaling.py --csv out.csv --plot out.png

Time per iteration is (t(niter=6) - t(niter=1)) / 5, so fixed costs (device
upload, CUDA context, final model download) are excluded here but reported in
the 'fixed' column.  A backend is dropped from a sweep once one point takes
longer than --budget seconds.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import empca_auto as auto  # noqa: E402

BASE = dict(nobs=2000, nvar=1000, nvec=5)
SWEEPS = dict(
    nobs=[50, 200, 1000, 5000, 20000, 100000],
    nvar=[50, 200, 1000, 5000, 20000],
    nvec=[1, 2, 5, 10, 20, 32, 50],
)
QUICK_BASE = dict(nobs=500, nvar=200, nvec=5)
QUICK_SWEEPS = dict(nobs=[50, 500, 5000], nvar=[50, 200, 2000], nvec=[2, 5, 20])


def run(base, sweeps, budget, repeats):
    backends = auto.available_backends()
    if 'torch-cuda' in backends:
        auto._run('torch-cuda', *auto._synthetic(50, 20, 2), 1, 2, 0, 1, True)  # CUDA warm-up
    rows = []
    for axis, values in sweeps.items():
        dead = set()
        for v in values:
            shape = dict(base, **{axis: v})
            row = dict(axis=axis, value=v, **shape)
            for b in backends:
                if b in dead:
                    continue
                fixed, per = auto._fit_cost(b, shape['nobs'], shape['nvar'], shape['nvec'], repeats)
                row['%s_iter_s' % b] = per
                row['%s_fixed_s' % b] = fixed
                if fixed + 6 * per > budget:
                    dead.add(b)
            ok = {b: row['%s_iter_s' % b] + row['%s_fixed_s' % b] / 25.0
                  for b in backends if '%s_iter_s' % b in row}
            row['fastest@25it'] = min(ok, key=ok.get)
            row['auto_pick'] = auto.choose_backend(shape['nobs'], shape['nvar'], shape['nvec'])
            rows.append(row)
            print_row(row, backends)
    return rows


def print_row(row, backends):
    cells = []
    for b in backends:
        k = '%s_iter_s' % b
        cells.append('%s=%9.5f' % (b, row[k]) if k in row else '%s=%9s' % (b, '-'))
    print('%-5s=%-7d (nobs=%d nvar=%d nvec=%d)  s/iter: %s  fastest=%s auto=%s' % (
        row['axis'], row['value'], row['nobs'], row['nvar'], row['nvec'],
        '  '.join(cells), row['fastest@25it'], row['auto_pick']))
    sys.stdout.flush()


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot(rows, path, backends):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axes_names = list(dict.fromkeys(r['axis'] for r in rows))
    fig, axs = plt.subplots(1, len(axes_names), figsize=(5 * len(axes_names), 4))
    for ax, name in zip(axs, axes_names):
        sub = [r for r in rows if r['axis'] == name]
        for b in backends:
            pts = [(r['value'], r['%s_iter_s' % b]) for r in sub if '%s_iter_s' % b in r]
            if pts:
                ax.loglog(*zip(*pts), marker='o', label=b)
        ax.set_xlabel(name)
        ax.set_ylabel('seconds / EM iteration')
        ax.grid(True, which='both', alpha=0.3)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--budget', type=float, default=20.0, help='drop a backend from a sweep after a point exceeds this many seconds')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--csv', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv'))
    ap.add_argument('--plot', default=None, help='PNG path (needs matplotlib)')
    ap.add_argument('--calibrate', action='store_true', help='also run empca_auto.calibrate() and save it')
    ap.add_argument('--calibrate-only', action='store_true', help='only run empca_auto.calibrate() (skip the sweeps)')
    args = ap.parse_args(argv)

    print('environment:', auto.environment_fingerprint())
    print('backends   :', auto.available_backends())
    if args.calibrate_only:
        auto.calibrate(verbose=True, repeats=args.repeats)
        print('calibration saved to', auto.calibration_path())
        return
    base, sweeps = (QUICK_BASE, QUICK_SWEEPS) if args.quick else (BASE, SWEEPS)
    rows = run(base, sweeps, args.budget, args.repeats)
    write_csv(rows, args.csv)
    print('wrote', args.csv)
    if args.plot:
        try:
            plot(rows, args.plot, auto.available_backends())
            print('wrote', args.plot)
        except ImportError:
            print('matplotlib not installed; skipping plot')
    if args.calibrate:
        grid = auto.DEFAULT_GRID
        if args.quick:
            grid = dict(nobs=(100, 1000), nvar=(100, 1000), nvec=(2, 10))
        auto.calibrate(grid=grid, verbose=True)
        print('calibration saved to', auto.calibration_path())


if __name__ == '__main__':
    main()
