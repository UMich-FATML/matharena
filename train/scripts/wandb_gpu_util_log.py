#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import select
import subprocess
import sys
import time

stop = False


def on_stop(signum, frame):
    global stop
    stop = True


def as_float(value):
    value = value.strip()
    return float('nan') if not value or value.upper() == 'N/A' else float(value)


def finite(values):
    return [v for v in values if v == v]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nodes', required=True)
    p.add_argument('--gpus-per-node', type=int, required=True)
    p.add_argument('--trainer-nodes', type=int, default=1)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--project', required=True)
    p.add_argument('--group', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--from-stdin', action='store_true')
    args = p.parse_args()

    if not os.environ.get('WANDB_API_KEY'):
        print('WANDB_API_KEY not set; skipping W&B GPU monitor.', file=sys.stderr)
        return 0

    import wandb

    nodes = [n for n in args.nodes.split(',') if n]
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    expected_rows = max(1, len(nodes) * args.gpus_per_node)
    os.environ.setdefault('WANDB_DISABLE_STATS', 'true')
    settings = wandb.Settings(
        mode='shared',
        x_primary=False,
        x_update_finish_state=False,
        x_disable_stats=True,
        console='off',
    )
    run = wandb.init(
        project=args.project,
        group=args.group,
        name=args.name,
        id=os.environ.get('WANDB_RUN_ID'),
        resume=os.environ.get('WANDB_RESUME', 'allow'),
        job_type='gpu_monitor',
        settings=settings,
        config={
            'gpu_monitor_nodes': nodes,
            'gpu_monitor_gpus_per_node': args.gpus_per_node,
            'gpu_monitor_trainer_nodes': args.trainer_nodes,
            'gpu_monitor_interval_s': args.interval,
            'gpu_monitor_slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        },
    )
    wandb.define_metric('cluster_gpu/*', step_metric='cluster_gpu/time_s')
    print(f'Started W&B GPU monitor in run={run.id}', flush=True)

    query = 'index,utilization.gpu,memory.used,memory.total,power.draw'
    remote = (
        'while true; do '
        'ts=$(date -Is); node=$(hostname); '
        f'nvidia-smi --query-gpu={query} --format=csv,noheader,nounits | sed "s/^/${{ts}},${{node}},/"; '
        f'sleep {args.interval}; '
        'done'
    )
    cmd = [
        'srun', '--overlap', f'--nodes={len(nodes)}', f'--ntasks={len(nodes)}',
        '--ntasks-per-node=1', '-w', ','.join(nodes), '--mpi=pmix',
        '--network=disable_rdzv_get', '--environment=./train/env/cscs-verl.toml',
        'bash', '-lc', remote,
    ]

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    if args.from_stdin:
        print('GPU monitor reading rows from stdin', flush=True)
        proc = None
        stream = sys.stdin
    else:
        print('GPU monitor command:', ' '.join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stream = proc.stdout
    rows = []
    ignored_lines = 0
    logged_batches = 0
    started_at = time.time()
    try:
        while not stop:
            if stream is None:
                break
            ready, _, _ = select.select([stream], [], [], 1.0)
            if not ready:
                if proc is not None and proc.poll() is not None:
                    break
                continue
            line = stream.readline()
            if not line:
                if proc is not None and proc.poll() is not None:
                    break
                if proc is None:
                    break
                continue
            try:
                fields = next(csv.reader([line.strip()]))
                if len(fields) != 7:
                    if ignored_lines < 20:
                        print(f'GPU monitor ignored line: {line.strip()}', flush=True)
                    ignored_lines += 1
                    continue
                ts, node, gpu, util, mem_used, mem_total, power = [x.strip() for x in fields]
                node_idx = node_to_idx.get(node)
                if node_idx is None:
                    continue
                role = 'trainer' if node_idx < args.trainer_nodes else 'rollout'
                mem_used = as_float(mem_used)
                mem_total = as_float(mem_total)
                rows.append({
                    'role': role,
                    'node_idx': node_idx,
                    'gpu': gpu,
                    'util': as_float(util),
                    'mem_used_gb': mem_used / 1024.0,
                    'mem_frac': mem_used / mem_total if mem_total > 0 else float('nan'),
                    'power': as_float(power),
                })
            except Exception as exc:
                print(f'GPU monitor parse error: {exc}: {line.strip()}', file=sys.stderr)
                continue

            if len(rows) < expected_rows:
                continue

            metrics = {'cluster_gpu/time_s': time.time() - started_at}
            for row in rows:
                base = f"cluster_gpu/{row['role']}/node_{row['node_idx']}/gpu_{row['gpu']}"
                metrics[f'{base}/util_pct'] = row['util']
                metrics[f'{base}/mem_used_gb'] = row['mem_used_gb']
                metrics[f'{base}/mem_frac'] = row['mem_frac']
                metrics[f'{base}/power_w'] = row['power']
            for role in ('trainer', 'rollout'):
                subset = [r for r in rows if r['role'] == role]
                vals = finite([r['util'] for r in subset])
                if vals:
                    metrics[f'cluster_gpu/{role}/util_pct_mean'] = sum(vals) / len(vals)
                    metrics[f'cluster_gpu/{role}/util_pct_max'] = max(vals)
            vals = finite([r['util'] for r in rows])
            if vals:
                metrics['cluster_gpu/all/util_pct_mean'] = sum(vals) / len(vals)
                metrics['cluster_gpu/all/util_pct_max'] = max(vals)
            wandb.log(metrics)
            logged_batches += 1
            print(
                'GPU monitor logged batch',
                logged_batches,
                f"all_mean={metrics.get('cluster_gpu/all/util_pct_mean')}",
                f"trainer_mean={metrics.get('cluster_gpu/trainer/util_pct_mean')}",
                f"rollout_mean={metrics.get('cluster_gpu/rollout/util_pct_mean')}",
                flush=True,
            )
            rows = []
        if proc is not None:
            remaining = proc.stdout.read() if proc.stdout else ''
            if remaining:
                print('GPU monitor remaining output:', remaining.strip(), flush=True)
            rc = proc.poll()
            if rc is None:
                proc.terminate()
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait(timeout=5)
            print(f'GPU monitor srun exited rc={rc}, logged_batches={logged_batches}', flush=True)
        else:
            print(f'GPU monitor input ended, logged_batches={logged_batches}', flush=True)
    finally:
        wandb.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
