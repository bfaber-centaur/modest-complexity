"""Read-only runtime inventory and phase timing; no optional dependencies."""
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def peak_rss_bytes():
    """Lifetime high-water mark of this process only; excludes subprocesses."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def inventory(out):
    import numpy as np
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    root = Path(__file__).parent
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    # Preserve the actual membership and mount mapping, not a guessed parent limit.
    cgroup = {}
    if sys.platform == 'linux':
        for name in ('/proc/self/cgroup', '/proc/self/mountinfo'):
            try:
                content = Path(name).read_text()
                cgroup[name] = content if name.endswith('cgroup') else '\n'.join(
                    line for line in content.splitlines() if ' - cgroup' in line)
            except OSError:
                pass
        for membership in cgroup.get('/proc/self/cgroup', '').splitlines():
            _, controllers, member = membership.split(':', 2)
            for line in cgroup.get('/proc/self/mountinfo', '').splitlines():
                fields = line.split(); sep = fields.index('-')
                fstype, mountroot, mountpoint = fields[sep+1], fields[3], fields[4]
                if not ((fstype == 'cgroup2' and not controllers) or
                        (fstype == 'cgroup' and set(controllers.split(',')) & set(fields[sep+3].split(',')))):
                    continue
                try:
                    rel = Path(member).relative_to(mountroot)
                except ValueError:
                    continue
                leaf = Path(mountpoint) / rel
                # Include ancestors: an unlimited leaf does not override a parent cap.
                for folder in (leaf, *leaf.parents):
                    if not folder.is_relative_to(mountpoint):
                        break
                    for filename in ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective',
                                     'memory.max', 'memory.current', 'memory.events',
                                     'cpu.cfs_quota_us', 'cpu.cfs_period_us',
                                     'memory.limit_in_bytes', 'memory.usage_in_bytes'):
                        p = folder / filename
                        try:
                            cgroup[str(p)] = p.read_text().strip()
                        except OSError:
                            pass
    disk = shutil.disk_usage(out)
    return dict(python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), machine=platform.machine(),
                visible_cpus=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                thread_environment={k: os.environ.get(k) for k in THREAD_VARS},
                numpy_config=buffer.getvalue(), cgroup=cgroup,
                disk_free_bytes=disk.free, git_commit=git('rev-parse', 'HEAD'),
                git_status=git('status', '--porcelain'),
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(root.glob('*.py'))},
                environment_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in ('pyproject.toml', 'uv.lock', '.python-version')
                                    if (root/name).exists()},
                peak_rss_scope='current process lifetime; excludes children',
                cgroup_note='Raw process membership and visible ancestor limits; unavailable on macOS.')


class Timings:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.phases = {}

    @contextmanager
    def measure(self, name):
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            row = self.phases.setdefault(name, dict(calls=0, wall_seconds=0., cpu_seconds=0.))
            row['calls'] += 1
            row['wall_seconds'] += time.perf_counter() - wall
            row['cpu_seconds'] += time.process_time() - cpu

    def snapshot(self):
        return dict(wall_seconds=time.perf_counter()-self.start,
                    cpu_seconds=time.process_time()-self.cpu_start,
                    peak_rss_bytes=peak_rss_bytes(), phases=self.phases)
