"""Starts, feeds and restarts the editing model's Python kernel (kernel_server.py) in a sandbox.

docker: `docker run --network none` with /model and /practice read-only and the run's work folder at /work
        (the paths the task message names). Recommended.
local:  a subprocess on this machine; the network is cut with `unshare -rn` when that works. The paths /model,
        /practice and /work are symlinked when possible (e.g. as root in a container); otherwise the harness rewrites
        them in every message (see Kernel.paths).
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import MODEL_DIR, PRACTICE_DIR

SERVER = Path(__file__).resolve().parent / "kernel_server.py"


def _bytes(s):
    s = str(s).strip().lower()
    mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}.get(s[-1:], 1)
    return int(float(s.rstrip("kmgt")) * mult)


def _try_symlink(link, target):
    link = Path(link)
    try:
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            return link.resolve() == Path(target).resolve()
        link.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        return False


class Kernel:
    def __init__(self, cfg, work_dir, log_path):
        self.cfg = cfg
        self.work = Path(work_dir).resolve()
        self.log = open(log_path, "a", encoding="utf-8")
        self.proc = None
        self.q = None
        self.name = None
        self.mode = cfg["mode"]
        # how the model sees the paths: identical (/model, /practice, /work) unless local mode cannot symlink them
        self.paths = {"/model": "/model", "/practice": "/practice", "/work": "/work"}
        if self.mode == "local":
            # symlinks at the filesystem root only on Linux and only when allowed (e.g. root in a container);
            # never on Windows, where /model would become C:\model
            ok = os.name == "posix" and all(_try_symlink(a, b) for a, b in (("/model", MODEL_DIR), ("/practice", PRACTICE_DIR),
                                                      ("/work", self.work)))
            if not ok:
                self.paths = {"/model": MODEL_DIR.resolve().as_posix(), "/practice": PRACTICE_DIR.resolve().as_posix(),
                              "/work": self.work.as_posix()}

    # ------------------------------------------------------------------ process control
    def _cmd(self):
        if self.mode == "docker":
            self.name = f"weight-space-bench-kernel-{uuid.uuid4().hex[:8]}"
            cmd = ["docker", "run", "--rm", "-i", "--name", self.name, "--network", "none",
                   "--memory", self.cfg["memory"], "--cpus", str(self.cfg["cpus"]),
                   "-e", "CUDA_VISIBLE_DEVICES=", "-e", "OMP_NUM_THREADS=" + str(self.cfg["cpus"]),
                   "-v", f"{MODEL_DIR.resolve()}:/model:ro", "-v", f"{PRACTICE_DIR.resolve()}:/practice:ro",
                   "-v", f"{self.work}:/work", "-w", "/work"]
            if hasattr(os, "getuid"):
                cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
            return cmd + [self.cfg["image"], "python", "-u", "/opt/weight_space_bench/kernel_server.py"], None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WSB_MODEL=self.paths["/model"], WSB_WORK=self.paths["/work"],
                   OMP_NUM_THREADS=str(self.cfg["cpus"]))
        cmd = [sys.executable, "-u", str(SERVER)]
        if shutil.which("unshare") and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0:
            cmd = ["unshare", "-rn"] + cmd
        elif os.environ.get("WSB_ALLOW_NETWORK") != "1":
            raise SystemExit("kernel.mode = 'local' cannot cut the network here (`unshare -rn` failed), so the editing "
                             "model would not be isolated. Use kernel.mode = 'docker', or set WSB_ALLOW_NETWORK=1 for a "
                             "test run that is not benchmark-valid.")
        return cmd, env

    def start(self):
        cmd, env = self._cmd()
        pre = None
        if self.mode == "local" and hasattr(os, "setsid"):
            mem = _bytes(self.cfg["memory"])

            def pre():  # noqa: E306
                import resource
                os.setsid()
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                     cwd=str(self.work), preexec_fn=pre, text=True, encoding="utf-8", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        msg = self._get(300)
        if not msg or not msg.get("ready"):
            raise RuntimeError(f"kernel did not start (see {self.log.name})")

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            try:
                q.put(json.loads(line))
            except ValueError:
                continue
        q.put(None)

    def _get(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return "timeout"

    def kill(self):
        if self.proc is None:
            return
        if self.mode == "docker" and self.name:
            subprocess.run(["docker", "kill", self.name], capture_output=True)
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None

    def restart(self):
        self.kill()
        self.start()

    # ------------------------------------------------------------------ running code
    def run(self, code):
        """-> (output text, restarted). Output is cut and made ASCII as the task message says."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        t0 = time.time()
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        msg = self._get(self.cfg["timeout_s"])
        restarted = False
        if msg == "timeout":
            out = (f"[the run exceeded {self.cfg['timeout_s']} seconds and was stopped; the kernel was restarted, "
                   "so its variables are gone]")
            self.restart()
            restarted = True
        elif msg is None:
            out = "[the kernel died (probably the memory limit) and was restarted, so its variables are gone]"
            self.restart()
            restarted = True
        else:
            out = msg.get("out", "")
        out = out.encode("ascii", "replace").decode("ascii")
        n = self.cfg["output_chars"]
        if len(out) > n:
            out = out[:n] + f"\n[output cut at {n:,} characters]"
        return out or "(no output)", restarted, time.time() - t0
