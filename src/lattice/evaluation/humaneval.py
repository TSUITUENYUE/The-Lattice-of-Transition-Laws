"""HumanEval as lm-eval's humaneval task scores it, with every candidate executed in a sandbox.

A completion is the decoded generation cut at the task's stop sequences in their order (the
special tokens kept while cutting, then removed by a round trip through the tokenizer, as the
LLaDA authors' evaluation does). The program is prompt + completion + the problem's tests +
check(entry point); it passes when it runs to its end within three seconds.

Generated code never runs on the host. A candidate runs in a fresh sandbox of one of four
runtimes, each with no network, a read-only root file system, a private writable /tmp and
resource limits:
    docker, podman   a rootless runtime with an active seccomp policy and a local image named by
                     its sha256 digest (no pulls): --network=none, --read-only, an unprivileged
                     user, all capabilities dropped, no new privileges, 32 processes, 4 GB, 1 CPU
    bwrap            bubblewrap, unprivileged: the host root bound read-only, a new /dev and
                     /proc, a tmpfs on /tmp, every namespace unshared (so no network), killed with
                     its parent, in a new session
    unshare          unprivileged namespaces made with util-linux's unshare, for hosts without
                     bubblewrap: new user, mount, network and process namespaces, every mounted
                     file system (other than /proc, /sys and /dev) bound again read-only, and
                     private tmpfs on /tmp, /var/tmp and /dev/shm
Inside, the runner limits the address space to 4 GB, CPU time to 4 s, processes to 16 and core
files to none, disables os._exit, and times the program out after 3 s. check_sandbox() verifies
the sandbox before any candidate runs: the network is unreachable, the root and home directories
cannot be written, /tmp is not the host's, every canonical solution passes, every empty body
fails and an endless loop times out. A failure of the sandbox itself stops the evaluation; it is
never counted as a wrong answer.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

UNTIL = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
TASKS = 164
RUNTIMES = ("docker", "podman", "bwrap", "unshare")

# Executed only inside a sandbox: it limits resources, it does not isolate by itself.
RUNNER = r'''
import contextlib, io, json, os, resource, signal, sys
payload = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (4, 5))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os._exit = None
def timed_out(signum, frame):
    raise TimeoutError('candidate exceeded three seconds')
signal.signal(signal.SIGALRM, timed_out)
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        signal.setitimer(signal.ITIMER_REAL, 3.0)
        exec(payload['program'], {'__name__': '__main__'})
    result = {'status': 'passed'}
except TimeoutError:
    result = {'status': 'timeout'}
except BaseException as exc:
    result = {'status': 'failed', 'exception': type(exc).__name__}
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
sys.__stdout__.write(json.dumps(result) + '\n')
'''


# unshare: remount every file system read-only and give /tmp, /var/tmp and /dev/shm private tmpfs,
# then run the command in /tmp
UNSHARE_SETUP = r'''
awk "{print \$2}" /proc/self/mounts | sort -u | while read -r m; do
  case "$m" in /proc*|/sys*|/dev*) continue ;; esac
  mount --bind "$m" "$m" 2>/dev/null && mount -o remount,bind,ro "$m" 2>/dev/null
done
for t in /tmp /var/tmp /dev/shm; do mount -t tmpfs -o size=64m tmpfs "$t" || exit 90; done
cd /tmp && exec "$@"
'''


class SandboxUnavailable(RuntimeError):
    """The sandbox cannot be used or failed; no result may be scored from it."""


def load_problems(path):
    """The 164 problems of HumanEval.jsonl(.gz), in order."""
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    if len(problems) != TASKS or len({p["task_id"] for p in problems}) != TASKS:
        raise ValueError(f"expected {TASKS} distinct problems in {path}")
    return problems


def data_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def completion_of(tokenizer, ids):
    """The generation cut at each stop sequence in order, then stripped of special tokens."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for stop in UNTIL:
        text = text.split(stop)[0]
    return tokenizer.decode(tokenizer(text)["input_ids"], skip_special_tokens=True)


def program_of(problem, completion):
    return problem["prompt"] + completion + "\n" + problem["test"] + "\ncheck(" + problem["entry_point"] + ")"


def verify_sandbox(runtime, image=None):
    """The sandbox's description, after checking the runtime can provide the isolation above."""
    if runtime not in RUNTIMES or not shutil.which(runtime):
        raise SandboxUnavailable(f"{runtime!r} is not available; use one of {RUNTIMES}")
    try:
        if runtime in ("bwrap", "unshare"):
            version = subprocess.check_output([runtime, "--version"], text=True, timeout=20).strip()
            return dict(runtime=runtime, version=version, python=sys.executable, network="none",
                        read_only_root=True, private_tmp=True, timeout_seconds=3)
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image or ""):
            raise SandboxUnavailable("name a local image by its sha256 digest; tags and pulls are refused")
        if runtime == "docker":
            info = json.loads(subprocess.check_output(
                ["docker", "info", "--format", "{{json .SecurityOptions}}"], text=True, timeout=20))
            rootless = any("rootless" in str(x) for x in info)
            seccomp = any("seccomp" in str(x) and "unconfined" not in str(x) for x in info)
        else:
            info = json.loads(subprocess.check_output(["podman", "info", "--format", "json"],
                                                      text=True, timeout=20))
            security = info.get("host", {}).get("security", {})
            rootless, seccomp = bool(security.get("rootless")), bool(security.get("seccompEnabled"))
        if not rootless:
            raise SandboxUnavailable("the container runtime is not rootless")
        if not seccomp:
            raise SandboxUnavailable("the container runtime has no active seccomp policy")
        image_id = subprocess.check_output([runtime, "image", "inspect", image, "--format", "{{.Id}}"],
                                           text=True, timeout=20).strip()
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        raise SandboxUnavailable(f"sandbox preflight failed: {e}") from e
    return dict(runtime=runtime, image=image, image_id=image_id, rootless=True, network="none",
                read_only_root=True, private_tmp=True, memory="4g", cpus=1, timeout_seconds=3)


def sandbox_command(verified, name):
    """The argv that runs RUNNER in a fresh sandbox."""
    runtime = verified["runtime"]
    if runtime == "bwrap":
        return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                "--unshare-all", "--die-with-parent", "--new-session", "--chdir", "/tmp",
                verified["python"], "-I", "-c", RUNNER]
    if runtime == "unshare":
        return ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "--mount-proc",
                "bash", "-c", UNSHARE_SETUP, "sandbox", verified["python"], "-I", "-c", RUNNER]
    return [runtime, "run", "--rm", "--pull=never", "--name", name, "--network=none",
            "--read-only", "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=32", "--memory=4g", "--memory-swap=4g", "--cpus=1",
            "--ulimit=core=0", "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m", "--workdir=/tmp",
            "--env=PYTHONHASHSEED=0", "--interactive", verified["image"], "python", "-I", "-c", RUNNER]


def run_in_sandbox(program, verified):
    """{'status': 'passed' | 'failed' | 'timeout', ...} of one program in a fresh sandbox."""
    if not verified or verified.get("runtime") not in RUNTIMES or verified.get("network") != "none":
        raise SandboxUnavailable("the sandbox was not verified")
    name = "humaneval-" + uuid.uuid4().hex
    try:
        result = subprocess.run(sandbox_command(verified, name), input=json.dumps(dict(program=program)),
                                capture_output=True, text=True, timeout=25)
        if result.returncode != 0:
            raise SandboxUnavailable(f"the sandbox exited {result.returncode}: {result.stderr[-1000:]}")
        answer = json.loads(result.stdout.strip().splitlines()[-1])
        if answer.get("status") not in ("passed", "failed", "timeout"):
            raise SandboxUnavailable("the sandbox returned no valid result")
        return answer
    except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
        raise SandboxUnavailable(f"the sandbox failed; not a wrong answer: {e}") from e
    finally:
        if verified["runtime"] in ("docker", "podman"):
            try:
                subprocess.run([verified["runtime"], "rm", "--force", name], capture_output=True, timeout=15)
            except (subprocess.SubprocessError, OSError) as e:
                raise SandboxUnavailable(f"could not remove the container {name}: {e}") from e


def _probes(token, host_marker):
    """Programs that pass only when the sandbox has the property named by their key."""
    return {
        "network_unreachable": (
            "import socket\n"
            "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3).close()\n"
            "except OSError:\n    pass\n"
            "else:\n    raise AssertionError('network reachable')\n"),
        "root_read_only": (
            f"try:\n    open('/.sandbox_probe_{token}', 'w').close()\n"
            "except OSError:\n    pass\n"
            "else:\n    raise AssertionError('root writable')\n"),
        "home_read_only": (
            f"import os\ntry:\n    open(os.path.join(os.path.expanduser('~'), '.sandbox_probe_{token}'), 'w').close()\n"
            "except OSError:\n    pass\n"
            "else:\n    raise AssertionError('home writable')\n"),
        "private_tmp": (
            f"import os\nassert not os.path.exists({host_marker!r}), 'host /tmp visible'\n"
            f"open('/tmp/sandbox_probe_{token}', 'w').close()\n"),
    }


def check_sandbox(problems, verified):
    """The record of the sandbox checks; raises SandboxUnavailable if any fails."""
    token = uuid.uuid4().hex
    with tempfile.NamedTemporaryFile(dir="/tmp", prefix="host_probe_") as marker:
        probes = {k: run_in_sandbox(program, verified)["status"] == "passed"
                  for k, program in _probes(token, marker.name).items()}
    # a probe's file must not reach the host; the probe's own files are removed if one did
    leaked = []
    for key, path in (("private_tmp", "/tmp/sandbox_probe_" + token),
                      ("root_read_only", "/.sandbox_probe_" + token),
                      ("home_read_only", os.path.join(os.path.expanduser("~"), ".sandbox_probe_" + token))):
        if os.path.exists(path):
            probes[key] = False
            leaked.append(path)
            os.remove(path)
    canonical = sum(run_in_sandbox(program_of(p, p["canonical_solution"]), verified)["status"] == "passed"
                    for p in problems)
    empty = sum(run_in_sandbox(program_of(p, "    pass\n"), verified)["status"] == "passed" for p in problems)
    loop = run_in_sandbox("while True:\n    pass\n", verified)["status"]
    record = dict(sandbox=verified, probes=probes, canonical_passed=canonical, empty_passed=empty,
                  endless_loop=loop, total=len(problems), leaked_files=leaked)
    if not all(probes.values()) or canonical != len(problems) or empty != 0 or loop != "timeout":
        raise SandboxUnavailable(f"the sandbox check failed: {json.dumps(record)}")
    return record
