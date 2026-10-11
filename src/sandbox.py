import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (
    SANDBOX_IMAGE, SANDBOX_TIMEOUT_SEC, SANDBOX_MEMORY,
    SANDBOX_CPUS, MAX_OUTPUT_CHARS,
)


@dataclass
class SandboxResult:
    """Everything we learn from running one piece of code."""
    ok: bool                  # True only if the code finished cleanly
    stdout: str               # what the code printed
    stderr: str               # error messages (tracebacks)
    returncode: int           # 0 = success; 137 usually means killed for using too much memory
    timed_out: bool           # True if we had to kill it for running too long
    new_files: list = field(default_factory=list)  # files the code created in the workspace

    def summary(self) -> str:
        """A compact text version to feed back to the LLM later."""
        parts = [f"ok={self.ok}", f"returncode={self.returncode}"]
        if self.timed_out:
            parts.append("TIMED OUT")
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        if self.new_files:
            parts.append(f"NEW FILES: {self.new_files}")
        return "\n".join(parts)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and end of long output. Errors usually appear at the end."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[output truncated]...\n" + text[-half:]


def run_in_sandbox(code: str, workspace: Path, timeout: int = SANDBOX_TIMEOUT_SEC) -> SandboxResult:
    """Run a string of Python code inside a locked-down Docker container.

    workspace: a host folder that is shared with the container as /workspace.
               Put data files there; the code can save charts/models there.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # Save the code as a file the container will run
    (workspace / "step.py").write_text(code, encoding="utf-8")

    # Remember which files existed before, so we can detect new ones afterwards
    files_before = {p.name for p in workspace.iterdir()}

    # Give the container a unique name so we can force-kill it on timeout
    container_name = f"ads-{uuid.uuid4().hex[:8]}"

    cmd = [
        "docker", "run",
        "--rm",                                   # delete the container when finished
        "--name", container_name,
        "--network", "none",                      # no internet
        "--memory", SANDBOX_MEMORY,               # RAM cap
        "--cpus", SANDBOX_CPUS,                   # CPU cap
        "--pids-limit", "128",                    # no process explosions
        "--security-opt", "no-new-privileges",
        "--read-only",                            # container filesystem is read-only...
        "--tmpfs", "/tmp",                        # ...except a temporary folder
        "-e", "MPLCONFIGDIR=/tmp/mpl",            # matplotlib needs a writable config dir
        "-e", "HOME=/tmp",
        "-v", f"{workspace}:/workspace",          # share the workspace folder
        "-w", "/workspace",
        SANDBOX_IMAGE,
        "python", "step.py",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        stdout, stderr = proc.stdout, proc.stderr
        returncode, timed_out = proc.returncode, False

    except subprocess.TimeoutExpired:
        # Python killed the docker command, but the container may still be running. Remove it.
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        stdout, stderr = "", f"Execution exceeded the {timeout}s time limit and was killed."
        returncode, timed_out = -1, True

    except FileNotFoundError:
        raise RuntimeError("Docker was not found. Is Docker Desktop installed and running?")

    files_after = {p.name for p in workspace.iterdir()}
    new_files = sorted(files_after - files_before)

    return SandboxResult(
        ok=(returncode == 0 and not timed_out),
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        returncode=returncode,
        timed_out=timed_out,
        new_files=new_files,
    )