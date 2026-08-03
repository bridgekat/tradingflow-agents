"""File-system, search, shell, and web tools exposed to the agent.

Every function tool returns a plain string — including error messages — so the
model can observe a failure and react to it instead of the whole run aborting.
Outputs are truncated at fixed caps to keep them from flooding the model's
context window.

`web_search` is the exception: it is the SDK's built-in `WebSearchTool`, a
*hosted* tool that DeepSeek executes server-side (it is the one hosted tool
type DeepSeek's Responses API implements — `file_search`, `code_interpreter`
and `computer_use` are silently ignored). Its results never pass through this
process, so there is nothing here to truncate. `web_fetch` stays a local
function tool: it retrieves a specific URL verbatim, which hosted search does
not do.
"""

from __future__ import annotations

import asyncio
import atexit
import codecs
import functools
import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from html.parser import HTMLParser
import httpx
from agents import Tool, WebSearchTool, function_tool

# Caps on tool output size.
_MAX_READ_LINES = 2000
_MAX_LINE_CHARS = 1000
_MAX_OUTPUT_CHARS = 50_000
_MAX_JOB_BUFFER_CHARS = 200_000
_MAX_MATCHES = 200
_MAX_GLOB_RESULTS = 500
_MAX_FETCH_CHARS = 20_000
_MAX_FETCH_BYTES = 5_000_000
_MAX_WAIT_SECONDS = 60

# Some sites reject non-browser user agents outright.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# When False, `run_command` asks for confirmation on stdin before executing
# (or refuses if stdin is not a terminal). The CLI sets this from `--yes`.
AUTO_APPROVE = False


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated; {len(text) - limit} more characters]"


def _sleep_until_keypress(seconds: float) -> bool:
    """Sleep for up to `seconds`; return True if a keypress ended it early.

    Keypresses can only be observed when stdin is a terminal; otherwise this
    is a plain uninterruptible sleep. Any keystrokes already buffered before
    the wait starts are discarded so a stale key cannot cancel it instantly.
    """

    deadline = time.monotonic() + seconds

    if not sys.stdin.isatty():
        time.sleep(seconds)
        return False

    if os.name == "nt":
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if msvcrt.kbhit():
                msvcrt.getwch()
                return True
            time.sleep(min(0.05, remaining))

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # deliver single keystrokes without waiting for Enter
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            readable, _, _ = select.select([fd], [], [], remaining)
            if readable:
                os.read(fd, 1)
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


@functools.lru_cache(maxsize=1)
def shell_executable() -> str:
    """The shell `run_command` executes through. On Windows this prefers
    PowerShell 7+ (`pwsh`), then Windows PowerShell, then cmd.exe; on POSIX it
    is /bin/sh. Cached so the model's instructions and `run_command` can never
    disagree. Reported to the model so it writes the right dialect."""
    if os.name == "nt":
        for name in ("pwsh", "powershell"):
            exe = shutil.which(name)
            if exe:
                return exe
        return os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
    return "/bin/sh"


def _shell_argv(command: str) -> list[str]:
    """Build the argv that runs `command` through `run_command_shell()`."""
    shell = shell_executable()
    match Path(shell).stem.lower():
        case "pwsh" | "powershell":
            # -NoProfile: fast and reproducible startup; -NonInteractive: fail
            # instead of hanging when something prompts for input.
            #
            # `powershell -Command` exits with a bare 0/1, discarding a native
            # command's exit code, so propagate $LASTEXITCODE ourselves: keep 1
            # for failures with no native code (cmdlet errors), and let an
            # explicit `exit` inside the command win by exiting earlier.
            # NativeCommandError(-Message) is exempt from the cmdlet-failure
            # fallback: it is how PowerShell 5.1 reports a `2>&1`-redirected
            # native command's stderr, which flips $? even on success. The
            # wrapper lines are joined with newlines, not `;`, so a trailing
            # `#` comment in the command cannot swallow them.
            #
            # The encoding preamble makes PowerShell read native output, write
            # its own output, and feed native stdin in UTF-8, and (via the
            # console codepage) tells console-aware natives to emit UTF-8 too.
            wrapped = "\n".join([
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
                "$OutputEncoding = [System.Text.Encoding]::UTF8",
                "$LASTEXITCODE = 0",
                command,
                "if (-not $? -and $LASTEXITCODE -eq 0) {",
                "  $eid = if ($Error.Count) { $Error[0].FullyQualifiedErrorId } else { '' }",
                "  if ($eid -notin 'NativeCommandError', 'NativeCommandErrorMessage') { exit 1 }",
                "}",
                "exit $LASTEXITCODE",
            ])  # fmt: skip
            return [shell, "-NoProfile", "-NonInteractive", "-Command", wrapped]
        case "cmd":
            # chcp 65001 switches the (private, see _spawn) console to UTF-8;
            # errorlevel still comes from the user's command, not chcp.
            return [shell, "/d", "/s", "/c", f"chcp 65001>nul & {command}"]
        case _:
            return [shell, "-c", command]


def _rg_executable() -> str | None:
    """Locate ripgrep: the system PATH first, then the binary the `ripgrep`
    pip package installs next to the Python interpreter."""
    exe = shutil.which("rg")
    if exe:
        return exe
    bundled = Path(sys.executable).parent / ("rg.exe" if os.name == "nt" else "rg")
    return str(bundled) if bundled.is_file() else None


async def _spawn(argv: list[str], merge_stderr: bool = False) -> asyncio.subprocess.Process:
    """Start a subprocess with its output captured in pipes.

    All tool I/O is UTF-8, so children are nudged to produce it: Python
    subprocesses via PYTHONIOENCODING/PYTHONUTF8 (piped Python otherwise
    falls back to the ANSI codepage on Windows), the shells via _shell_argv's
    preambles. On Windows the child gets its own invisible console
    (CREATE_NO_WINDOW) so those preambles cannot reconfigure the codepage of
    the terminal the harness runs in; on POSIX it gets its own session, so
    its process group is `pid` and `_kill_tree` can take out descendants
    with one killpg.
    """

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    return await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE,
        env=env,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        start_new_session=(False if os.name == "nt" else True),
    )


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a `_spawn`ed process and (best-effort) its descendants, then reap it."""

    if proc.returncode is None:
        if os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/F", "/T", "/PID", str(proc.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )  # fmt: skip
                await killer.wait()
            except OSError:
                pass
        else:
            import signal

            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    await proc.wait()


async def _run_capture(argv: list[str], timeout_seconds: float) -> tuple[int, str, str]:
    """Run a subprocess without blocking the event loop and capture its output.

    Returns (returncode, stdout, stderr); raises TimeoutError (after killing
    the process and its descendants) or OSError.
    """

    proc = await _spawn(argv)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except (TimeoutError, asyncio.TimeoutError):
        await _kill_tree(proc)
        raise TimeoutError(f"timed out after {timeout_seconds:g} seconds") from None

    return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


class _Job:
    """A background command: its process plus a rolling buffer of its output.

    stdout and stderr are merged into one stream so `check_command` can show
    them in order. The buffer keeps only the most recent
    `_MAX_JOB_BUFFER_CHARS` characters; `discarded` counts what was dropped
    from the front so offsets stay absolute.
    """

    def __init__(self, job_id: int, command: str, proc: asyncio.subprocess.Process) -> None:
        self.id = job_id
        self.command = command
        self.proc = proc
        self.output = ""
        self.discarded = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pump_task: asyncio.Task[None] | None = None

    def _append(self, text: str) -> None:
        if not text:
            return
        self.output += text
        overflow = len(self.output) - _MAX_JOB_BUFFER_CHARS
        if overflow > 0:
            self.output = self.output[overflow:]
            self.discarded += overflow

    async def pump(self) -> None:
        """Drain the process's output into the buffer until it exits."""
        assert self.proc.stdout is not None
        while True:
            chunk = await self.proc.stdout.read(65536)
            if not chunk:
                break
            self._append(self._decoder.decode(chunk))
        self._append(self._decoder.decode(b"", final=True))
        await self.proc.wait()


_jobs: dict[int, _Job] = {}
_job_ids = itertools.count(1)


@atexit.register
def kill_remaining_jobs() -> None:
    """Kill background jobs still running when the harness exits.

    Runs after the event loop is gone, so it must use synchronous kills.
    """

    import signal

    for job in _jobs.values():
        if job.proc.returncode is not None:
            continue
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(job.proc.pid)],
                    capture_output=True,
                )
            else:
                os.killpg(job.proc.pid, signal.SIGKILL)
        except Exception:
            pass


@function_tool
def wait(seconds: float) -> str:
    """Wait for a given duration, e.g. for an external process or a rate limit.

    While waiting, the user can press any key to end the wait immediately.
    The result reports how long was actually waited and whether the wait ran
    to completion or was cut short by the user.

    Args:
        seconds: How long to wait, in seconds (at most 60; wait repeatedly for longer pauses).
    """

    if not seconds > 0:
        return "Error: seconds must be positive."
    if seconds > _MAX_WAIT_SECONDS:
        return f"Error: seconds must be at most {_MAX_WAIT_SECONDS}; wait repeatedly for longer pauses."

    if sys.stdin.isatty():
        print(f"\n  waiting {seconds:g}s - press any key to end the wait early", flush=True)

    start = time.monotonic()
    try:
        stopped_by_user = _sleep_until_keypress(seconds)
    except OSError as e:
        return f"Error: {e}"
    elapsed = time.monotonic() - start

    if stopped_by_user:
        return f"waited for {elapsed:.1f} seconds (user ended the wait)"
    return f"waited for {elapsed:.1f} seconds"


@function_tool
async def run_command(command: str, wait_seconds: float = 10) -> str:
    """Run a shell command in the working directory and return its output.

    Use this to build, test, lint, or inspect things the other tools cannot.
    The command runs through the shell named in your environment information;
    the user may be asked to approve it first. stdout and stderr are merged.

    If the command finishes within `wait_seconds` you get its exit code and
    output directly. Otherwise it keeps running as a background job and you
    get a job id plus the output so far: poll it with `check_command` (use
    `wait` between polls) and stop it with `kill_command`. Commands are never
    killed by a timeout. Set `wait_seconds` high to sit out something like a
    long build in one call, or to 0 to launch e.g. a server and move on.

    Args:
        command: The shell command line to execute.
        wait_seconds: How long to wait before letting the command continue in the background (at most 60).
    """

    if not 0 <= wait_seconds <= _MAX_WAIT_SECONDS:
        return f"Error: wait_seconds must be between 0 and {_MAX_WAIT_SECONDS}."

    if not AUTO_APPROVE:
        if not sys.stdin.isatty():
            return (
                "Error: command not run — approval required but stdin is not a terminal (rerun the harness with --yes)."
            )

        try:
            reply = await asyncio.to_thread(input, f"\n  approve command? [y/N] {command}\n  > ")
        except EOFError:
            return "Error: command not run — could not read approval from stdin (rerun the harness with --yes)."

        if reply.strip().lower() not in ("y", "yes"):
            return "Error: command rejected by the user."

    try:
        proc = await _spawn(_shell_argv(command), merge_stderr=True)
    except OSError as e:
        return f"Error running command: {e}"

    job = _Job(next(_job_ids), command, proc)
    _jobs[job.id] = job
    job.pump_task = asyncio.create_task(job.pump())

    # `shield` so the grace-period timeout does not cancel the pump itself.
    if wait_seconds > 0:
        try:
            await asyncio.wait_for(asyncio.shield(job.pump_task), timeout=wait_seconds)
        except (TimeoutError, asyncio.TimeoutError):
            pass

    if job.proc.returncode is not None:
        del _jobs[job.id]
        parts = [f"exit code: {job.proc.returncode}"]
        if job.output or job.discarded:
            parts.append("--- output ---")
            if job.discarded:
                parts.append(f"... [first {job.discarded} characters dropped; only the tail is kept]")
            parts.append(job.output)
        return _truncate("\n".join(parts))

    shown = job.output[:_MAX_OUTPUT_CHARS]
    parts = [
        f"Still running after {wait_seconds:g} seconds; continuing as background job {job.id} (pid {proc.pid}).",
        f"Poll new output with check_command(job_id={job.id}, offset={job.discarded + len(shown)})"
        f" (use `wait` between polls); stop it with kill_command.",
    ]
    if shown:
        parts.append("--- output so far ---")
        parts.append(shown)
        if len(job.output) > len(shown):
            parts.append(f"... [{len(job.output) - len(shown)} more characters already produced]")
    return "\n".join(parts)


@function_tool
async def check_command(job_id: int, offset: int = 0) -> str:
    """Check on a background command: its status and its output so far.

    Output offsets are absolute (character positions since the job started);
    pass the offset from the previous check's continuation hint to read only
    what is new. Note that a running job may produce more output after this
    returns — poll again (using `wait` between polls) until it exits.

    Args:
        job_id: A job id reported by `run_command` when the command kept running.
        offset: Character offset into the job's output to continue reading from.
    """

    job = _jobs.get(job_id)
    if job is None:
        return f"Error: no background job with id {job_id}."

    if job.proc.returncode is None:
        status = "running"
    else:
        status = f"exited with code {job.proc.returncode}"
    parts = [f"job {job.id} ({status}): {job.command}"]

    total = job.discarded + len(job.output)
    offset = max(offset, 0)
    if offset >= total:
        parts.append(f"no output at offset {offset} (the job has produced {total} characters).")
        return "\n".join(parts)
    if offset < job.discarded:
        parts.append(f"... [first {job.discarded} characters no longer buffered]")
        offset = job.discarded

    window = job.output[offset - job.discarded : offset - job.discarded + _MAX_OUTPUT_CHARS]
    parts.append("--- output ---")
    parts.append(window)
    remaining = total - (offset + len(window))
    if remaining > 0:
        parts.append(f"... [{remaining} more characters; continue with offset={offset + len(window)}]")

    return "\n".join(parts)


@function_tool
async def kill_command(job_id: int) -> str:
    """Kill a background command that is no longer needed.

    Args:
        job_id: A job id reported by `run_command` when the command kept running.
    """

    job = _jobs.get(job_id)
    if job is None:
        return f"Error: no background job with id {job_id}."
    if job.proc.returncode is not None:
        return f"Job {job_id} already exited with code {job.proc.returncode}."

    await _kill_tree(job.proc)
    if job.pump_task is not None:
        try:
            await asyncio.wait_for(job.pump_task, timeout=5)
        except (TimeoutError, asyncio.TimeoutError):
            pass  # an orphan holding the pipe open; the buffer is still readable

    return f"Killed job {job_id}; its output remains available via check_command."


@function_tool
def list_dir(path: str = ".") -> str:
    """List the entries of a directory (directories first, with file sizes).

    Args:
        path: Directory path, absolute or relative to the working directory.
    """

    p = _resolve(path)
    if not p.is_dir():
        return f"Error: {p} is not a directory."

    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except OSError as e:
        return f"Error listing {p}: {e}"

    lines = [f"{p}:"]
    for e in entries:
        if e.is_dir():
            lines.append(f"  {e.name}/")
        else:
            try:
                size = e.stat().st_size
            except OSError:
                size = 0
            lines.append(f"  {e.name}  ({size} bytes)")

    return _truncate("\n".join(lines))


@function_tool
def read_file(path: str, offset: int = 1, limit: int = _MAX_READ_LINES) -> str:
    """Read a text file, returning its contents with 1-based line numbers.

    Args:
        path: File path, absolute or relative to the working directory.
        offset: Line number to start reading from (1-based).
        limit: Maximum number of lines to return.
    """

    p = _resolve(path)
    if not p.is_file():
        return f"Error: {p} is not a file."

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"Error reading {p}: {e}"

    offset = max(offset, 1)
    window = lines[offset - 1 : offset - 1 + max(limit, 1)]
    if not window:
        return f"{p} has {len(lines)} lines; nothing at offset {offset}."
    numbered = "\n".join(f"{offset + i:6d} | {line[:_MAX_LINE_CHARS]}" for i, line in enumerate(window))
    remaining = len(lines) - (offset - 1 + len(window))
    if remaining > 0:
        numbered += f"\n... [{remaining} more lines; continue with offset={offset + len(window)}]"

    return _truncate(numbered)


@function_tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content.

    Parent directories are created as needed. Prefer `edit_file` for small
    changes to existing files.

    Args:
        path: File path, absolute or relative to the working directory.
        content: Full new contents of the file.
    """

    p = _resolve(path)

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
    except OSError as e:
        return f"Error writing {p}: {e}"

    return f"Wrote {len(content)} characters to {p}."


@function_tool
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file.

    `old_string` must match the file contents exactly (including whitespace),
    and must be unique in the file unless `replace_all` is set. Read the file
    first to get the exact text.

    Args:
        path: File path, absolute or relative to the working directory.
        old_string: Exact text to find.
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of requiring uniqueness.
    """

    p = _resolve(path)
    if not p.is_file():
        return f"Error: {p} is not a file."

    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading {p}: {e}"

    count = text.count(old_string)
    if count == 0:
        return "Error: old_string not found in the file. Read the file and match its contents exactly."
    if count > 1 and not replace_all:
        return f"Error: old_string occurs {count} times; add more surrounding context to make it unique, or set replace_all=true."

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    try:
        p.write_text(new_text, encoding="utf-8", newline="\n")
    except OSError as e:
        return f"Error writing {p}: {e}"

    return f"Replaced {count if replace_all else 1} occurrence(s) in {p}."


@function_tool
async def glob(pattern: str, path: str = ".") -> str:
    """List files matching a glob pattern (gitignore-style semantics).

    Respects .gitignore and skips hidden files. A bare pattern like `*.py`
    matches at any directory depth; patterns containing `/` (e.g.
    `src/**/*.rs`) are anchored to the search root.

    Args:
        pattern: Glob pattern, e.g. `*.py` or `src/**/test_*.rs`.
        path: Directory to search in, absolute or relative to the working directory.
    """

    exe = _rg_executable()
    if exe is None:
        return "Error: ripgrep not found (pip install ripgrep, or put `rg` on PATH)."

    root = _resolve(path)
    if not root.is_dir():
        return f"Error: {root} is not a directory."

    cmd = [exe, "--files", "--glob", pattern, "--", str(root)]
    try:
        returncode, stdout, stderr = await _run_capture(cmd, 60)
    except (TimeoutError, OSError) as e:
        return f"Error: ripgrep failed to run: {e}"

    if not stdout:
        if returncode in (0, 1):  # rg --files exits 1 when nothing was listed
            return f"No files match {pattern!r} in {root}."
        return f"Error: ripgrep failed: {stderr.strip() or f'exit code {returncode}'}"

    matches = sorted(stdout.splitlines())
    shown = matches[:_MAX_GLOB_RESULTS]
    out = "\n".join(shown)
    if len(matches) > len(shown):
        out += f"\n... [{len(matches) - len(shown)} more matches; narrow the pattern]"

    return _truncate(out)


@function_tool
async def grep(pattern: str, path: str = ".", glob: str = "", context: int = 0, ignore_case: bool = False) -> str:
    """Search file contents with ripgrep, like grep -rn.

    Respects .gitignore and skips hidden and binary files. Output lines are
    `path:line: text` (context lines use `path-line- text`).

    Args:
        pattern: Regular expression (Rust regex syntax) to search for.
        path: Directory (or single file) to search, absolute or relative.
        glob: Optional file filter, e.g. `*.py` or `src/**/*.rs`.
        context: Lines of context to show around each match.
        ignore_case: Match case-insensitively.
    """

    exe = _rg_executable()
    if exe is None:
        return "Error: ripgrep not found (pip install ripgrep, or put `rg` on PATH)."

    root = _resolve(path)
    if not root.exists():
        return f"Error: {root} does not exist."

    cmd = [exe, "--line-number", "--no-heading", "--color", "never", "--max-columns", str(_MAX_LINE_CHARS)]
    if glob:
        cmd += ["--glob", glob]
    if context > 0:
        cmd += ["--context", str(context)]
    if ignore_case:
        cmd.append("--ignore-case")
    cmd += ["--regexp", pattern, "--", str(root)]
    try:
        returncode, stdout, stderr = await _run_capture(cmd, 60)
    except (TimeoutError, OSError) as e:
        return f"Error: ripgrep failed to run: {e}"

    if not stdout:
        if returncode == 1:  # rg: 0 = matches, 1 = no matches, 2 = error
            return f"No matches for {pattern!r} under {root}."
        return f"Error: ripgrep failed: {stderr.strip() or f'exit code {returncode}'}"

    lines = stdout.splitlines()
    if len(lines) > _MAX_MATCHES:
        lines = lines[:_MAX_MATCHES] + [f"... [stopped at {_MAX_MATCHES} lines; narrow the search]"]

    return _truncate("\n".join(lines))


class _HtmlText(HTMLParser):
    """Extracts the visible text of an HTML document."""

    _SKIP_TAGS = {"script", "style", "noscript", "template"}
    _BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
        "header", "footer", "nav", "pre", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }  # fmt: skip

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP_TAGS:
            self._skip += 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def text(self) -> str:
        collapsed = re.sub(r"[ \t]+", " ", "".join(self._parts))
        return re.sub(r"\n\s*\n+", "\n\n", collapsed).strip()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _HtmlText()

    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup; fall back to a crude tag strip
        return re.sub(r"<[^>]+>", " ", html)

    return parser.text()


@function_tool
async def web_fetch(url: str, offset: int = 0) -> str:
    """Fetch a web page and return its visible text (HTML tags stripped).

    Args:
        url: The http(s) URL to fetch.
        offset: Character offset to continue reading a long page from.
    """

    if not url.startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported."

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
    except httpx.HTTPError as e:
        return f"Error fetching {url}: {e}"

    if resp.status_code >= 400:
        return f"Error: HTTP {resp.status_code} for {url}."

    if len(resp.content) > _MAX_FETCH_BYTES:
        return f"Error: response too large ({len(resp.content)} bytes)."

    ctype = resp.headers.get("content-type", "")
    text = _html_to_text(resp.text) if "html" in ctype else resp.text
    offset = max(offset, 0)
    window = text[offset : offset + _MAX_FETCH_CHARS]
    if not window:
        return f"{url} has {len(text)} characters of text; nothing at offset {offset}."

    remaining = len(text) - (offset + len(window))
    if remaining > 0:
        window += f"\n... [{remaining} more characters; continue with" f" offset={offset + len(window)}]"

    return window


# Hosted, executed by DeepSeek rather than by this process. `search_context_size`
# and `user_location` are part of the OpenAI tool schema but DeepSeek ignores
# them, so they are left at their defaults.
web_search = WebSearchTool()


ALL_TOOLS: list[Tool] = [
    wait,
    run_command,
    check_command,
    kill_command,
    list_dir,
    read_file,
    write_file,
    edit_file,
    glob,
    grep,
    web_fetch,
    web_search,
]
