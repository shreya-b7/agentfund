"""Run ``agentfund daily`` automatically every weekday on a Mac (launchd).

``launchd`` is macOS's built-in scheduler. The job runs as you, from the project
folder, with the project's own Python. If the Mac is asleep at the scheduled
time, macOS runs the job when it wakes (and ``daily`` waits if the market is open
by then). Output goes to ``runs/paper/daily.log``.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

LABEL = "com.agentfund.daily"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
PROTECTED = ("Downloads", "Desktop", "Documents")  # macOS blocks background access to these


def default_local_time() -> tuple[int, int]:
    """5:00pm New York time, expressed in this computer's local time zone."""
    ny = datetime.combine(datetime.now(ZoneInfo("America/New_York")).date(), time(17, 0), ZoneInfo("America/New_York"))
    local = ny.astimezone()
    return local.hour, local.minute


def protected_folder(project: Path) -> str | None:
    home = Path.home().resolve()
    for name in PROTECTED:
        try:
            project.resolve().relative_to(home / name)
            return name
        except ValueError:
            continue
    return None


def build_plist(project: Path, python: str, hour: int, minute: int) -> dict:
    log = str(project / "runs" / "paper" / "daily.log")
    return {
        "Label": LABEL,
        "ProgramArguments": [python, "-m", "agentfund", "daily"],
        "WorkingDirectory": str(project),
        "StartCalendarInterval": [{"Weekday": d, "Hour": hour, "Minute": minute} for d in range(1, 6)],
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONUNBUFFERED": "1"},
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def install(project: Path, hour: int | None = None, minute: int | None = None) -> str:
    if sys.platform != "darwin":
        return ("Automatic scheduling is set up for macOS. On Linux, add this line with 'crontab -e':\n"
                f"30 17 * * 1-5 cd {project} && {sys.executable} -m agentfund daily >> runs/paper/daily.log 2>&1")
    bad = protected_folder(project)
    if bad:
        raise RuntimeError(
            f"The project is inside your {bad} folder, which macOS doesn't let background jobs read.\n"
            f"Move it to your home folder first (see the README section 'Running it automatically')."
        )
    if hour is None or minute is None:
        hour, minute = default_local_time()
    (project / "runs" / "paper").mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    _launchctl("bootout", f"gui/{uid}/{LABEL}")  # replace any earlier version; fine if none
    with PLIST.open("wb") as fh:
        plistlib.dump(build_plist(project, sys.executable, hour, minute), fh)
    res = _launchctl("bootstrap", f"gui/{uid}", str(PLIST))
    if res.returncode != 0:
        res = _launchctl("load", str(PLIST))  # older macOS
        if res.returncode != 0:
            raise RuntimeError(f"launchctl refused the job: {res.stderr.strip()}")
    return f"Scheduled: every weekday at {hour:02d}:{minute:02d} (your local time). Log: runs/paper/daily.log"


def remove() -> str:
    if sys.platform != "darwin":
        return "Remove the agentfund line with 'crontab -e'."
    _launchctl("bootout", f"gui/{os.getuid()}/{LABEL}")
    if PLIST.exists():
        PLIST.unlink()
        return "Automatic daily runs removed."
    return "No schedule was installed."


def status(project: Path) -> str:
    lines = []
    if sys.platform == "darwin" and PLIST.exists():
        cfg = plistlib.loads(PLIST.read_bytes())
        when = cfg["StartCalendarInterval"][0]
        lines.append(f"Scheduled: weekdays at {when['Hour']:02d}:{when['Minute']:02d} local time, from {cfg['WorkingDirectory']}")
    else:
        lines.append("Not scheduled. Run 'python -m agentfund schedule install'.")
    log = project / "runs" / "paper" / "daily.log"
    if log.exists():
        tail = log.read_text().strip().splitlines()[-8:]
        lines += ["", "Last log lines:", *tail]
    return "\n".join(lines)
