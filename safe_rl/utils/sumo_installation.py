from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SumoInstallation:
    sumo_binary: str
    sumo_gui_binary: str
    netconvert_binary: str
    tools_directory: str
    sumo_home: str
    sumo_version: str
    netconvert_version: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def major_minor_version(self) -> str:
        tokens = self.sumo_version.replace(",", " ").split()
        for token in tokens:
            if token and token[0].isdigit():
                parts = token.strip("vV").split(".")
                return ".".join(parts[:2])
        return self.sumo_version


def _version(binary: Path) -> str:
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (completed.stdout or completed.stderr).splitlines()[0].strip()
    except Exception:
        return "unknown"


def _windows_candidate_homes() -> list[Path]:
    if os.name != "nt":
        return []
    return [
        Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
        Path(r"C:\Program Files\Eclipse\Sumo"),
    ]


def _resolve_binary(value: str | Path | None, name: str) -> Path | None:
    if value:
        configured = Path(str(value))
        if configured.is_absolute() and configured.exists():
            return configured.resolve()
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        executable = Path(sumo_home) / "bin" / (f"{name}.exe" if os.name == "nt" else name)
        if executable.exists():
            return executable.resolve()
    found = shutil.which(str(value or name))
    if found:
        return Path(found).resolve()
    for home in _windows_candidate_homes():
        executable = home / "bin" / (f"{name}.exe" if os.name == "nt" else name)
        if executable.exists():
            return executable.resolve()
    return None


def resolve_sumo_installation(scenario_cfg: Any) -> SumoInstallation:
    sumo_binary = _resolve_binary(scenario_cfg.get("sumo_binary", "sumo"), "sumo")
    if sumo_binary is None:
        raise FileNotFoundError(
            "SUMO was not found. Set scenario.sumo_binary to an absolute path, "
            "set SUMO_HOME, or add SUMO/bin to PATH."
        )
    home = sumo_binary.parent.parent
    sumo_gui = _resolve_binary(home / "bin" / ("sumo-gui.exe" if os.name == "nt" else "sumo-gui"), "sumo-gui")
    netconvert = _resolve_binary(home / "bin" / ("netconvert.exe" if os.name == "nt" else "netconvert"), "netconvert")
    if netconvert is None:
        raise FileNotFoundError(f"netconvert was not found next to SUMO installation: {home}")
    tools = home / "tools"
    if not tools.is_dir():
        raise FileNotFoundError(f"SUMO tools directory does not exist: {tools}")
    return SumoInstallation(
        sumo_binary=str(sumo_binary),
        sumo_gui_binary=str(sumo_gui or ""),
        netconvert_binary=str(netconvert),
        tools_directory=str(tools.resolve()),
        sumo_home=str(home.resolve()),
        sumo_version=_version(sumo_binary),
        netconvert_version=_version(netconvert),
    )


def sumo_subprocess_environment(installation: SumoInstallation) -> dict[str, str]:
    env = dict(os.environ)
    env["SUMO_HOME"] = installation.sumo_home
    current_path = env.get("PATH", "")
    bin_directory = str(Path(installation.sumo_binary).parent)
    if bin_directory.lower() not in current_path.lower():
        env["PATH"] = f"{bin_directory}{os.pathsep}{current_path}"
    return env
