from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
import sys
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
    sumo_binary_sha256: str
    netconvert_binary_sha256: str
    traci_module_path: str
    traci_version: str

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _traci_version(tools: Path) -> tuple[str, str]:
    module_path = (tools / "traci" / "__init__.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"SUMO TraCI package does not exist: {module_path}")
    constants_path = tools / "traci" / "constants.py"
    version = "unknown"
    if constants_path.exists():
        text = constants_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^TRACI_VERSION\s*=\s*(\d+)", text, flags=re.MULTILINE)
        if match:
            version = f"protocol_{match.group(1)}"
    return str(module_path), version


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
        if configured.is_absolute():
            if not configured.exists():
                raise FileNotFoundError(
                    f"Configured {name} binary does not exist: {configured}"
                )
            if not configured.is_file():
                raise FileNotFoundError(
                    f"Configured {name} binary is not a file: {configured}"
                )
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
    executable_suffix = ".exe" if os.name == "nt" else ""
    sumo_gui = (home / "bin" / f"sumo-gui{executable_suffix}").resolve()
    netconvert = (home / "bin" / f"netconvert{executable_suffix}").resolve()
    if not netconvert.exists():
        raise FileNotFoundError(f"netconvert was not found next to SUMO installation: {home}")
    tools = home / "tools"
    if not tools.is_dir():
        raise FileNotFoundError(f"SUMO tools directory does not exist: {tools}")
    traci_module_path, traci_version = _traci_version(tools)
    return SumoInstallation(
        sumo_binary=str(sumo_binary),
        sumo_gui_binary=str(sumo_gui) if sumo_gui.exists() else "",
        netconvert_binary=str(netconvert),
        tools_directory=str(tools.resolve()),
        sumo_home=str(home.resolve()),
        sumo_version=_version(sumo_binary),
        netconvert_version=_version(netconvert),
        sumo_binary_sha256=_sha256(sumo_binary),
        netconvert_binary_sha256=_sha256(netconvert),
        traci_module_path=traci_module_path,
        traci_version=traci_version,
    )


def sumo_installation_from_config(scenario_cfg: Any) -> SumoInstallation:
    fingerprint = dict(scenario_cfg.get("sumo_installation_fingerprint", {}) or {})
    if fingerprint:
        return SumoInstallation(**{field: str(fingerprint.get(field, "")) for field in SumoInstallation.__dataclass_fields__})
    return resolve_sumo_installation(scenario_cfg)


def configure_sumo_python(installation: SumoInstallation) -> None:
    tools = str(Path(installation.tools_directory).resolve())
    expected_module = Path(installation.traci_module_path).resolve()
    loaded = sys.modules.get("traci")
    if loaded is not None:
        loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if loaded_path != expected_module:
            raise RuntimeError(
                "TraCI was imported from a different SUMO installation: "
                f"loaded={loaded_path}, expected={expected_module}"
            )
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).resolve()) != tools]
    sys.path.insert(0, tools)
    importlib.invalidate_caches()


def sumo_subprocess_environment(installation: SumoInstallation) -> dict[str, str]:
    env = dict(os.environ)
    env["SUMO_HOME"] = installation.sumo_home
    current_path = env.get("PATH", "")
    bin_directory = str(Path(installation.sumo_binary).parent)
    if bin_directory.lower() not in current_path.lower():
        env["PATH"] = f"{bin_directory}{os.pathsep}{current_path}"
    current_python_path = env.get("PYTHONPATH", "")
    tools = installation.tools_directory
    env["PYTHONPATH"] = (
        f"{tools}{os.pathsep}{current_python_path}" if current_python_path else tools
    )
    return env
