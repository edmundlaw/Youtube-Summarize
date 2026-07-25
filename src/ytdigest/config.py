"""Configuration and secret loading.

config/config.toml holds everything checkable into git; .env holds secrets and
is chmod 600. Nothing here reaches out to the network or touches the DB.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader. Does not overwrite variables already in the env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def _path(self, section: str, key: str, default: str) -> Path:
        value = Path(self.get(section, key, default))
        return value if value.is_absolute() else self.root / value

    @property
    def data_dir(self) -> Path:
        return self._path("paths", "data_dir", "data")

    @property
    def out_dir(self) -> Path:
        return self._path("paths", "out_dir", "out")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / ".lock"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "ytdigest.jsonl"

    def secret(self, name: str) -> str | None:
        return os.environ.get(name) or None

    def require_secret(self, name: str) -> str:
        value = self.secret(name)
        if not value:
            raise RuntimeError(
                f"{name} is not set. Add it to {self.root / '.env'} (chmod 600)."
            )
        return value

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.out_dir,
            self.data_dir / "audio",
            self.data_dir / "transcripts",
            self.data_dir / "normalized",
            self.data_dir / "logs",
        ):
            directory.mkdir(parents=True, exist_ok=True)


@cache
def load_config(root: Path | None = None) -> Config:
    root = root or REPO_ROOT
    _load_dotenv(root / ".env")
    config_file = root / "config" / "config.toml"
    raw = tomllib.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    return Config(raw=raw, root=root)


@cache
def load_glossary(root: Path | None = None) -> list[str]:
    """Protected terms that must never be translated out of English."""
    import yaml

    root = root or REPO_ROOT
    path = root / "config" / "glossary.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    terms: list[str] = []
    for group in data.values():
        if isinstance(group, list):
            terms.extend(str(t) for t in group)
    return terms
