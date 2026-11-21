"""Configuration loading.

Every path, constant and hyper-parameter of the original scripts lives in ``configs/default.yaml``.
Values under ``paths`` are resolved against the project root (the directory holding ``pyproject.toml``).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) to the first directory containing ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {current}")


class Section:
    """Frozen, attribute-access view over one mapping of the YAML file."""

    __slots__ = ("_data", "_name")

    def __init__(self, data: Mapping[str, Any], name: str) -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_name", name)

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"Config section '{self._name}' has no key '{key}'")
        return data[key]

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Config is frozen")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._data.items():
            out[k] = v.to_dict() if isinstance(v, Section) else v
        return out

    def __repr__(self) -> str:
        return f"Section({self._name}, {self.to_dict()!r})"


def _build(data: Mapping[str, Any], name: str, root: Path, is_path: bool) -> Section:
    built: dict[str, Any] = {}
    for key, value in data.items():
        child_name = f"{name}.{key}" if name else key
        child_is_path = is_path or child_name == "paths"
        if isinstance(value, Mapping):
            built[key] = _build(value, child_name, root, child_is_path)
        elif child_is_path and isinstance(value, str):
            built[key] = (root / value).resolve() if not Path(value).is_absolute() else Path(value)
        else:
            built[key] = value
    return Section(built, name)


@dataclass(frozen=True)
class Config:
    """Typed root of the configuration tree."""

    root: Path
    seed: int | None
    paths: Section
    keyword_filter: Section
    message_features: Section
    classifier: Section
    sessions: Section
    labeling: Section
    klines: Section
    negative_samples: Section
    features: Section
    dataset: Section
    model: Section
    coingecko_fetch: Section
    telegram_fetch: Section
    raw: Mapping[str, Any]


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must look like dotted.key=value, got {override!r}")
    dotted, raw_value = override.split("=", 1)
    keys = dotted.strip().split(".")
    node: Any = data
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node = node[key]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(f"Unknown config key: {dotted}")
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: Path | None = None, overrides: Iterable[str] = (), root: Path | None = None) -> Config:
    """Load the YAML config, apply ``dotted.key=value`` overrides and resolve paths."""
    project_root = root or find_project_root()
    config_path = path or (project_root / DEFAULT_CONFIG_RELATIVE)
    with open(config_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    data = copy.deepcopy(data)
    for override in overrides:
        _apply_override(data, override)
    tree = _build(data, "", project_root, False)
    return Config(
        root=project_root,
        seed=data.get("seed"),
        paths=tree.paths,
        keyword_filter=tree.keyword_filter,
        message_features=tree.message_features,
        classifier=tree.classifier,
        sessions=tree.sessions,
        labeling=tree.labeling,
        klines=tree.klines,
        negative_samples=tree.negative_samples,
        features=tree.features,
        dataset=tree.dataset,
        model=tree.model,
        coingecko_fetch=tree.coingecko_fetch,
        telegram_fetch=tree.telegram_fetch,
        raw=data,
    )
