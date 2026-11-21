"""Stage abstraction: every pipeline step is a class with a ``name`` and ``run(cfg)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pnd.config import Config


@runtime_checkable
class Stage(Protocol):
    """A runnable pipeline stage. Implementations call ``set_global_seed(cfg.seed)`` first."""

    name: ClassVar[str]

    def run(self, cfg: Config) -> None: ...
