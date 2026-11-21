"""Interactive I/O port so that the manual labelers are testable."""

from __future__ import annotations

from typing import Protocol


class InputPort(Protocol):
    """What the labelers need from a terminal."""

    def ask(self, prompt: str) -> str: ...

    def show(self, *values: object) -> None: ...


class StdInputPort:
    """Real terminal: ``input`` / ``print`` (R7: byte-identical prompts)."""

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def show(self, *values: object) -> None:
        print(*values)


class ScriptedInputPort:
    """Test double: answers come from a list, everything shown/asked is recorded."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.shown: list[tuple[object, ...]] = []
        self.prompts: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._answers:
            raise EOFError("ScriptedInputPort ran out of answers")
        return self._answers.pop(0)

    def show(self, *values: object) -> None:
        self.shown.append(values)
