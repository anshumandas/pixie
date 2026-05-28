"""Natural-language command parsing — the seam for the optional LLM bar.

Today a lightweight rule-based parser turns phrases like
"remove the background and make it darker" into a list of structured
:class:`Command` objects the engine can execute. An LLM parser (Ollama or a
hosted model) can be dropped in later behind the same :class:`CommandParser`
interface — see :class:`LLMCommandParser` for the stub and wiring notes.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Command:
    action: str                       # 'remove_background' | 'erase' | 'stylize' | 'adjust'
    args: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.args:
            kv = ", ".join(f"{k}={v}" for k, v in self.args.items())
            return f"{self.action}({kv})"
        return f"{self.action}()"


class CommandParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> List[Command]:
        ...


class RuleBasedParser(CommandParser):
    """Keyword/regex parser — works offline, no model required."""

    _STYLES = {
        "ink": "Ink",
        "shadow": "Shadow",
        "silhouette": "Shadow",
        "pencil": "Pencil",
        "sketch": "Pencil",
        "oil": "Oil / paint",
        "paint": "Oil / paint",
    }

    def parse(self, text: str) -> List[Command]:
        t = text.lower().strip()
        cmds: List[Command] = []

        if re.search(r"\b(remove|delete|cut out)\b.*\bback ?ground\b", t) or "remove bg" in t:
            cmds.append(Command("remove_background"))

        if re.search(r"\b(erase|remove|delete)\b", t) and "background" not in t:
            # only treat as object-erase if not the background phrasing above
            if not cmds or cmds[-1].action != "remove_background":
                cmds.append(Command("erase"))

        for kw, style in self._STYLES.items():
            if re.search(rf"\b{kw}\b", t):
                cmds.append(Command("stylize", {"style": style}))
                break

        # brightness / contrast
        b = 0.0
        c = 0.0
        if re.search(r"\b(darker|darken|dim)\b", t):
            b -= 0.3
        if re.search(r"\b(brighter|brighten|lighter)\b", t):
            b += 0.3
        if re.search(r"\b(more contrast|higher contrast|punchy|contrasty)\b", t):
            c += 0.3
        if re.search(r"\b(less contrast|flatter|lower contrast)\b", t):
            c -= 0.3
        if b or c:
            cmds.append(Command("adjust", {"brightness": round(b, 2), "contrast": round(c, 2)}))

        return cmds


class LLMCommandParser(CommandParser):
    """Roadmap: parse with a local Ollama model (or hosted LLM).

    Wiring (left intentionally inert so the app runs offline):
      1. `pip install ollama` and `ollama pull llama3.2:3b`
      2. Prompt the model to emit JSON: a list of {action, args} objects whose
         actions match RuleBasedParser's vocabulary.
      3. Parse the JSON into Command objects; on any failure, fall back to
         RuleBasedParser so the bar always does *something* sensible.
    """

    def __init__(self, model: str = "llama3.2:3b", fallback: CommandParser | None = None):
        self.model = model
        self.fallback = fallback or RuleBasedParser()

    def available(self) -> bool:
        try:
            import ollama  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False

    def parse(self, text: str) -> List[Command]:
        if not self.available():
            return self.fallback.parse(text)
        # TODO: call ollama.chat(...) with a JSON-emitting system prompt and
        # decode into Command objects. Until then, defer to the rule-based parser.
        return self.fallback.parse(text)


def default_parser() -> CommandParser:
    """Return the best parser available right now."""
    llm = LLMCommandParser()
    return llm if llm.available() else RuleBasedParser()
