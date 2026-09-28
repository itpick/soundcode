"""In-memory model of a .sc document.

Deliberately close to the file's own shape: the parser's job is to lex and
group, not to interpret. Stream namespaces define what their statements mean
(§4.6 "two-level by intent"), so anything a stream schema doesn't recognise is
kept verbatim in `unknown` and re-emitted — that is what makes adding stream
types later non-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeSpec:
    """Either a bar:beat position or absolute seconds. Resolved by the Grid."""

    bar: int | None = None
    beat: float | None = None
    seconds: float | None = None

    @property
    def is_absolute(self) -> bool:
        return self.seconds is not None


@dataclass
class Event:
    time: TimeSpec
    atom: str | None = None          # pitch, percussion voice, or "." (chord root)
    dur: str | None = None           # raw "0.5b" / "0.234s"
    vel: int | None = None
    text: str | None = None          # :text streams
    blocks: dict[str, str] = field(default_factory=dict)   # env{...}, vib{...}
    dev_ms: float = 0.0              # micro-timing deviation — the groove
    conf: float | None = None        # ?0.NN
    alt: str | None = None
    raw: str = ""


@dataclass
class Binding:
    """`bars 5-11 basic ~0.96` or `bars 13-16 like 1-4 vel -8`."""

    first: int
    last: int
    pattern: str | None = None
    like: tuple[int, int] | None = None
    repeat: int = 1
    transpose: int = 0
    vel_delta: int = 0
    similarity: float | None = None   # None == exact binding
    chords: list[str] | None = None   # :harmony only
    raw: str = ""


@dataclass
class Stream:
    name: str                                  # "notes.vox"
    fields: dict[str, str] = field(default_factory=dict)
    blocks: dict[str, str] = field(default_factory=dict)
    gloss: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    warns: list[str] = field(default_factory=list)
    patterns: dict[str, list[Event]] = field(default_factory=dict)
    bindings: list[Binding] = field(default_factory=list)
    statements: list[tuple[str, list[str]]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    autos: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.name.split(".", 1)[0]

    @property
    def track(self) -> str | None:
        return self.name.split(".", 1)[1] if "." in self.name else None


@dataclass
class Document:
    pragmas: dict[str, str] = field(default_factory=dict)
    header: dict[str, str] = field(default_factory=dict)
    streams: list[Stream] = field(default_factory=list)
    path: Any = None                 # where it was read from (parse_file), for relative refs

    def stream(self, name: str) -> Stream | None:
        for s in self.streams:
            if s.name == name:
                return s
        return None

    def of_kind(self, kind: str) -> list[Stream]:
        return [s for s in self.streams if s.kind == kind]

    @property
    def duration(self) -> float:
        return float(self.header.get("duration", 0.0) or 0.0)

    @property
    def sample_rate(self) -> int:
        return int(float(self.header.get("sr", 44100)))
