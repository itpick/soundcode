"""Parse .sc text into a Document.

Two passes. First `logical_lines` folds the physical file into logical lines,
resolving the three things that can span lines: unbalanced braces (`%pat {`,
`timbre{`), unbalanced quotes (`@style "..."`), and indented continuations of a
stream declaration. Second pass classifies each logical line against the
current stream's schema.
"""

from __future__ import annotations

import re

from .model import Binding, Document, Event, Stream, TimeSpec

_DUR_RE = re.compile(r"^[\d.]+[bs]$")
_VEL_RE = re.compile(r"^\d{1,3}$")
_DEV_RE = re.compile(r"^dev([+-][\d.]+)ms$")
_CONF_RE = re.compile(r"^\?([\d.]+)$")
_BLOCK_RE = re.compile(r"^([A-Za-z_][\w]*)\{(.*)\}$", re.DOTALL)
_RANGE_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


class ParseError(ValueError):
    def __init__(self, line_no: int, text: str, msg: str) -> None:
        super().__init__(f"line {line_no}: {msg}\n  {text.strip()}")
        self.line_no = line_no


def _balanced(text: str) -> bool:
    """True when braces and double quotes are both closed in `text`."""
    depth, in_str = 0, False
    for ch in text:
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depth <= 0 and not in_str


def logical_lines(text: str) -> list[tuple[int, str, bool]]:
    """Fold physical lines into (line_no, text, was_indented) logical lines."""
    out: list[tuple[int, str, bool]] = []
    buf, buf_no, buf_indent = "", 0, False

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not buf:
            # `#` is a comment only as the first non-whitespace character
            if not stripped or stripped.startswith("#"):
                continue
            buf, buf_no, buf_indent = stripped, n, raw[:1].isspace()
        else:
            buf += " " + stripped
        if _balanced(buf):
            out.append((buf_no, buf, buf_indent))
            buf = ""
    if buf:
        out.append((buf_no, buf, buf_indent))
    return out


def split_tokens(text: str) -> list[str]:
    """Whitespace split that keeps {...} blocks and "..." strings intact."""
    toks, cur, depth, in_str = [], "", 0, False
    for ch in text:
        if ch == '"':
            in_str = not in_str
            cur += ch
        elif ch == "{" and not in_str:
            depth += 1
            cur += ch
        elif ch == "}" and not in_str:
            depth -= 1
            cur += ch
        elif ch.isspace() and depth == 0 and not in_str:
            if cur:
                toks.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        toks.append(cur)
    return toks


def split_events(text: str) -> list[str]:
    """Split on `|`, which separates events and nothing else."""
    parts, cur, depth, in_str = [], "", 0, False
    for ch in text:
        if ch == '"':
            in_str = not in_str
        elif ch == "{" and not in_str:
            depth += 1
        elif ch == "}" and not in_str:
            depth -= 1
        if ch == "|" and depth == 0 and not in_str:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_time(tok: str) -> TimeSpec:
    if tok.startswith("@"):
        return TimeSpec(seconds=float(tok[1:]))
    if ":" in tok:
        bar, beat = tok.split(":", 1)
        return TimeSpec(bar=int(bar), beat=float(beat))
    return TimeSpec(bar=None, beat=float(tok))     # bare beat, inside a %pat


def parse_event(text: str) -> Event:
    toks = split_tokens(text)
    if not toks:
        raise ValueError("empty event")
    ev = Event(time=parse_time(toks[0]), raw=text)

    rest = toks[1:]
    # positional head: text | atom [dur] [vel]
    if rest and rest[0].startswith('"'):
        ev.text = rest[0].strip('"')
        rest = rest[1:]
        if rest and _DUR_RE.match(rest[0]):
            ev.dur = rest[0]
            rest = rest[1:]
    elif rest and not _BLOCK_RE.match(rest[0]) and not rest[0].startswith("?"):
        ev.atom = rest[0]
        rest = rest[1:]
        if rest and _DUR_RE.match(rest[0]):
            ev.dur = rest[0]
            rest = rest[1:]
        if rest and _VEL_RE.match(rest[0]):
            ev.vel = int(rest[0])
            rest = rest[1:]

    for tok in rest:
        if m := _DEV_RE.match(tok):
            ev.dev_ms = float(m.group(1))
        elif m := _CONF_RE.match(tok):
            ev.conf = float(m.group(1))
        elif tok.startswith("alt="):
            ev.alt = tok[4:].strip('"')
        elif m := _BLOCK_RE.match(tok):
            ev.blocks[m.group(1)] = m.group(2).strip()
        elif _VEL_RE.match(tok) and ev.vel is None:
            ev.vel = int(tok)
    return ev


def _parse_binding(toks: list[str], raw: str) -> Binding:
    m = _RANGE_RE.match(toks[1])
    if not m:
        raise ValueError(f"bad bar range: {toks[1]!r}")
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) else first
    b = Binding(first=first, last=last, raw=raw)

    rest = toks[2:]
    # trailing ~0.NN similarity marker
    if rest and rest[-1].startswith("~"):
        b.similarity = float(rest[-1][1:])
        rest = rest[:-1]

    if rest and rest[0] == "like":
        m2 = _RANGE_RE.match(rest[1])
        b.like = (int(m2.group(1)), int(m2.group(2) or m2.group(1)))
        rest = rest[2:]
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok.startswith("x") and tok[1:].isdigit():
                b.repeat = int(tok[1:])
            elif tok == "transpose":
                i += 1
                b.transpose = int(rest[i])
            elif tok == "vel":
                i += 1
                b.vel_delta = int(rest[i])
            i += 1
    elif rest and "|" in raw:
        b.chords = [c.strip() for c in raw.split(None, 2)[2].split("|")]
    elif rest:
        b.pattern = rest[0]
    return b


def parse(text: str) -> Document:
    doc = Document()
    cur: Stream | None = None
    last_was_decl = False

    for n, line, indented in logical_lines(text):
        try:
            # ---- pragmas and header ----------------------------------------
            if line.startswith("%") and not line.startswith("%pat"):
                key, _, val = line[1:].partition(" ")
                doc.pragmas[key] = val.strip()
                last_was_decl = False
                continue
            if line.startswith("@") and cur is None:
                key, _, val = line[1:].partition(" ")
                doc.header[key] = val.strip().strip('"')
                continue

            # ---- stream declaration ----------------------------------------
            if line.startswith(":"):
                toks = split_tokens(line)
                cur = Stream(name=toks[0][1:])
                _absorb_decl(cur, toks[1:])
                doc.streams.append(cur)
                last_was_decl = True
                continue

            if cur is None:
                continue

            # indented line right after a declaration continues it
            if indented and last_was_decl:
                _absorb_decl(cur, split_tokens(line))
                continue
            last_was_decl = False

            # ---- stream body -----------------------------------------------
            if line.startswith("meta"):
                for tok in split_tokens(line)[1:]:
                    k, _, v = tok.partition("=")
                    v = v.strip('"')
                    if k == "warn":
                        cur.warns.append(v)
                    else:
                        cur.meta[k] = v
                continue

            if line.startswith("%pat"):
                m = re.match(r"^%pat\s+(\S+)\s*\{(.*)\}$", line, re.DOTALL)
                if not m:
                    raise ValueError("malformed %pat")
                cur.patterns[m.group(1)] = [
                    parse_event(e) for e in split_events(m.group(2))
                ]
                continue

            if line.startswith("bars "):
                cur.bindings.append(_parse_binding(split_tokens(line), line))
                continue

            if line.startswith("!auto"):
                t = split_tokens(line)
                cur.autos.append({
                    "target": t[1],
                    "from": parse_time(t[2]), "from_val": float(t[3]),
                    "to": parse_time(t[5]), "to_val": float(t[6]),
                })
                continue

            # event lines start with a time token
            head = split_tokens(line)[0]
            if head.startswith("@") or (":" in head and head[0].isdigit()):
                cur.events.extend(parse_event(e) for e in split_events(line))
                continue

            # everything else is a stream-schema statement, kept verbatim too
            toks = split_tokens(line)
            cur.statements.append((toks[0], toks[1:]))
        except ParseError:
            raise
        except Exception as exc:                      # noqa: BLE001
            raise ParseError(n, line, str(exc)) from exc

    return doc


def _absorb_decl(stream: Stream, toks: list[str]) -> None:
    """Attach declaration fields, blocks and the freeform gloss."""
    for tok in toks:
        if tok.startswith("~") and '"' in tok:
            stream.gloss = tok.split('"', 1)[1].rsplit('"', 1)[0]
        elif m := _BLOCK_RE.match(tok):
            stream.blocks[m.group(1)] = m.group(2).strip()
        elif "=" in tok:
            k, _, v = tok.partition("=")
            stream.fields[k] = v.strip('"')
        else:
            stream.unknown.append(tok)


def parse_file(path: str) -> Document:
    from pathlib import Path

    with open(path, encoding="utf-8") as fh:
        doc = parse(fh.read())
    doc.path = Path(path)
    return doc
