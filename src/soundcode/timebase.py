"""Timebase: seconds are canonical, bar:beat is sugar.

The `:grid` stream supplies a tempo *curve* (linearly interpolated between
points) plus `anchor` lines pinning specific bars to absolute seconds.

Anchors win. Between two anchors the tempo curve only supplies the *shape* of
the accelerando/ritardando; it is rescaled so the segment lands exactly on the
next anchor. That is what stops beat-tracking drift from accumulating over a
four-minute song — the spec requires an anchor at least every 16 bars.

After the final anchor the curve is used directly, unscaled.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass
class Grid:
    # (time_seconds, bpm), sorted
    tempo: list[tuple[float, float]] = field(default_factory=list)
    # (time_seconds, numerator, denominator), sorted
    meter: list[tuple[float, int, int]] = field(default_factory=list)
    # (bar_number, time_seconds), sorted by bar
    anchors: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.tempo.sort(key=lambda p: p[0])
        self.meter.sort(key=lambda p: p[0])
        self.anchors.sort(key=lambda p: p[0])
        if not self.tempo:
            self.tempo = [(0.0, 120.0)]
        if not self.meter:
            self.meter = [(0.0, 4, 4)]
        if not self.anchors:
            self.anchors = [(1, 0.0)]

    # -- meter ---------------------------------------------------------------

    def beats_per_bar(self, bar: int) -> float:
        """Beats in `bar`. Meter changes are rare, so the linear scan is fine."""
        num, den = self.meter[0][1], self.meter[0][2]
        for t, n, d in self.meter:
            # meter changes are declared at a time; resolve via the bar it starts
            if t <= self._approx_bar_time(bar):
                num, den = n, d
        # a beat is a quarter note; 6/8 counts as 3 quarter-note beats
        return num * (4.0 / den)

    def _approx_bar_time(self, bar: int) -> float:
        """Cheap estimate used only to select a meter, avoiding recursion."""
        a_bar, a_time = self.anchors[0]
        bpm = self.tempo[0][1]
        return a_time + (bar - a_bar) * self.meter[0][1] * (60.0 / bpm)

    def beat_index(self, bar: int, beat: float) -> float:
        """Absolute beat count from the start of bar 1 (bar 1 beat 1 -> 0.0)."""
        total = 0.0
        for b in range(1, bar):
            total += self.beats_per_bar(b)
        return total + (beat - 1.0)

    # -- tempo curve ---------------------------------------------------------

    def _bpm_at(self, t: float) -> float:
        pts = self.tempo
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        i = bisect_right([p[0] for p in pts], t) - 1
        t0, b0 = pts[i]
        t1, b1 = pts[i + 1]
        if t1 == t0:
            return b1
        return b0 + (b1 - b0) * (t - t0) / (t1 - t0)

    def _beats_between(self, t0: float, t1: float) -> float:
        """Integrate bpm/60 over [t0, t1]. Piecewise-linear, so trapezoids are exact."""
        if t1 <= t0:
            return 0.0
        edges = [t0] + [t for t, _ in self.tempo if t0 < t < t1] + [t1]
        total = 0.0
        for a, b in zip(edges, edges[1:]):
            total += (b - a) * (self._bpm_at(a) + self._bpm_at(b)) / 2.0 / 60.0
        return total

    def _time_after(self, t0: float, beats: float) -> float:
        """Invert the integral: time at which `beats` have elapsed since t0."""
        if beats <= 0:
            return t0
        # walk tempo segments, then solve the final partial segment
        t = t0
        remaining = beats
        edges = [x for x, _ in self.tempo if x > t0] + [float("inf")]
        for edge in edges:
            span = self._beats_between(t, edge) if edge != float("inf") else float("inf")
            if span >= remaining:
                b0 = self._bpm_at(t)
                if edge == float("inf"):
                    return t + remaining * 60.0 / b0
                b1 = self._bpm_at(edge)
                dt = edge - t
                if abs(b1 - b0) < 1e-9:
                    return t + remaining * 60.0 / b0
                # solve (b0*x + (b1-b0)*x^2/(2*dt)) / 60 = remaining
                a = (b1 - b0) / (2.0 * dt)
                disc = b0 * b0 + 4.0 * a * remaining * 60.0
                x = (-b0 + disc**0.5) / (2.0 * a)
                return t + x
            remaining -= span
            t = edge
        return t

    # -- public conversion ---------------------------------------------------

    def time_of(self, bar: int, beat: float = 1.0) -> float:
        """Absolute seconds for a bar:beat position, honouring anchors."""
        target = self.beat_index(bar, beat)

        prev = self.anchors[0]
        nxt = None
        for a in self.anchors:
            if self.beat_index(a[0], 1.0) <= target:
                prev = a
            else:
                nxt = a
                break

        p_beats = self.beat_index(prev[0], 1.0)
        want = target - p_beats
        unscaled = self._time_after(prev[1], want)

        if nxt is None:
            return unscaled

        # rescale the segment so it lands exactly on the next anchor
        seg_beats = self.beat_index(nxt[0], 1.0) - p_beats
        seg_unscaled = self._time_after(prev[1], seg_beats) - prev[1]
        seg_actual = nxt[1] - prev[1]
        if seg_unscaled <= 0:
            return prev[1] + seg_actual * (want / seg_beats if seg_beats else 0.0)
        scale = seg_actual / seg_unscaled
        return prev[1] + (unscaled - prev[1]) * scale

    def duration_seconds(self, bar: int, beat: float, beats: float) -> float:
        """Length in seconds of `beats` beats starting at bar:beat."""
        start = self.time_of(bar, beat)
        idx = self.beat_index(bar, beat) + beats
        # convert the end beat index back to a bar:beat, then to time
        b = 1
        remaining = idx
        while remaining >= self.beats_per_bar(b):
            remaining -= self.beats_per_bar(b)
            b += 1
        return self.time_of(b, remaining + 1.0) - start
