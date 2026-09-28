"""stage_lyrics: drop non-speech segments and lone hallucinations (task 5a)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import encode as enc  # noqa: E402

GRID = {"downbeat": 1.0, "bar_dur": 2.0}


class FakeWord:
    def __init__(self, start, end, word, probability):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class FakeSegment:
    def __init__(self, no_speech_prob, words):
        self.no_speech_prob = no_speech_prob
        self.words = words


class FakeModel:
    calls = []

    def __init__(self, *a, **k):
        pass

    def transcribe(self, path, **kwargs):
        FakeModel.calls.append(kwargs)
        return FakeModel.segments, None


def _run(monkeypatch, segments, lines=None):
    import faster_whisper
    FakeModel.segments = segments
    FakeModel.calls = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    return enc.stage_lyrics(Path("fake.wav"), 44100, GRID, lines, 30.0)


def _text(st) -> str:
    return " | ".join(st.lines)


def test_a_high_no_speech_prob_segment_is_dropped_a_low_one_is_kept(monkeypatch):
    segs = [
        FakeSegment(0.9, [FakeWord(1.0, 1.3, " dropped", 0.9)]),
        FakeSegment(0.1, [FakeWord(2.0, 2.3, " kept", 0.9)]),
    ]
    st = _run(monkeypatch, segs)
    assert "kept" in _text(st) and "dropped" not in _text(st)
    assert FakeModel.calls[0]["word_timestamps"] is True
    assert FakeModel.calls[0]["condition_on_previous_text"] is False


def test_low_confidence_thank_you_is_a_hallucination(monkeypatch):
    segs = [FakeSegment(0.1, [FakeWord(4.288, 4.5, " Thank", 0.19),
                              FakeWord(4.5, 4.7, " you.", 0.4)])]
    st = _run(monkeypatch, segs)
    assert st.ok is False
    assert st.lines == []
    assert any('ASR heard only "Thank you."' in w and "hallucination" in w for w in st.warns)


def test_one_low_confidence_word_is_a_hallucination_even_if_the_other_is_confident(monkeypatch):
    """Real regression (fix round 2): 999999-30s heard "Thank" ?0.19 / "you."
    ?0.9-ish -- the MEAN cleared 0.5 so the old guard kept it. A truly sung
    "thank you" is confident on every word, so the guard must use the
    MINIMUM word probability, not the mean."""
    segs = [FakeSegment(0.1, [FakeWord(4.288, 4.5, " Thank", 0.19),
                              FakeWord(4.5, 4.7, " you.", 0.95)])]
    st = _run(monkeypatch, segs)
    assert st.ok is False
    assert st.lines == []
    assert any('ASR heard only "Thank you."' in w and "hallucination" in w for w in st.warns)


def test_corona_radiata_thank_regression_is_a_hallucination(monkeypatch):
    """Real regression (fix round 2): corona_radiata-30s heard "Thank" ?0.02."""
    segs = [FakeSegment(0.1, [FakeWord(4.288, 4.5, " Thank", 0.02),
                              FakeWord(4.5, 4.7, " you.", 0.97)])]
    st = _run(monkeypatch, segs)
    assert st.ok is False
    assert st.lines == []
    assert any('ASR heard only "Thank you."' in w and "hallucination" in w for w in st.warns)


def test_high_confidence_thank_you_is_kept_a_singer_can_sing_it(monkeypatch):
    segs = [FakeSegment(0.1, [FakeWord(4.288, 4.5, " Thank", 0.95),
                              FakeWord(4.5, 4.7, " you.", 0.95)])]
    st = _run(monkeypatch, segs)
    assert st.ok is True
    assert "Thank" in _text(st) and "you." in _text(st)
    assert not any("hallucination" in w for w in st.warns)


def test_a_normal_transcript_is_unchanged(monkeypatch):
    segs = [FakeSegment(0.05, [FakeWord(1.0, 1.3, " Walk", 0.9),
                               FakeWord(1.3, 1.6, " me", 0.9),
                               FakeWord(1.6, 2.0, " down", 0.9)])]
    st = _run(monkeypatch, segs)
    assert st.ok is True
    assert "Walk" in _text(st) and "me" in _text(st) and "down" in _text(st)
    assert not st.warns
