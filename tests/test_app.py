import pytest

from ema_voice.api import EMAVoice
from ema_voice.app import build, enroll, speak, voice_choices


@pytest.fixture(scope="module")
def tts(voice_ckpt):
    return EMAVoice.load(voice_ckpt, device="cpu")


def test_speak_returns_audio_and_seed(tts):
    (rate, audio), info = speak(tts, "Merhaba.", "v0", 1.0, 3)
    assert rate == 48000 and len(audio) > 0 and "seed 3" in info


@pytest.mark.parametrize("text,voice", [("", "v0"), ("   ", "v0"), ("Merhaba.", None)])
def test_speak_rejects_empty_input(tts, text, voice):
    with pytest.raises(ValueError):
        speak(tts, text, voice)


def test_enroll_adds_voice_listed_first(voice_ckpt, recordings, tmp_path):
    files = sorted(p for p in recordings.iterdir() if p.suffix in (".wav", ".txt"))
    path = enroll(voice_ckpt, tmp_path / "voices", "cpu", "ben", files, None, "instant")
    tts = EMAVoice.load(voice_ckpt, device="cpu", voices_dir=tmp_path / "voices")
    assert path.stem == "ben" and voice_choices(tts)[0] == "ben"


@pytest.mark.parametrize("name", ["", "a b", "../x", "x" * 41])
def test_enroll_rejects_bad_names(voice_ckpt, recordings, tmp_path, name):
    with pytest.raises(ValueError):
        enroll(voice_ckpt, tmp_path, "cpu", name, list(recordings.glob("*.wav")), None, "instant")


def test_enroll_needs_audio(voice_ckpt, tmp_path):
    with pytest.raises(ValueError):
        enroll(voice_ckpt, tmp_path, "cpu", "ben", [], None, "instant")


def test_page_builds(voice_ckpt, tmp_path):
    pytest.importorskip("gradio")
    assert build(voice_ckpt, tmp_path, "cpu") is not None
