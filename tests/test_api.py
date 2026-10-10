import numpy as np
import pytest
import soundfile as sf

from ema_voice.api import EMAVoice


@pytest.fixture(scope="module")
def tts(voice_ckpt):
    return EMAVoice.load(voice_ckpt, device="cpu")


def test_say_writes_wav_48k(tts, tmp_path):
    speech = tts.say("Merhaba, bu bir deneme.", voice="v0", seed=1, steps=2, path=tmp_path / "a.wav")
    audio, sr = sf.read(tmp_path / "a.wav")
    assert sr == 48000 and speech.sample_rate == 48000 and len(audio) == len(speech.audio) > 48000 * 0.5
    assert speech.seed == 1 and np.abs(speech.audio).max() <= 1.0


def test_unknown_voice_raises_value_error(tts):
    with pytest.raises(ValueError, match="nope"):
        tts.say("Merhaba.", voice="nope")


@pytest.mark.parametrize("text", ["", "   ", "...", "\u200f\x00"])
def test_textless_input_returns_silence(tts, text):
    speech = tts.say(text, voice="v1", steps=2)
    assert speech.duration <= 0.1


def test_emoji_never_raises(tts):
    tts.say("😀", voice="v1", steps=1)  # EMA's normaliser reads symbols aloud; it must just not fail


def test_long_text_chunked(tts):
    text = "Bu cümle uzun bir metnin parçasıdır ve okunması biraz zaman alır. " * 9
    speech = tts.say(text, voice="v0", steps=1, seed=0)
    assert speech.duration > 20


def test_voices_lists_table(tts):
    assert tts.voices()[:2] == ["v0", "v1"]
