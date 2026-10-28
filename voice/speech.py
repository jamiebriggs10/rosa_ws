"""Speaking and state sounds: Rosa's mouth.

The console composes the words (it knows the facts); this module only turns text into
played audio. Two parts:

  Speaker  text -> samples (an *engine*: Piper, or anything else with the same shape)
           -> the output device (a *player*), on its own thread, with a cache so the
           fixed phrases ("Cancelled.") cost no synthesis the second time. While it
           plays, `on_speaking(True)` lets the pipeline mute the microphone, so Rosa
           never hears herself; `on_speaking(False)` always follows, even on an error.
  Sounds   six short tones synthesised with numpy on first use (plus a quiet "think" pulse),
           played through the same kind of player. One per state of the console's loop.

Engines and players are plain callables, so the tests run with fakes:
    engine(text) -> (float32 samples, rate) or None when it cannot speak
    player(samples, rate, stop_event) -> blocks until played or stop_event is set
"""

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RATE = 22050


# ---------------------------------------------------------------- the speaker

class Speaker:
    def __init__(self, engine, player, cache_dir=None, on_speaking=None, voice_key="voice"):
        self.engine, self.player = engine, player
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.on_speaking = on_speaking or (lambda on: None)
        self.voice_key = voice_key
        self.speaking = False
        self.reason = ""                     # why the last say() did not speak
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def say(self, text):
        """Speak `text`, cancelling anything still playing. True if playback started."""
        text = (text or "").strip()
        if not text:
            self.reason = "nothing to say"
            return False
        self.cancel()
        samples, rate = self._samples(text)
        if samples is None:
            return False
        with self._lock:
            self._stop = threading.Event()
            stop = self._stop
            self.speaking = True
            self._thread = threading.Thread(target=self._play, args=(samples, rate, stop), daemon=True)
            self._thread.start()
        return True

    def cancel(self):
        """Stop the current utterance, if any, and wait for its thread to let go."""
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def wait(self, timeout_s):
        """Block until the current utterance has finished (tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    # -- inside --------------------------------------------------------------

    def _samples(self, text):
        cached = self._cache_path(text)
        if cached is not None and cached.exists():
            try:
                return np.load(cached), RATE
            except (OSError, ValueError):
                pass
        try:
            result = self.engine(text)
        except Exception as exc:              # a broken engine must never take the console down
            self.reason = f"engine failed: {exc}"
            log.warning(self.reason)
            return None, None
        if result is None:
            self.reason = "no speech engine is available"
            return None, None
        samples, rate = result
        samples = np.asarray(samples, dtype=np.float32)
        if cached is not None and rate == RATE:
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, samples)
            except OSError:
                pass
        self.reason = ""
        return samples, rate

    def _cache_path(self, text):
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(f"{self.voice_key}\n{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def _play(self, samples, rate, stop):
        try:
            self.on_speaking(True)
            self.player(samples, rate, stop)
        except Exception as exc:
            log.warning("playback failed: %s", exc)
        finally:
            with self._lock:
                if self._stop is stop:
                    self.speaking = False
            self.on_speaking(False)


# ----------------------------------------------------------------- the sounds

def _tone(freq, dur, gain=0.25, kind="sine", rate=RATE):
    """One note with a 12 ms attack and an exponential decay."""
    t = np.arange(int(rate * dur)) / rate
    wave = np.sin(2 * np.pi * freq * t)
    if kind == "square":
        wave = np.sign(wave)
    attack = np.minimum(1.0, t / 0.012)
    decay = np.exp(-t / (dur / 3.5))
    return (gain * wave * attack * decay).astype(np.float32)


def _seq(*notes):
    """Notes played one after another: (freq, dur) or (freq, dur, gain) or (freq, dur, gain, kind)."""
    return np.concatenate([_tone(*n) for n in notes])


def make_sounds():
    return {
        "listen": _seq((660, 0.09), (880, 0.12)),
        "heard": _tone(1250, 0.045, 0.18),
        "ask": _seq((520, 0.11), (780, 0.16)),
        "go": _seq((780, 0.09), (520, 0.14)),
        "done": _seq((523, 0.08), (659, 0.08), (784, 0.18)),
        "cancel": _tone(300, 0.16, 0.2),
        "refused": _tone(160, 0.26, 0.12, "square"),
        "think": _tone(330, 0.14, 0.07),
    }


_SOUNDS = None


def sounds():
    """The sounds, made on first use (so importing this module costs nothing)."""
    global _SOUNDS
    if _SOUNDS is None:
        _SOUNDS = make_sounds()
    return _SOUNDS


class Sounds:
    def __init__(self, player, sounds=None):
        self.player = player
        self.sounds = sounds

    def play(self, name):
        """Play one sound without blocking. False for a name that is not a sound."""
        samples = (self.sounds or sounds()).get(name)
        if samples is None:
            return False
        threading.Thread(target=self._play, args=(samples,), daemon=True).start()
        return True

    def _play(self, samples):
        try:
            self.player(samples, RATE, threading.Event())
        except Exception as exc:
            log.warning("sound failed: %s", exc)


# ------------------------------------------------------- the real engine and player

def piper_engine(voice, data_dir):
    """An engine built on Piper, or None (with the reason logged) when it cannot be built.
    Voices live in data_dir as <voice>.onnx and <voice>.onnx.json."""
    model = Path(data_dir) / f"{voice}.onnx"
    try:
        from piper import PiperVoice
    except ImportError as exc:
        log.warning("piper is not installed (%s): Rosa will not speak", exc)
        return None
    if not model.exists():
        log.warning("no voice file %s: Rosa will not speak", model)
        return None
    try:
        piper_voice = PiperVoice.load(str(model))
    except Exception as exc:
        log.warning("could not load the voice %s: %s", model, exc)
        return None
    rate = int(getattr(piper_voice.config, "sample_rate", RATE))

    def engine(text):
        chunks = []
        for chunk in piper_voice.synthesize(text):
            audio = getattr(chunk, "audio_float_array", None)
            if audio is None:                   # older piper: int16 bytes
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768
            chunks.append(audio)
        if not chunks:
            return np.zeros(0, dtype=np.float32), rate
        return np.concatenate(chunks).astype(np.float32), rate

    return engine


def sounddevice_player(output_device=None, sd_module=None):
    """A player on the sound card: each playback gets its own output stream, written in short
    chunks, and only that stream is stopped when `stop` is set. (sounddevice's module-level
    play()/stop() share one stream, so a sound played next to a spoken line would cut the
    line off; that is why not those.)"""
    sd = sd_module
    if sd is None:
        import sounddevice as sd

    def player(samples, rate, stop):
        samples = np.asarray(samples, dtype=np.float32)
        stream = sd.OutputStream(samplerate=rate, channels=1, device=output_device, dtype="float32")
        chunk = max(256, int(rate * 0.05))                 # 50 ms: how quickly a cancel takes effect
        try:
            stream.start()
            for i in range(0, len(samples), chunk):
                if stop.is_set():
                    break
                stream.write(samples[i:i + chunk])
        finally:
            try:
                stream.stop()
            finally:
                stream.close()

    return player
