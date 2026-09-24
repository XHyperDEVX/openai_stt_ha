"""Provider/transport contract tests with minimal HA stubs and fake networks.

These do not test a running Home Assistant instance or OpenAI's recognition
quality. Stub imports are scoped to module loading; no real API key is used.
"""

import asyncio
import base64
from dataclasses import dataclass
from enum import IntEnum, StrEnum
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import numpy as np
import voluptuous as vol


class AudioFormats(StrEnum):
    WAV = "wav"
    OGG = "ogg"


class AudioCodecs(StrEnum):
    PCM = "pcm"
    OPUS = "opus"


class AudioBitRates(IntEnum):
    BITRATE_16 = 16


class AudioChannels(IntEnum):
    CHANNEL_MONO = 1


class AudioSampleRates(IntEnum):
    SAMPLERATE_16000 = 16000


class SpeechResultState(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class SpeechResult:
    text: str
    result: SpeechResultState


def load_integration():
    """Load the real provider/clients, substituting only HA's import surface."""
    root = Path(__file__).resolve().parents[1] / "custom_components" / "openai_stt"
    stubs = {}
    for name in (
        "homeassistant", "homeassistant.components", "homeassistant.components.stt",
        "homeassistant.core", "homeassistant.helpers",
        "homeassistant.helpers.aiohttp_client",
        "homeassistant.helpers.config_validation",
    ):
        stubs[name] = ModuleType(name)
        stubs[name].__path__ = []
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(stubs[parent], child, stubs[name])
    stt = stubs["homeassistant.components.stt"]
    for cls in (
        AudioFormats, AudioCodecs, AudioBitRates, AudioChannels, AudioSampleRates,
        SpeechResult, SpeechResultState,
    ):
        setattr(stt, cls.__name__, cls)
    stt.Provider = type("Provider", (), {})
    stt.SpeechMetadata = SimpleNamespace
    stubs["homeassistant.core"].HomeAssistant = SimpleNamespace
    stubs["homeassistant.helpers.aiohttp_client"].async_get_clientsession = (
        lambda hass: hass.session
    )
    cv = stubs["homeassistant.helpers.config_validation"]
    cv.PLATFORM_SCHEMA = vol.Schema({vol.Required("platform"): str})
    cv.string = str
    cv.boolean = vol.Boolean()
    cv.positive_int = vol.All(vol.Coerce(int), vol.Range(min=0))

    modules = {}
    with patch.dict(sys.modules, stubs):
        # Temporary standard names allow the provider's relative imports to
        # resolve to these exact clients. Do not leave fake HA imports installed.
        with patch.dict(sys.modules):
            for name in ("http_client", "websocket_client", "stt"):
                qualified = f"custom_components.openai_stt.{name}"
                spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
                modules[name] = module
    return modules


MODULES = load_integration()
STT = MODULES["stt"]
HTTP = MODULES["http_client"]
WS = MODULES["websocket_client"]


def metadata():
    return SimpleNamespace(
        language="de", format=AudioFormats.WAV, codec=AudioCodecs.PCM,
        bit_rate=16, channel=1, sample_rate=16000,
    )


def make_pcm(count=4000):
    samples = 8000 * np.sin(2 * np.pi * 330 * np.arange(count) / 16000)
    return np.rint(samples).astype("<i2").tobytes()


async def source(pcm, size=777):
    yield b""  # Empty chunks must not mark end-of-stream.
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


def fake_hass(root):
    response = SimpleNamespace(
        raise_for_status=Mock(), json=AsyncMock(return_value={"text": "Licht an"})
    )
    return SimpleNamespace(
        config=SimpleNamespace(path=lambda *parts: str(Path(root).joinpath(*parts))),
        async_add_executor_job=asyncio.to_thread,
        session=SimpleNamespace(post=AsyncMock(return_value=response)),
    )


class FakeWebSocket:
    closed = False

    def __init__(self, fail_after=None):
        self.events = []
        self._fail_after = fail_after

    async def send_json(self, event):
        if self._fail_after is not None and len(self.events) >= self._fail_after:
            raise WS.OpenAIRealtimeError("Simulated network failure")
        self.events.append(event)

    def outgoing_pcm(self):
        return b"".join(
            base64.b64decode(event["audio"])
            for event in self.events
            if event["type"] == "input_audio_buffer.append"
        )


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_requested_yaml_options(self):
        config = STT.PLATFORM_SCHEMA({"platform": "openai_stt", "api_key": "test"})
        self.assertFalse(config["debug_save"])
        self.assertFalse(config["pitch_up_vocoder"])
        self.assertEqual(config["pitch_semitones"], 1.0)
        self.assertEqual(config["vocoder_strength"], 0.15)
        config = STT.PLATFORM_SCHEMA({
            "platform": "openai_stt", "api_key": "test", "realtime": True,
            "model": "gpt-transcribe", "debug_save": True,
            "pitch_up_vocoder": True, "pitch_semitones": 1.0,
            "vocoder_strength": 0.15,
        })
        self.assertTrue(config["debug_save"])
        self.assertTrue(config["pitch_up_vocoder"])

    def test_rejects_invalid_ranges_and_nonfinite_numbers(self):
        for key, values in (
            ("pitch_semitones", (-13, 13, float("nan"), float("inf"))),
            ("vocoder_strength", (-0.01, 1.01, float("nan"), float("inf"))),
        ):
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(vol.Invalid):
                        STT.PLATFORM_SCHEMA({
                            "platform": "openai_stt", "api_key": "test", key: value,
                        })


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def provider(self, root, **options):
        hass = fake_hass(root)
        config = STT.PLATFORM_SCHEMA({
            "platform": "openai_stt", "api_key": "test", **options,
        })
        return await STT.async_get_engine(hass, config), hass

    @staticmethod
    def http_payload(hass):
        form = hass.session.post.await_args.kwargs["data"]
        return next(value for options, _, value in form._fields if options["name"] == "file")

    async def test_http_processed_capture_equals_uploaded_wav(self):
        with tempfile.TemporaryDirectory() as root:
            provider, hass = await self.provider(root, debug_save=True, pitch_up_vocoder=True)
            pcm = make_pcm()
            result = await provider.async_process_audio_stream(metadata(), source(pcm))
            self.assertEqual(result.result, SpeechResultState.SUCCESS)
            payload = self.http_payload(hass)
            file, = (Path(root) / "openai_stt_debug").glob("*.wav")
            self.assertEqual(file.read_bytes(), payload)
            with wave.open(io.BytesIO(payload), "rb") as wav:
                processed = wav.readframes(wav.getnframes())
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(len(processed), len(pcm))
                self.assertNotEqual(processed, pcm)

    async def test_disabled_processing_keeps_original_pcm_and_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as root:
            provider, hass = await self.provider(root)
            pcm = make_pcm()
            await provider.async_process_audio_stream(metadata(), source(pcm))
            with wave.open(io.BytesIO(self.http_payload(hass)), "rb") as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), pcm)
            self.assertFalse((Path(root) / "openai_stt_debug").exists())

    async def test_debug_save_works_without_voice_effects(self):
        with tempfile.TemporaryDirectory() as root:
            provider, hass = await self.provider(root, debug_save=True)
            pcm = make_pcm()
            await provider.async_process_audio_stream(metadata(), source(pcm))
            file, = (Path(root) / "openai_stt_debug").glob("*.wav")
            self.assertEqual(file.read_bytes(), self.http_payload(hass))
            with wave.open(str(file), "rb") as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), pcm)

    async def test_processing_failure_never_uploads_original_audio(self):
        with tempfile.TemporaryDirectory() as root:
            provider, hass = await self.provider(root, pitch_up_vocoder=True)
            with self.assertLogs(STT.__name__, level="ERROR"):
                result = await provider.async_process_audio_stream(
                    metadata(), source(make_pcm() + b"\x01")
                )
            self.assertEqual(result.result, SpeechResultState.ERROR)
            hass.session.post.assert_not_awaited()

    async def test_compressed_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            provider, hass = await self.provider(root, pitch_up_vocoder=True)
            info = metadata()
            info.codec = AudioCodecs.OPUS
            info.format = AudioFormats.OGG
            with self.assertLogs(STT.__name__, level="ERROR"):
                result = await provider.async_process_audio_stream(info, source(b"OggS"))
            self.assertEqual(result.result, SpeechResultState.ERROR)
            hass.session.post.assert_not_awaited()
            self.assertNotIn(AudioCodecs.OPUS, provider.supported_codecs)

    async def test_realtime_provider_processes_before_resampling_and_capture(self):
        with tempfile.TemporaryDirectory() as root:
            provider, _ = await self.provider(
                root, realtime=True, debug_save=True, pitch_up_vocoder=True,
            )
            ws = FakeWebSocket()
            client = provider._client
            captured_processed_pcm = bytearray()
            committed = asyncio.Event()

            async def run_without_network(info, stream):
                async def observe_processed_stream():
                    async for chunk in stream:
                        captured_processed_pcm.extend(chunk)
                        yield chunk
                await client._send_audio_stream(ws, info, observe_processed_stream(), committed)
                return SpeechResult("Licht an", SpeechResultState.SUCCESS)

            client.async_process_audio_stream = run_without_network
            pcm = make_pcm()
            result = await provider.async_process_audio_stream(metadata(), source(pcm))
            self.assertEqual(result.result, SpeechResultState.SUCCESS)
            self.assertTrue(committed.is_set())
            self.assertEqual(ws.events[-1]["type"], "input_audio_buffer.commit")
            self.assertEqual(len(captured_processed_pcm), len(pcm))
            self.assertNotEqual(bytes(captured_processed_pcm), pcm)
            expected = WS.PCM16Resampler(16000, 24000).process(bytes(captured_processed_pcm))
            self.assertEqual(ws.outgoing_pcm(), expected)
            file, = (Path(root) / "openai_stt_debug").glob("*.wav")
            with wave.open(str(file), "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.readframes(wav.getnframes()), expected)

    async def test_realtime_failure_saves_only_successful_appends_and_does_not_commit(self):
        with tempfile.TemporaryDirectory() as root:
            client = WS.OpenAIWebSocketClient(
                None, "test", "https://invalid.test/v1", "gpt-transcribe", "", None,
                debug_directory=root,
            )
            ws = FakeWebSocket(fail_after=1)
            committed = asyncio.Event()
            with self.assertRaises(WS.OpenAIRealtimeError):
                await client._send_audio_stream(ws, metadata(), source(make_pcm()), committed)
            self.assertFalse(committed.is_set())
            file, = Path(root).glob("*.wav")
            with wave.open(str(file), "rb") as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), ws.outgoing_pcm())

    async def test_realtime_disabled_capture_does_not_record(self):
        client = WS.OpenAIWebSocketClient(
            None, "test", "https://invalid.test/v1", "gpt-transcribe", "", None,
        )
        with patch.object(WS, "async_save_debug_audio", new_callable=AsyncMock) as save:
            await client._send_audio_stream(
                FakeWebSocket(), metadata(), source(make_pcm()), asyncio.Event()
            )
            save.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
