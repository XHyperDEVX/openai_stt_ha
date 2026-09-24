"""Debug captures are private, unique and byte-exact; no API calls are made."""

import asyncio
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from custom_components.openai_stt.debug_audio import async_save_debug_audio


class DebugAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_wav_bytes_are_not_reencoded(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x02" * 100)
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "debug"
            await asyncio.gather(*(
                async_save_debug_audio(str(folder), payload, "http") for _ in range(3)
            ))
            files = list(folder.glob("*.wav"))
            self.assertEqual(len(files), 3)
            for path in files:
                self.assertEqual(path.read_bytes(), payload)
                if os.name == "posix":
                    self.assertEqual(path.stat().st_mode & 0o077, 0)
            if os.name == "posix":
                self.assertEqual(folder.stat().st_mode & 0o077, 0)

    async def test_realtime_capture_has_exact_pcm_and_24khz_header(self):
        payload = b"\x11\x22" * 2400
        with tempfile.TemporaryDirectory() as root:
            await async_save_debug_audio(root, payload, "realtime", sample_rate=24000)
            path, = Path(root).glob("*.wav")
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 24000)
                self.assertEqual(wav.readframes(wav.getnframes()), payload)

    async def test_disabled_and_empty_capture_do_not_touch_disk(self):
        with patch("custom_components.openai_stt.debug_audio._save_debug_audio") as save:
            await async_save_debug_audio(None, b"audio", "http")
            await async_save_debug_audio("unused", b"", "realtime", 24000)
            save.assert_not_called()

    async def test_disk_failure_is_logged_not_propagated(self):
        async def broken_executor(*args):
            raise OSError("Disk full")

        with self.assertLogs("custom_components.openai_stt.debug_audio", level="ERROR"):
            await async_save_debug_audio("unused", b"audio", "http", executor=broken_executor)


if __name__ == "__main__":
    unittest.main()
