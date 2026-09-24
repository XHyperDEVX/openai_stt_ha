"""Save the outgoing audio payload outside Home Assistant's public www folder."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import io
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave

_LOGGER = logging.getLogger(__name__)


def _save_debug_audio(
    directory: str,
    audio: bytes | bytearray,
    transport: str,
    sample_rate: int | None,
) -> Path:
    """Save exact HTTP WAV bytes, or wrap outgoing Realtime PCM in a WAV."""
    if sample_rate is not None:
        wav_stream = io.BytesIO()
        with wave.open(wav_stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio)
        audio = wav_stream.getvalue()

    folder = Path(directory)
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    path = folder / f"{timestamp}_{transport}_{uuid4().hex}.wav"
    # No overwriting; files should not be readable by other local OS users.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(audio)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


async def async_save_debug_audio(
    directory: str | None,
    audio: bytes | bytearray,
    transport: str,
    sample_rate: int | None = None,
    executor: Callable[..., Awaitable[Any]] | None = None,
) -> None:
    """Perform file I/O off-loop; a debug disk failure must not break STT.

    sample_rate=None means audio already is the complete HTTP WAV payload.
    Otherwise audio is the exact successfully appended Realtime PCM payload.
    Debug captures can therefore be partial if the network fails mid-stream.
    """
    if directory is None or not audio:
        return
    run = executor or asyncio.to_thread
    try:
        path = await run(_save_debug_audio, directory, audio, transport, sample_rate)
    except Exception:
        _LOGGER.exception("Could not save outgoing OpenAI STT debug audio")
    else:
        _LOGGER.debug("Saved outgoing OpenAI STT audio to %s", path)
