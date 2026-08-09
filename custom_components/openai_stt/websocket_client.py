"""WebSocket client for OpenAI Realtime transcription."""

from __future__ import annotations

import asyncio
from array import array
import base64
from collections.abc import AsyncIterable
import json
import logging
import math
import sys
import time
from typing import Final

from aiohttp import ClientError, WSMsgType

from homeassistant.components.stt import SpeechMetadata, SpeechResult, SpeechResultState

_LOGGER = logging.getLogger(__name__)

OPENAI_SAMPLE_RATE: Final = 24000
WEBSOCKET_RESPONSE_TIMEOUT: Final = 30

# The current Realtime transcription API uses different model names than the
# file-transcription API. Keep existing configurations working by translating
# the old names when realtime mode is enabled.
REALTIME_MODEL_ALIASES: Final = {
    "gpt-4o-mini-transcribe": "gpt-live-transcribe",
    "gpt-4o-transcribe": "gpt-transcribe",
    "whisper-1": "gpt-live-transcribe",
}


class OpenAIRealtimeError(RuntimeError):
    """Raised when the Realtime API reports an error."""


class PCM16Resampler:
    """Small stateful mono PCM16 linear resampler.

    Home Assistant Voice satellites normally provide 16 kHz PCM, while the
    current OpenAI Realtime transcription API requires 24 kHz PCM. State is
    retained between chunks so chunk boundaries do not introduce gaps.
    """

    def __init__(self, input_rate: int, output_rate: int) -> None:
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._step = input_rate / output_rate
        self._next_output_position = 0.0
        self._input_samples_seen = 0
        self._previous_sample: int | None = None
        self._pending_byte = b""

    def process(self, pcm: bytes) -> bytes:
        """Resample one little-endian PCM16 chunk."""
        if not pcm:
            return b""

        pcm = self._pending_byte + pcm
        if len(pcm) % 2:
            self._pending_byte = pcm[-1:]
            pcm = pcm[:-1]
        else:
            self._pending_byte = b""

        if not pcm:
            return b""

        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()

        start_index = self._input_samples_seen
        end_index = start_index + len(samples) - 1
        output = array("h")

        while self._next_output_position <= end_index + 1e-9:
            left_index = math.floor(self._next_output_position)
            fraction = self._next_output_position - left_index

            if left_index < start_index:
                if self._previous_sample is None:
                    break
                left = self._previous_sample
            else:
                left = samples[left_index - start_index]

            if fraction <= 1e-9:
                value = left
            else:
                right_index = left_index + 1
                if right_index > end_index:
                    # The next chunk supplies the right-hand sample.
                    break
                right = samples[right_index - start_index]
                value = round(left + (right - left) * fraction)

            output.append(max(-32768, min(32767, value)))
            self._next_output_position += self._step

        self._input_samples_seen += len(samples)
        self._previous_sample = samples[-1]

        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()


class OpenAIWebSocketClient:
    """WebSocket client for the current OpenAI Realtime transcription API."""

    def __init__(
        self,
        client,
        api_key: str,
        api_url: str,
        model: str,
        prompt: str,
        noise_reduction: str | None,
    ) -> None:
        """Initialize the WebSocket client."""
        self.client = client
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.transcription_model = REALTIME_MODEL_ALIASES.get(model, model)
        self.prompt = prompt
        self.noise_reduction = noise_reduction

        if self.transcription_model != model:
            _LOGGER.info(
                "Realtime transcription model %s is now mapped to %s",
                model,
                self.transcription_model,
            )

    def _create_session_config(self, language: str) -> dict:
        """Create a session.update event for the current Realtime API."""
        transcription: dict = {"model": self.transcription_model}

        if self.prompt:
            transcription["prompt"] = self.prompt

        # gpt-live-transcribe supports language hints through the plural
        # `languages` field. gpt-transcribe detects the language after commit.
        if self.transcription_model == "gpt-live-transcribe" and language:
            transcription["languages"] = [language]

        audio_input: dict = {
            "format": {
                "type": "audio/pcm",
                "rate": OPENAI_SAMPLE_RATE,
            },
            "transcription": transcription,
            # Home Assistant owns the turn boundary. OpenAI must not commit on
            # the short pause between the wake word and the actual command.
            "turn_detection": None,
        }

        if self.noise_reduction:
            audio_input["noise_reduction"] = {"type": self.noise_reduction}

        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": audio_input},
            },
        }

    @staticmethod
    def _api_error_message(data: dict) -> str:
        """Extract a useful message from a Realtime error event."""
        error = data.get("error", {})
        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or "unknown_error"
            message = error.get("message") or str(error)
            return f"{code}: {message}"
        return str(error or data)

    async def _send_audio_stream(
        self,
        ws,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
        commit_sent: asyncio.Event,
    ) -> int:
        """Resample, send and finally commit Home Assistant's audio stream."""
        input_rate = int(metadata.sample_rate)
        if int(metadata.channel) != 1 or int(metadata.bit_rate) != 16:
            raise OpenAIRealtimeError(
                "Realtime transcription requires mono 16-bit PCM input; "
                f"received channels={metadata.channel}, bit_rate={metadata.bit_rate}"
            )

        resampler = PCM16Resampler(input_rate, OPENAI_SAMPLE_RATE)
        input_bytes = 0
        output_bytes = 0

        async for chunk in stream:
            if ws.closed:
                raise OpenAIRealtimeError("WebSocket closed while sending audio")
            if not chunk:
                # Only exhaustion of the async iterator marks end-of-stream.
                continue

            input_bytes += len(chunk)
            openai_pcm = (
                chunk
                if input_rate == OPENAI_SAMPLE_RATE
                else resampler.process(chunk)
            )
            if not openai_pcm:
                continue

            output_bytes += len(openai_pcm)
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(openai_pcm).decode("ascii"),
                }
            )

        _LOGGER.debug(
            "Audio stream ended: %d input bytes at %d Hz, %d output bytes at %d Hz",
            input_bytes,
            input_rate,
            output_bytes,
            OPENAI_SAMPLE_RATE,
        )

        if output_bytes == 0:
            return 0

        # Set the event first. This prevents a very fast completion response
        # from being mistaken for an unsolicited pre-commit completion.
        commit_sent.set()
        await ws.send_json({"type": "input_audio_buffer.commit"})
        _LOGGER.debug("Committed OpenAI input audio buffer")
        return output_bytes

    async def _receive_transcription(
        self,
        ws,
        commit_sent: asyncio.Event,
    ) -> str:
        """Receive transcript events until the manually committed turn completes."""
        deltas_by_item: dict[str, list[str]] = {}
        commit_time: float | None = None

        while True:
            # While audio is arriving, receive() is polled once per second. The
            # commit can happen during such a poll. In that case its one-second
            # TimeoutError must start the real post-commit timeout, not be treated
            # as if all 30 seconds had already elapsed.
            now = time.perf_counter()
            if commit_sent.is_set():
                if commit_time is None:
                    commit_time = now
                remaining = WEBSOCKET_RESPONSE_TIMEOUT - (now - commit_time)
                if remaining <= 0:
                    raise OpenAIRealtimeError(
                        "Timed out waiting for transcription after audio commit"
                    )
                timeout = remaining
            else:
                timeout = 1

            try:
                msg = await ws.receive(timeout=timeout)
            except TimeoutError:
                if not commit_sent.is_set():
                    continue

                if commit_time is None:
                    # The commit occurred while receive() was still using its
                    # one-second pre-commit polling timeout. Start a fresh full
                    # response timeout now.
                    commit_time = time.perf_counter()
                    _LOGGER.debug(
                        "Audio commit occurred during receive poll; starting %.0f "
                        "second transcription timeout",
                        WEBSOCKET_RESPONSE_TIMEOUT,
                    )
                    continue

                if time.perf_counter() - commit_time >= WEBSOCKET_RESPONSE_TIMEOUT:
                    raise OpenAIRealtimeError(
                        "Timed out waiting for transcription after audio commit"
                    )
                continue

            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                event_type = data.get("type", "")
                _LOGGER.debug("Received Realtime event: %s", data)

                if event_type == "error":
                    raise OpenAIRealtimeError(self._api_error_message(data))

                if event_type == "session.updated":
                    _LOGGER.debug("OpenAI Realtime transcription session configured")
                    continue

                if (
                    event_type
                    == "conversation.item.input_audio_transcription.delta"
                ):
                    item_id = data.get("item_id", "")
                    delta = data.get("delta", "")
                    if delta:
                        deltas_by_item.setdefault(item_id, []).append(delta)
                    continue

                if (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    if not commit_sent.is_set():
                        # This should be impossible with turn_detection=null, but do
                        # not let a stray event terminate the Voice PE microphone.
                        _LOGGER.warning(
                            "Ignoring transcription completion received before commit"
                        )
                        continue

                    item_id = data.get("item_id", "")
                    transcript = (data.get("transcript") or "").strip()
                    if not transcript:
                        transcript = "".join(deltas_by_item.get(item_id, [])).strip()

                    # A completion may arrive in the same receive() call during
                    # which commit_sent changed, before commit_time was initialized.
                    duration = (
                        time.perf_counter() - commit_time
                        if commit_time is not None
                        else 0.0
                    )
                    _LOGGER.debug(
                        'Final Realtime transcript after %.2f seconds: "%s"',
                        duration,
                        transcript,
                    )
                    return transcript

                continue

            if msg.type == WSMsgType.ERROR:
                raise OpenAIRealtimeError(f"WebSocket error: {ws.exception()}")

            if msg.type in (
                WSMsgType.CLOSE,
                WSMsgType.CLOSING,
                WSMsgType.CLOSED,
            ):
                raise OpenAIRealtimeError(
                    f"WebSocket closed before transcription completed "
                    f"(code={ws.close_code})"
                )

    async def _run_stream(self, ws, metadata, stream) -> str:
        """Run sender and receiver while propagating either task's errors."""
        commit_sent = asyncio.Event()
        send_task = asyncio.create_task(
            self._send_audio_stream(ws, metadata, stream, commit_sent)
        )
        receive_task = asyncio.create_task(
            self._receive_transcription(ws, commit_sent)
        )

        try:
            done, _ = await asyncio.wait(
                (send_task, receive_task),
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receive_task in done:
                # Usually an API/configuration error. Do not hide it as an empty
                # successful transcript.
                transcript = receive_task.result()
                await send_task
                return transcript

            output_bytes = send_task.result()
            if output_bytes == 0:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                return ""

            return await receive_task
        finally:
            for task in (send_task, receive_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, receive_task, return_exceptions=True)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an Assist audio stream through OpenAI Realtime transcription."""
        # Select the dedicated GA transcription session during the WebSocket
        # handshake. Without intent=transcription, OpenAI creates a normal
        # Realtime conversation session and rejects session.type=transcription.
        # No model belongs in this URL; the STT model is supplied in
        # session.audio.input.transcription.model.
        uri = f"{self.api_url}/realtime?intent=transcription"
        # The current /v1/realtime endpoint is GA. Sending the former
        # `OpenAI-Beta: realtime=v1` header forces the retired Beta API shape
        # and is rejected with `beta_api_shape_disabled`.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            _LOGGER.debug("Opening OpenAI Realtime WebSocket at %s", uri)
            async with self.client.ws_connect(
                uri,
                headers=headers,
                heartbeat=30,
            ) as ws:
                config = self._create_session_config(metadata.language)
                _LOGGER.debug("Sending Realtime session configuration: %s", config)
                await ws.send_json(config)

                final_text = (await self._run_stream(ws, metadata, stream)).strip()
                if not final_text:
                    _LOGGER.warning("WebSocket transcription resulted in empty text")
                    return SpeechResult("", SpeechResultState.SUCCESS)

                return SpeechResult(final_text, SpeechResultState.SUCCESS)

        except asyncio.CancelledError:
            raise
        except (ClientError, OpenAIRealtimeError) as err:
            _LOGGER.error("OpenAI Realtime transcription failed: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
        except Exception:
            _LOGGER.exception("Unexpected error in OpenAI Realtime transcription")
            return SpeechResult("", SpeechResultState.ERROR)
