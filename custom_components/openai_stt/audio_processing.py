"""Local, streaming PCM16 voice effects (not a training-prevention mechanism)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
import math
from typing import Any

import numpy as np

Executor = Callable[..., Awaitable[Any]]

# Bound each executor job even if the source supplies a whole recording at once.
MAX_JOB_BYTES = 8192


class PCM16VoiceProcessor:
    """Duration-preserving phase-vocoder pitch shift and spectral noise vocoder.

    Input/output are mono, signed little-endian PCM16 at the same sample rate.
    Analysis, phase and overlap-add state belong to ONE utterance and survive
    arbitrary chunk boundaries. About 64 ms of lookahead is needed at 16 kHz.
    flush() drains that lookahead without adding silence to the recording.

    The optional noise vocoder mixes a speech-shaped noise carrier with the
    pitched signal. It changes timbre, but does not anonymize speech or make it
    unusable for training. In particular, the spoken content remains present.
    """

    def __init__(
        self,
        sample_rate: int,
        pitch_semitones: float = 1.0,
        vocoder_strength: float = 0.15,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not math.isfinite(pitch_semitones) or not -12 <= pitch_semitones <= 12:
            raise ValueError("pitch_semitones must be finite and between -12 and 12")
        if not math.isfinite(vocoder_strength) or not 0 <= vocoder_strength <= 1:
            raise ValueError("vocoder_strength must be finite and between 0 and 1")

        self._pitch_factor = 2.0 ** (pitch_semitones / 12.0)
        self._strength = vocoder_strength
        # Nearest power of two to 64 ms; 1024 samples at HA's 16 kHz.
        self._size = max(256, 2 ** round(math.log2(sample_rate * 0.064)))
        self._hop = self._size // 4
        self._window = np.hanning(self._size)
        self._window_squared = self._window**2
        self._skip = self._size - self._hop
        self._pending = np.zeros(self._skip, dtype=np.float64)
        self._overlap = np.zeros(self._size, dtype=np.float64)
        self._weights = np.zeros(self._size, dtype=np.float64)
        self._bins = np.arange(self._size // 2 + 1, dtype=np.float64)
        self._omega = 2 * np.pi * self._bins / self._size
        self._previous_phase: np.ndarray | None = None
        self._synthesis_phase = np.zeros_like(self._bins)
        self._rng = np.random.default_rng()
        self._pending_byte = b""
        self._input_samples = 0
        self._output_samples = 0
        self._closed = False

        # Spread mapped energy between adjacent bins to avoid spectral holes.
        destinations = self._bins * self._pitch_factor
        self._lower = np.floor(destinations).astype(np.int64)
        self._fraction = destinations - self._lower
        self._lower_valid = self._lower < len(self._bins)
        self._upper_valid = self._lower + 1 < len(self._bins)

        # Smooth power over roughly 250 Hz for the noise carrier's envelope.
        envelope_bins = max(3, round(250 * self._size / sample_rate))
        envelope_bins = min(envelope_bins | 1, len(self._bins))
        self._envelope_kernel = np.ones(envelope_bins) / envelope_bins

    def _pitch_shift(self, spectrum: np.ndarray) -> np.ndarray:
        """Map magnitudes and instantaneous frequencies, preserving duration."""
        if self._pitch_factor == 1.0:
            return spectrum

        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        if self._previous_phase is None:
            frequency = self._omega
        else:
            delta = phase - self._previous_phase - self._omega * self._hop
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            frequency = self._omega + delta / self._hop

        shifted_magnitude = np.zeros_like(magnitude)
        frequency_sum = np.zeros_like(magnitude)
        for offset, weight, valid in (
            (0, 1.0 - self._fraction, self._lower_valid),
            (1, self._fraction, self._upper_valid),
        ):
            indices = self._lower[valid] + offset
            contribution = magnitude[valid] * weight[valid]
            np.add.at(shifted_magnitude, indices, contribution)
            np.add.at(
                frequency_sum,
                indices,
                contribution * frequency[valid] * self._pitch_factor,
            )

        shifted_frequency = np.divide(
            frequency_sum,
            shifted_magnitude,
            out=self._omega.copy(),
            where=shifted_magnitude > 1e-12,
        )
        if self._previous_phase is None:
            self._synthesis_phase = np.interp(
                self._bins / self._pitch_factor, self._bins, phase, right=0.0
            )
        else:
            self._synthesis_phase += shifted_frequency * self._hop
            self._synthesis_phase = (
                self._synthesis_phase + np.pi
            ) % (2 * np.pi) - np.pi
        self._previous_phase = phase
        return shifted_magnitude * np.exp(1j * self._synthesis_phase)

    def _process_frame(self) -> bytes:
        """Consume one analysis hop and return finalized output samples."""
        frame = self._pending[: self._size]
        spectrum = self._pitch_shift(np.fft.rfft(frame * self._window))
        if self._strength:
            envelope = np.sqrt(
                np.convolve(
                    np.abs(spectrum) ** 2, self._envelope_kernel, mode="same"
                )
            )
            carrier = envelope * np.exp(
                1j * self._rng.uniform(-np.pi, np.pi, len(envelope))
            )
            carrier[0] = carrier[-1] = 0.0
            spectrum = (1.0 - self._strength) * spectrum + self._strength * carrier

        self._overlap += np.fft.irfft(spectrum, n=self._size) * self._window
        self._weights += self._window_squared
        block = np.divide(
            self._overlap[: self._hop],
            self._weights[: self._hop],
            out=np.zeros(self._hop),
            where=self._weights[: self._hop] > 1e-12,
        )
        self._overlap[: -self._hop] = self._overlap[self._hop :]
        self._overlap[-self._hop :] = 0.0
        self._weights[: -self._hop] = self._weights[self._hop :]
        self._weights[-self._hop :] = 0.0
        self._pending = self._pending[self._hop :]

        if self._skip:
            skip = min(self._skip, len(block))
            block = block[skip:]
            self._skip -= skip
        # Final zero padding is analysis context, never extra output audio.
        count = min(len(block), self._input_samples - self._output_samples)
        self._output_samples += count
        block = block[:count]
        return np.clip(np.rint(block * 32768.0), -32768, 32767).astype("<i2").tobytes()

    def process(self, pcm: bytes) -> bytes:
        """Process a chunk, retaining partial samples and frame/phase history."""
        if self._closed:
            raise ValueError("Audio processor has already been flushed")
        pcm = self._pending_byte + pcm
        complete_length = len(pcm) - len(pcm) % 2
        self._pending_byte = pcm[complete_length:]
        if not complete_length:
            return b""
        samples = np.frombuffer(pcm[:complete_length], dtype="<i2").astype(np.float64)
        samples /= 32768.0
        self._input_samples += len(samples)
        self._pending = np.concatenate((self._pending, samples))
        output = []
        while len(self._pending) >= self._size:
            output.append(self._process_frame())
        return b"".join(output)

    def flush(self) -> bytes:
        """Finish the recording; never silently discard a truncated PCM sample."""
        if self._closed:
            return b""
        self._closed = True
        if self._pending_byte:
            raise ValueError("Truncated PCM16 input: an unmatched final byte remains")
        output = []
        while self._output_samples < self._input_samples:
            if len(self._pending) < self._size:
                self._pending = np.pad(
                    self._pending, (0, self._size - len(self._pending))
                )
            output.append(self._process_frame())
        return b"".join(output)


async def async_process_pcm_stream(
    stream: AsyncIterable[bytes],
    sample_rate: int,
    pitch_semitones: float,
    vocoder_strength: float,
    executor: Executor | None = None,
) -> AsyncIterator[bytes]:
    """Run DSP off the event loop, with independent state for each utterance."""
    run = executor or asyncio.to_thread
    processor = await run(
        PCM16VoiceProcessor, sample_rate, pitch_semitones, vocoder_strength
    )
    async for chunk in stream:
        for start in range(0, len(chunk), MAX_JOB_BYTES):
            output = await run(processor.process, chunk[start : start + MAX_JOB_BYTES])
            if output:
                yield output
    output = await run(processor.flush)
    if output:
        yield output
