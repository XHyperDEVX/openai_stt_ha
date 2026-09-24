"""DSP regression tests; run with unittest discover, without Home Assistant."""

import asyncio
import threading
import unittest

import numpy as np

from custom_components.openai_stt.audio_processing import (
    PCM16VoiceProcessor,
    async_process_pcm_stream,
)


def tone(sample_count=32000, frequency=440.0):
    time = np.arange(sample_count) / 16000
    return np.rint(9000 * np.sin(2 * np.pi * frequency * time)).astype("<i2").tobytes()


def process(pcm, pitch=1.0, strength=0.15, chunk_size=777):
    processor = PCM16VoiceProcessor(16000, pitch, strength)
    # The same carrier across runs makes chunk-invariance testable.
    processor._rng = np.random.default_rng(123)
    output = [
        processor.process(pcm[start : start + chunk_size])
        for start in range(0, len(pcm), chunk_size)
    ]
    output.append(processor.flush())
    return b"".join(output)


class VoiceProcessorTests(unittest.TestCase):
    def test_neutral_effect_reconstructs_pcm_exactly(self):
        rng = np.random.default_rng(42)
        for count in (0, 1, 255, 256, 257, 1023, 1024, 16003):
            with self.subTest(count=count):
                pcm = rng.integers(-32768, 32768, count, dtype=np.int16)
                pcm = pcm.astype("<i2").tobytes()
                self.assertEqual(process(pcm, pitch=0, strength=0), pcm)

    def test_default_effect_preserves_duration_and_changes_signal(self):
        pcm = tone()
        output = process(pcm)
        self.assertEqual(len(output), len(pcm))
        self.assertNotEqual(output, pcm)
        self.assertGreater(np.max(np.abs(np.frombuffer(output, "<i2"))), 100)

    def test_silence_stays_silent_and_short_tail_is_retained(self):
        for count in (0, 1, 17, 255, 1023, 16001):
            with self.subTest(count=count):
                pcm = b"\x00\x00" * count
                self.assertEqual(process(pcm), pcm)

    def test_chunk_boundaries_do_not_change_output(self):
        pcm = tone(4501)
        whole = process(pcm, chunk_size=len(pcm))
        for chunk_size in (1, 7, 512, 777, 4096):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(process(pcm, chunk_size=chunk_size), whole)

    def test_pitch_shift_changes_frequency_not_speed(self):
        pcm = tone()
        for semitones in (-3.0, 1.0, 3.0):
            with self.subTest(semitones=semitones):
                output = process(pcm, pitch=semitones, strength=0)
                self.assertEqual(len(output), len(pcm))
                # Ignore initial/final phase-vocoder transients.
                samples = np.frombuffer(output, "<i2")[4000:-4000].astype(float)
                spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
                frequency = np.fft.rfftfreq(len(samples), 1 / 16000)[spectrum.argmax()]
                self.assertAlmostEqual(frequency, 440 * 2 ** (semitones / 12), delta=4)

    def test_vocoder_works_without_pitch_shift(self):
        pcm = tone(4000)
        self.assertNotEqual(process(pcm, pitch=0, strength=0.15), pcm)
        self.assertEqual(len(process(pcm, pitch=0, strength=1)), len(pcm))

    def test_invalid_parameters_and_truncated_sample(self):
        for pitch, strength in (
            (13, 0), (-13, 0), (float("nan"), 0), (float("inf"), 0),
            (0, -0.1), (0, 1.1), (0, float("nan")), (0, float("inf")),
        ):
            with self.subTest(pitch=pitch, strength=strength):
                with self.assertRaises(ValueError):
                    PCM16VoiceProcessor(16000, pitch, strength)
        processor = PCM16VoiceProcessor(16000)
        processor.process(b"\x01")
        with self.assertRaisesRegex(ValueError, "unmatched final byte"):
            processor.flush()

    def test_flush_is_idempotent_and_processor_cannot_be_reused(self):
        processor = PCM16VoiceProcessor(16000)
        processor.process(tone(17))
        self.assertEqual(len(processor.flush()), 34)
        self.assertEqual(processor.flush(), b"")
        with self.assertRaises(ValueError):
            processor.process(b"\x00\x00")


class AsyncProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_is_used_and_streaming_does_not_wait_for_end(self):
        main_thread = threading.get_ident()
        worker_threads = []
        exhausted = False

        async def executor(func, *args):
            def work():
                worker_threads.append(threading.get_ident())
                return func(*args)
            return await asyncio.to_thread(work)

        async def source():
            nonlocal exhausted
            yield tone(16000)
            exhausted = True

        transformed = async_process_pcm_stream(source(), 16000, 1.0, 0.15, executor)
        first = await anext(transformed)
        self.assertTrue(first)
        self.assertFalse(exhausted)
        output = first + b"".join([chunk async for chunk in transformed])
        self.assertEqual(len(output), 32000)
        self.assertTrue(worker_threads)
        self.assertNotIn(main_thread, worker_threads)


if __name__ == "__main__":
    unittest.main()
