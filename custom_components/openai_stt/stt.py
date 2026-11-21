"""Setting up OpenAISTTProvider."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging
import math

import voluptuous as vol

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv

from .audio_processing import async_process_pcm_stream
from .http_client import OpenAIHTTPClient
from .websocket_client import OpenAIWebSocketClient

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_LANGUAGE = "language"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"
CONF_REALTIME = "realtime"
CONF_NOISE_REDUCTION = "noise_reduction"
CONF_DEBUG_SAVE = "debug_save"
CONF_PITCH_UP_VOCODER = "pitch_up_vocoder"
CONF_PITCH_SEMITONES = "pitch_semitones"
CONF_VOCODER_STRENGTH = "vocoder_strength"

DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_LANGUAGE = None
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0
DEFAULT_REALTIME = False
DEFAULT_NOISE_REDUCTION = None
DEFAULT_DEBUG_SAVE = False
DEFAULT_PITCH_UP_VOCODER = False
DEFAULT_PITCH_SEMITONES = 1.0
DEFAULT_VOCODER_STRENGTH = 0.15

SUPPORTED_MODELS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gpt-live-transcribe",
    "gpt-transcribe",
]

SUPPORTED_NOISE_REDUCTION = [
    None,
    "near_field",
    "far_field",
]

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)
LANGUAGE_SCHEMA = vol.Any(None, vol.In(SUPPORTED_LANGUAGES))
NOISE_REDUCTION_SCHEMA = vol.In(SUPPORTED_NOISE_REDUCTION)


def _finite_float(value: float) -> float:
    """Reject YAML .nan/.inf values, including NaN that evades range checks."""
    if not math.isfinite(value):
        raise vol.Invalid("value must be finite")
    return value


PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
        vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): LANGUAGE_SCHEMA,
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
        vol.Optional(CONF_REALTIME, default=DEFAULT_REALTIME): cv.boolean,
        vol.Optional(
            CONF_NOISE_REDUCTION, default=DEFAULT_NOISE_REDUCTION
        ): NOISE_REDUCTION_SCHEMA,
        vol.Optional(CONF_DEBUG_SAVE, default=DEFAULT_DEBUG_SAVE): cv.boolean,
        vol.Optional(
            CONF_PITCH_UP_VOCODER, default=DEFAULT_PITCH_UP_VOCODER
        ): cv.boolean,
        vol.Optional(CONF_PITCH_SEMITONES, default=DEFAULT_PITCH_SEMITONES): vol.All(
            vol.Coerce(float), vol.Range(min=-12.0, max=12.0), _finite_float
        ),
        vol.Optional(CONF_VOCODER_STRENGTH, default=DEFAULT_VOCODER_STRENGTH): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=1.0), _finite_float
        ),
    }
)


async def async_get_engine(
    hass: HomeAssistant, config: dict, discovery_info: dict | None = None
) -> OpenAISTTProvider:
    """Return the OpenAI STT provider."""
    api_key = config[CONF_API_KEY]
    api_url = config.get(CONF_API_URL, DEFAULT_API_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature = config.get(CONF_TEMP, DEFAULT_TEMP)
    realtime = config.get(CONF_REALTIME, DEFAULT_REALTIME)
    noise_reduction = config.get(CONF_NOISE_REDUCTION, DEFAULT_NOISE_REDUCTION)

    return OpenAISTTProvider(
        hass,
        api_key,
        api_url,
        model,
        prompt,
        temperature,
        realtime,
        noise_reduction,
        language=config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
        debug_save=config.get(CONF_DEBUG_SAVE, DEFAULT_DEBUG_SAVE),
        pitch_up_vocoder=config.get(CONF_PITCH_UP_VOCODER, DEFAULT_PITCH_UP_VOCODER),
        pitch_semitones=config.get(CONF_PITCH_SEMITONES, DEFAULT_PITCH_SEMITONES),
        vocoder_strength=config.get(CONF_VOCODER_STRENGTH, DEFAULT_VOCODER_STRENGTH),
    )


class OpenAISTTProvider(Provider):
    """The OpenAI STT provider."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_key: str,
        api_url: str,
        model: str,
        prompt: str,
        temperature: int,
        realtime: bool,
        noise_reduction: str | None,
        *,
        language: str | None = DEFAULT_LANGUAGE,
        debug_save: bool = DEFAULT_DEBUG_SAVE,
        pitch_up_vocoder: bool = DEFAULT_PITCH_UP_VOCODER,
        pitch_semitones: float = DEFAULT_PITCH_SEMITONES,
        vocoder_strength: float = DEFAULT_VOCODER_STRENGTH,
    ) -> None:
        """Init OpenAI STT service."""
        self.hass = hass
        self.name = "OpenAI STT"

        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._language = language
        self._prompt = prompt
        self._temperature = temperature
        self._realtime = realtime
        self._noise_reduction = noise_reduction
        self._pitch_up_vocoder = pitch_up_vocoder
        self._pitch_semitones = pitch_semitones
        self._vocoder_strength = vocoder_strength
        self._debug_directory = (
            hass.config.path("openai_stt_debug") if debug_save else None
        )
        self._client = self._create_client()
        if debug_save:
            _LOGGER.warning(
                "OpenAI STT debug_save is enabled: outgoing speech is saved to %s. "
                "Recordings contain sensitive content and must be deleted manually",
                self._debug_directory,
            )

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        # Assist supplies raw PCM for WAV/PCM metadata. Neither transport
        # decodes OGG/Opus, so do not advertise compressed audio support.
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    def _create_client(self):
        """Create and return the appropriate client based on configuration."""
        if self._realtime:
            # Use WebSocket client for OpenAI Realtime API
            return OpenAIWebSocketClient(
                async_get_clientsession(self.hass),
                self._api_key,
                self._api_url,
                self._model,
                self._prompt,
                self._noise_reduction,
                language=self._language,
                debug_directory=self._debug_directory,
                executor=self.hass.async_add_executor_job,
            )

        # Use HTTP client for OpenAI Transcription API
        return OpenAIHTTPClient(
            async_get_clientsession(self.hass),
            self._api_key,
            self._api_url,
            self._model,
            self._prompt,
            self._temperature,
            language=self._language,
            debug_directory=self._debug_directory,
            executor=self.hass.async_add_executor_job,
        )

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process audio stream using the configured method (HTTP or WebSocket)."""
        _LOGGER.debug(
            "Processing audio stream with %s", self._client.__class__.__name__
        )
        if (
            metadata.format != AudioFormats.WAV
            or metadata.codec != AudioCodecs.PCM
            or int(metadata.channel) != 1
            or int(metadata.bit_rate) != 16
            or int(metadata.sample_rate) != 16000
        ):
            _LOGGER.error("OpenAI STT requires raw mono PCM16 at 16000 Hz")
            return SpeechResult("", SpeechResultState.ERROR)

        if self._pitch_up_vocoder:
            stream = async_process_pcm_stream(
                stream,
                int(metadata.sample_rate),
                self._pitch_semitones,
                self._vocoder_strength,
                self.hass.async_add_executor_job,
            )
        try:
            return await self._client.async_process_audio_stream(metadata, stream)
        except Exception:
            # Never fall back to uploading the original voice after a DSP error.
            _LOGGER.exception("Failed to prepare/process OpenAI STT audio")
            return SpeechResult("", SpeechResultState.ERROR)
