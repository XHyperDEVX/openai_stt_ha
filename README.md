# OpenAI Speech-To-Text for Home Assistant

This custom component integrates [OpenAI Speech-to-Text](https://platform.openai.com/docs/guides/speech-to-text), also known as Whisper, into Home Assistant via the OpenAI API for use in the Assist pipeline.

## Installation

### HACS (Recommended)


This integration is part of the standard HACS repository. Just search for "OpenAI Whisper API" to install it, or use this link to go directly there:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=einToast&repository=openai_stt_ha&category=integration)


### Manual

1. Inside your `config` directory, create a new directory named `custom_components`
2. Create a new directory named `openai_stt` inside the `custom_components` directory
3. Place all the files from this repository in the `openai_stt` directory
4. Restart Home Assistant

## Configuration

You need to create an account on the [OpenAI website](https://platform.openai.com/signup) and get an [API key](https://platform.openai.com/api-keys).
Then add the following to your `configuration.yaml` and restart Home Assistant:

```yaml
stt:
  - platform: openai_stt
    api_key: YOUR_API_KEY
    #  Optional parameters
    realtime: false
    api_url: https://api.openai.com/v1
    model: gpt-4o-mini-transcribe
    language: de
    prompt: ""
    temperature: 0
    noise_reduction: null
    debug_save: false
    pitch_up_vocoder: false
    pitch_semitones: 1.0
    vocoder_strength: 0.15
```

### Parameters:

- `api_key` (Required): Your OpenAI API key
- `realtime` (Optional): If set to `true`, the integration will use the OpenAI Realtime API. This should generate faster results. If set to `false`, the integration will use the regular OpenAI Transcription API. The default is `false`. Keep in mind that the Realtime API is currently in beta and may not be as stable as the Transcription API. See the [OpenAI documentation](https://platform.openai.com/docs/guides/realtime-transcription) for more information
- `api_url` (Optional): The API URL to use. Specify this to use any compatible OpenAI API. The default is `https://api.openai.com/v1`.
- `model` (Optional): The model to use. Currently, the [supported models](#supported-models) are `gpt-4o-mini-transcribe`, `gpt-4o-transcribe` and `whisper-1`. The default is `gpt-4o-mini-transcribe`. All available models are listed in the [OpenAI model list](https://platform.openai.com/docs/models) under the Transcription section
- `language` (Optional): Fix the input language used for transcription with a supported lowercase ISO 639-1 code, e.g. `de` for German or `en` for English (not `de-DE` or `German`). This overrides the language supplied by the Home Assistant Assist pipeline. Omit it or set it to `null` to keep using the pipeline language (the default). Supported by HTTP transcription and by the Realtime model `gpt-live-transcribe` (including the aliases `gpt-4o-mini-transcribe` and `whisper-1`). The Realtime model `gpt-transcribe` (including the alias `gpt-4o-transcribe`) detects the language automatically and does not accept this hint; an explicitly configured `language` logs a warning for that model. This is an API language hint, not translation or a guaranteed filter that rejects speech in other languages.
- `prompt` (Optional): The prompt to use. The default is an empty string. See the [OpenAI documentation](https://platform.openai.com/docs/guides/speech-to-text#prompting) for more information
- `temperature` (Optional): The temperature to use between `0` and `1`. A higher temperature will make the model more creative, but less accurate. The default is `0`. Only applicable when `realtime: false`
- `noise_reduction` (Optional): The noise reduction to use. The available options are `null`, `near_field` and `far_field`. `near_field` is for close-range audio, `far_field` is for distant audio, `null` turns off noise reduction. The default is `null`. Only applicable when `realtime: true`

### Local voice processing and debug recordings

These options work with **both HTTP and Realtime transcription**:

| Option | Default | Meaning |
| --- | --- | --- |
| `debug_save` | `false` | Save the outgoing audio locally as a WAV file, after voice processing and any transport sample-rate conversion. Works independently of `pitch_up_vocoder`. |
| `pitch_up_vocoder` | `false` | Enable local pitch shifting and a mild spectral noise vocoder. If disabled, `pitch_semitones` and `vocoder_strength` have no effect. |
| `pitch_semitones` | `1.0` | Pitch shift in semitones, from `-12.0` to `12.0`. Positive values raise pitch; negative values lower it. `0` disables the pitch shift, but not the vocoder. |
| `vocoder_strength` | `0.15` | Noise-vocoder mix from `0.0` (pitched voice only) to `1.0` (noise-vocoder only). High values can significantly reduce intelligibility. |

For example:

```yaml
stt:
  - platform: openai_stt
    api_key: !secret openai_api_key
    realtime: true
    api_url: https://api.openai.com/v1
    model: gpt-transcribe
    prompt: ""
    temperature: 0
    noise_reduction: null
    debug_save: true
    pitch_up_vocoder: true
    pitch_semitones: 1.0
    vocoder_strength: 0.15
```

The local processor uses a streaming phase vocoder to shift pitch without changing
speech speed or the number of samples. A speech-shaped noise carrier is mixed in
at the configured strength to alter timbre. State is retained between chunks and
is separate for every utterance. DSP and debug file I/O run in Home Assistant's
executor, not on its event loop. Realtime stays streaming, with about 64 ms of
algorithmic lookahead at 16 kHz (plus processing/scheduling time). There is no
external audio service or FFmpeg requirement; Home Assistant installs the NumPy
requirement from `manifest.json`. The requirement accepts `numpy>=1.26.0,<3.0.0`
instead of forcing NumPy 2.4.0, allowing Home Assistant's own package constraints
and the Python/platform environment to determine a compatible version.

If setup reports `Requirements for openai_stt not found`, check the preceding
package-installation error in the Home Assistant logs. The summary alone does
not distinguish a version conflict from an unavailable wheel, a download error,
or another installation failure. Do not force-upgrade NumPy in Home Assistant's
managed environment. When updating from version 1.4.0, copy the updated
`custom_components/openai_stt/manifest.json` and fully restart Home Assistant;
no YAML change is needed for this dependency fix.

The input is Home Assistant Assist's **raw mono PCM16 at 16 kHz**, advertised as
WAV/PCM metadata. Encoded OGG/Opus is not supported and is no longer advertised
(the original transports did not decode it). For HTTP, the processed PCM is put
into a WAV file. For Realtime, it is subsequently resampled to OpenAI's 24 kHz
PCM format. A processing error aborts transcription; it never triggers a fallback
to uploading the original voice.

#### Debug files

Files are saved in **`<Home Assistant config directory>/openai_stt_debug/`**,
usually `/config/openai_stt_debug/`. This is deliberately outside `www/`, so the
recordings are not exposed through Home Assistant's public static-file route.
Names contain a UTC timestamp, the transport (`http` or `realtime`), and a UUID.
New directories/files use restrictive permissions (`0700`/`0600`).

- **HTTP:** the file is byte-for-byte the same WAV payload supplied to the API.
  It is written before the HTTP request, so it may also exist if upload fails.
- **Realtime:** the WAV's audio frames are exactly the concatenation of the PCM
  chunks successfully handed to the WebSocket, **after** resampling to 24 kHz.
  Only the WAV header is added locally. This captures what the client sends, not
  OpenAI's later server-side noise reduction. An interrupted stream may leave a
  partial capture; a successful local send is not proof of server receipt.
- Recording is opt-in. Realtime buffers the outgoing capture in memory until the
  stream ends; normal streaming without `debug_save` does not do this.
- Disk-write errors are logged and do not stop transcription.
- **No automatic retention/cleanup:** these files contain sensitive speech, can
  fill your disk, and may be included in backups. Delete them manually and turn
  `debug_save` off after testing.

#### Privacy limitations — important

**Pitch shifting and a vocoder do not make intelligible audio unsuitable for
training, do not reliably anonymize a speaker, and are not an anti-training
protection.** Even altered speech can be used for speech-recognition training,
and speaker characteristics can remain detectable or partly recoverable. The
spoken content is still sent to the configured API. The mild settings above are
intended to retain intelligibility, not to provide a privacy guarantee; actual
transcription quality must be checked with your own recordings.

For control over training use, use the provider's contractual/data-use controls
and check its current retention policy. OpenAI states that API data is not used
for model training by default unless you opt in; that is separate from storage
or abuse-monitoring retention, and third-party `api_url` providers may differ.
For speech that must never leave your system, use an entirely local STT engine.

After updating the component files and YAML configuration, **restart Home
Assistant**. Existing installations have both new switches disabled by default.

### Development checks

The tests cover PCM duration, neutral reconstruction, pitch, chunk boundaries,
validation, configured language overrides and pipeline-language fallback for both
transports, outgoing debug payloads, and failure handling. Transport/provider
tests use small Home Assistant stubs and do not contact OpenAI; they are not a
substitute for an Assist end-to-end test or an intelligibility evaluation.

```sh
python -m pip install -r tests/requirements.txt
python -m unittest discover -s tests -v
```

## Supported Models
See the accuracy comparison of the models [here](https://openai.com/index/introducing-our-next-generation-audio-models/).

- `gpt-4o-mini-transcribe`: model optimized for speed and cost. Cost: estimated `$0.003` per minute of audio
- `gpt-4o-transcribe`: model optimized for accuracy. Cost: estimated `$0.006` per minute of audio
- `whisper-1`: original `whisper-large-v2` model. Superseded by `gpt-4o-mini-transcribe` and `gpt-4o-transcribe`. Cost: `$0.006` per minute of audio

## Error

If you get the following error in the Home Assistant system log:

```
The stt integration does not support any configuration parameters, got [{'platform': 'openai_stt', 'api_key': 'YOUR_API_KEY'}]. Please remove the configuration parameters from your configuration.
```

This issues is a known [bug](https://github.com/home-assistant/core/issues/97161) in Home Assistant >= 2023.7. The reported message does not affect the functionality of this integration, it should still work.