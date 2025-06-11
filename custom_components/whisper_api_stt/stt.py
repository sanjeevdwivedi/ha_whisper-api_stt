"""
Support for Whisper API STT.
"""
from typing import AsyncIterable
import aiohttp
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
import homeassistant.helpers.config_validation as cv
import io


CONF_URL = 'url'

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_URL, default=None): cv.string,
    vol.Optional(CONF_PROMPT, default=None): cv.string,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up Whisper API STT speech component."""
    url = config.get(CONF_URL)
    return OpenAISTTProvider(hass, url)


class OpenAISTTProvider(Provider):
    """The Whisper API STT provider."""

    def __init__(self, hass, url):
        """Initialize Whisper API STT provider."""
        self.hass = hass
        self._prompt = prompt
        self._url = url

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return ["en"]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
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

    async def async_process_audio_stream(self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]) -> SpeechResult:
        import struct
        import io
        import logging
        _LOGGER = logging.getLogger(__name__)

        def convert_pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
            """Convert raw PCM audio data to WAV format."""
            wav_buffer = io.BytesIO()
            data_size = len(pcm_data)
            file_size = 36 + data_size

            wav_buffer.write(b'RIFF')
            wav_buffer.write(struct.pack('<I', file_size))
            wav_buffer.write(b'WAVE')
            wav_buffer.write(b'fmt ')
            wav_buffer.write(struct.pack('<I', 16))
            wav_buffer.write(struct.pack('<H', 1))
            wav_buffer.write(struct.pack('<H', channels))
            wav_buffer.write(struct.pack('<I', sample_rate))
            wav_buffer.write(struct.pack('<I', sample_rate * channels * sample_width))
            wav_buffer.write(struct.pack('<H', channels * sample_width))
            wav_buffer.write(struct.pack('<H', sample_width * 8))
            wav_buffer.write(b'data')
            wav_buffer.write(struct.pack('<I', data_size))
            wav_buffer.write(pcm_data)
            return wav_buffer.getvalue()

        data = b''
        async for chunk in stream:
            data += chunk

        if not data:
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            # Convert PCM to WAV in-memory
            wav_audio = convert_pcm_to_wav(
                data,
                sample_rate=metadata.sample_rate,
                channels=metadata.channel,
                sample_width=2
            )

            params = {
                'encode': 'true',
                'task': 'transcribe',
                'output': 'txt'
            }

            form = aiohttp.FormData()
            form.add_field(
                'audio_file',
                wav_audio,
                filename='recording.wav',
                content_type='audio/wav'
            )

            url = self._url or "http://sanjeev-debian-llm-vm.lan:9000/asr"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    data=form,
                    headers={'Accept': 'text/plain'}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(f"Transcription failed: {response.status} - {error_text}")
                        return SpeechResult("", SpeechResultState.ERROR)

                    transcribed_text = await response.text()
                    if not transcribed_text or not transcribed_text.strip():
                        _LOGGER.warning("Transcription returned empty text")
                        return SpeechResult("", SpeechResultState.ERROR)

                    return SpeechResult(transcribed_text.strip(), SpeechResultState.SUCCESS)

        except Exception as e:
            _LOGGER.error(f"Error during transcription: {e}")
            return SpeechResult("", SpeechResultState.ERROR)
