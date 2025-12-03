"""
Audio processing utilities for long text TTS concatenation
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    from pydub.utils import make_chunks
    PYDUB_AVAILABLE = True
except ImportError as e:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    # Log the import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"pydub import failed: {e}")
except Exception as e:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    # Log any other errors for debugging
    import logging
    logging.getLogger(__name__).error(f"Unexpected error importing pydub: {e}")

from app.config import Config

logger = logging.getLogger(__name__)


class AudioConcatenationError(Exception):
    """Exception raised when audio concatenation fails"""
    pass


def check_pydub_availability():
    """Check if pydub is available and properly configured"""
    if not PYDUB_AVAILABLE:
        raise AudioConcatenationError(
            "pydub is not available. Please install it with: pip install pydub"
        )

    # Test basic functionality
    try:
        # Create a small test audio segment
        test_audio = AudioSegment.silent(duration=100)  # 100ms of silence
        return True
    except Exception as e:
        raise AudioConcatenationError(f"pydub is not properly configured: {e}")


def create_silence_audio(duration_ms: int,
                        sample_rate: int = 22050,
                        channels: int = 1,
                        output_path: Optional[Union[str, Path]] = None,
                        output_format: str = "wav") -> Optional[str]:
    """
    Create a silence audio file of specified duration.

    Args:
        duration_ms: Duration of silence in milliseconds
        sample_rate: Sample rate for the audio
        channels: Number of audio channels
        output_path: Path to save the silence file (optional)
        output_format: Format for the output file

    Returns:
        Path to the created silence file if output_path is specified, None otherwise
    """
    check_pydub_availability()

    try:
        silence = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=sample_rate
        ).set_channels(channels)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            silence.export(str(output_path), format=output_format)

            return str(output_path)

        return None

    except Exception as e:
        raise AudioConcatenationError(f"Failed to create silence audio: {e}")


def validate_audio_file(file_path: Union[str, Path]) -> dict:
    """
    Validate and get metadata for an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary with audio file metadata:
        {
            'valid': bool,
            'duration_seconds': float,
            'sample_rate': int,
            'channels': int,
            'format': str,
            'file_size_bytes': int,
            'error': str (if valid=False)
        }
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {'valid': False, 'error': f'File not found: {file_path}'}

    try:
        check_pydub_availability()

        # Load the audio file
        audio = AudioSegment.from_file(str(file_path))

        return {
            'valid': True,
            'duration_seconds': len(audio) / 1000.0,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'format': file_path.suffix.lower().lstrip('.'),
            'file_size_bytes': file_path.stat().st_size,
            'error': None
        }

    except Exception as e:
        return {'valid': False, 'error': str(e)}


def estimate_concatenation_time(num_files: int, total_duration_seconds: float) -> int:
    """
    Estimate the time required to concatenate audio files.

    Args:
        num_files: Number of files to concatenate
        total_duration_seconds: Total duration of all audio files

    Returns:
        Estimated processing time in seconds
    """
    # Base processing time: 0.1 seconds per second of audio
    base_time = total_duration_seconds * 0.1

    # File I/O overhead: 1 second per file
    io_overhead = num_files * 1

    # Additional overhead for format conversion, normalization, etc.
    processing_overhead = 5

    return max(10, int(base_time + io_overhead + processing_overhead))