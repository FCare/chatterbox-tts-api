"""
Request models for API validation
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class QualityMode(str, Enum):
    """Quality vs Speed trade-off"""
    FAST = "fast"           # n_timesteps=3
    BALANCED = "balanced"   # n_timesteps=5
    QUALITY = "quality"     # n_timesteps=10


class TTSRequest(BaseModel):
    """Text-to-speech request model"""
    
    input: str = Field(..., description="The text to generate audio for", min_length=1, max_length=3000)
    voice: Optional[str] = Field("alloy", description="Voice to use (ignored - uses voice sample)")
    response_format: Optional[str] = Field("wav", description="Audio format (always returns WAV)")
    speed: Optional[float] = Field(1.0, description="Speed of speech (ignored)")
    stream_format: Optional[str] = Field("audio", description="Streaming format: 'audio' for raw audio stream, 'sse' for Server-Side Events")
    
    # Custom TTS parameters
    exaggeration: Optional[float] = Field(None, description="Emotion intensity", ge=0.25, le=2.0)
    cfg_weight: Optional[float] = Field(None, description="Pace control", ge=0.0, le=1.0)
    temperature: Optional[float] = Field(None, description="Sampling temperature", ge=0.05, le=5.0)
    
    # New parameters for chatterbox-multilingual
    quality_mode: Optional[QualityMode] = Field(default=QualityMode.BALANCED, description="Quality vs speed trade-off")
    stream_chunk_size: Optional[List[int]] = Field(default_factory=lambda: [20, 50, 100], description="Progressive chunk sizes for streaming")
    
    @validator('input')
    def validate_input(cls, v):
        if not v or not v.strip():
            raise ValueError('Input text cannot be empty')
        return v.strip()
    
    @validator('stream_format')
    def validate_stream_format(cls, v):
        if v is not None:
            allowed_formats = ['audio', 'sse']
            if v not in allowed_formats:
                raise ValueError(f'stream_format must be one of: {", ".join(allowed_formats)}')
        return v