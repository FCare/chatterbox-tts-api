"""
Text-to-speech endpoint
"""

import io
import os
import asyncio
import tempfile
import torch
import torchaudio as ta
import base64
import json
import struct
from typing import Optional, List, Dict, Any, AsyncGenerator
from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile, Request
from fastapi.responses import StreamingResponse

from app.models import TTSRequest, ErrorResponse
from app.config import Config
from app.core import (
    split_text_into_chunks, concatenate_audio_chunks, add_route_aliases,
    TTSStatus, start_tts_request, update_tts_status, get_voice_library
)
from app.core.tts_model import get_model, is_multilingual

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


# Supported audio formats for voice uploads
SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}


def create_wav_header(sample_rate: int, channels: int, bits_per_sample: int, data_size: int = 0xFFFFFFFF) -> bytes:
    """Creates a WAV header for streaming."""
    header = io.BytesIO()
    header.write(b'RIFF')
    # Use a large, but not max, value for chunk size to avoid overflow issues in some players
    chunk_size = 36 + data_size if data_size != 0xFFFFFFFF else 0x7FFFFFFF - 36
    header.write(struct.pack('<I', chunk_size))
    header.write(b'WAVE')
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))  # Subchunk1Size for PCM
    header.write(struct.pack('<H', 1))   # AudioFormat (1 for PCM)
    header.write(struct.pack('<H', channels))
    header.write(struct.pack('<I', sample_rate))
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    header.write(struct.pack('<I', byte_rate))
    block_align = channels * (bits_per_sample // 8)
    header.write(struct.pack('<H', block_align))
    header.write(struct.pack('<H', bits_per_sample))
    header.write(b'data')
    header.write(struct.pack('<I', data_size)) # Subchunk2Size
    return header.getvalue()


def resolve_voice_path_and_language(voice_name: Optional[str]) -> tuple[str, str]:
    """
    Resolve a voice name or alias to a file path and language.
    
    Args:
        voice_name: Voice name or alias from the request (can be None for default)
        
    Returns:
        Tuple of (path to the voice file, language code)
    """
    # If no voice specified, use default
    if not voice_name:
        return Config.VOICE_SAMPLE_PATH, "en"
    
    # Try to resolve from voice library (handles both names and aliases)
    voice_lib = get_voice_library()
    voice_path = voice_lib.get_voice_path(voice_name)
    voice_language = voice_lib.get_voice_language(voice_name)
    
    if voice_path is None:
        # Check if it's an OpenAI voice name without an alias mapping
        openai_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
        if voice_name.lower() in openai_voices:
            print(f"🎵 Using default voice for OpenAI voice '{voice_name}' (no alias mapping)")
            return Config.VOICE_SAMPLE_PATH, "en"
        
        # Voice not found, fall back to default voice and log a warning
        print(f"⚠️ Warning: Voice '{voice_name}' not found in voice library, using default voice")
        return Config.VOICE_SAMPLE_PATH, "en"
    
    return voice_path, voice_language or "en"


def resolve_voice_path(voice_name: Optional[str]) -> str:
    """
    Resolve a voice name or alias to a file path (backward compatibility).
    
    Args:
        voice_name: Voice name or alias from the request (can be None for default)
        
    Returns:
        Path to the voice file (falls back to default if voice not found)
    """
    path, _ = resolve_voice_path_and_language(voice_name)
    return path


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "No filename provided", "type": "invalid_request_error"}}
        )
    
    # Check file extension
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
                    "type": "invalid_request_error"
                }
            }
        )
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file, 'size') and file.size and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"File too large. Maximum size: {max_size // (1024*1024)}MB",
                    "type": "invalid_request_error"
                }
            }
        )


async def generate_speech_internal(
    text: str,
    voice_sample_path: str,
    language_id: str = "en",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    quality_mode: str = "balanced",
    stream_chunk_size: Optional[List[int]] = None
) -> io.BytesIO:
    """Internal function to generate speech with given parameters (non-streaming only)"""
    
    # Start TTS request tracking
    voice_source = "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "default"
    request_id = start_tts_request(
        text=text,
        voice_source=voice_source,
        parameters={
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "voice_sample_path": voice_sample_path
        }
    )
    
    update_tts_status(request_id, TTSStatus.INITIALIZING, "Checking model availability")
    
    model = get_model()
    if model is None:
        update_tts_status(request_id, TTSStatus.ERROR, error_message="Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Model not loaded", "type": "model_error"}}
        )
    
    # Always use multilingual model
    # Default to English if no language_id specified
    if not language_id:
        language_id = "en"

    
    # Validate total text length
    update_tts_status(request_id, TTSStatus.PROCESSING_TEXT, "Validating text length")
    if len(text) > Config.MAX_TOTAL_LENGTH:
        update_tts_status(request_id, TTSStatus.ERROR, 
                        error_message=f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error"
                }
            }
        )

    audio_chunks = []
    final_audio = None
    buffer = None
    
    try:
        # Get parameters with defaults
        exaggeration = exaggeration if exaggeration is not None else Config.EXAGGERATION
        cfg_weight = cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT
        temperature = temperature if temperature is not None else Config.TEMPERATURE
        
        # Map quality_mode to n_timesteps and set internal parameters
        quality_mapping = {"fast": 3, "balanced": 5, "quality": 10}
        effective_quality_mode = quality_mode or "balanced"
        n_timesteps = quality_mapping.get(effective_quality_mode, 5)
        
        # Split text into chunks
        update_tts_status(request_id, TTSStatus.CHUNKING, "Splitting text into chunks")
        chunks = split_text_into_chunks(text, Config.MAX_CHUNK_LENGTH)
        
        voice_source = "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "configured sample"
        print(f"Processing {len(chunks)} text chunks with {voice_source} and parameters:")
        print(f"  - Exaggeration: {exaggeration}")
        print(f"  - CFG Weight: {cfg_weight}")
        print(f"  - Temperature: {temperature}")
        
        # Update status with chunk information
        update_tts_status(request_id, TTSStatus.GENERATING_AUDIO, "Starting audio generation",
                        current_chunk=0, total_chunks=len(chunks))
        
        # Generate audio for each chunk with memory management
        loop = asyncio.get_event_loop()
        
        for i, chunk in enumerate(chunks):
            # Update progress
            current_step = f"Generating audio for chunk {i+1}/{len(chunks)}"
            update_tts_status(request_id, TTSStatus.GENERATING_AUDIO, current_step,
                            current_chunk=i+1, total_chunks=len(chunks))
            
            print(f"Generating audio for chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            
            # Use torch.no_grad() to prevent gradient accumulation
            with torch.no_grad():
                # Run TTS generation in executor to avoid blocking
                # Prepare generation kwargs - only pass parameters explicitly provided by client
                generate_kwargs = {
                    "text": chunk,
                    "audio_prompt_path": voice_sample_path,
                    "n_timesteps": n_timesteps,
                }
                
                # Only add parameters if they were explicitly provided by client
                if exaggeration is not None:
                    generate_kwargs["exaggeration"] = exaggeration
                if cfg_weight is not None:
                    generate_kwargs["cfg_weight"] = cfg_weight
                if temperature is not None:
                    generate_kwargs["temperature"] = temperature
                
                # Always add language_id (defaulted to "en" if not specified)
                generate_kwargs["language_id"] = language_id
                
                # Use generate() for standard generation (non-streaming)
                audio_tensor = await loop.run_in_executor(
                    None,
                    lambda: model.generate(**generate_kwargs)
                )
                
                # Ensure tensor is on the correct device and detached
                if hasattr(audio_tensor, 'detach'):
                    audio_tensor = audio_tensor.detach()
                
                audio_chunks.append(audio_tensor)
            
        
        # Concatenate all chunks
        if len(audio_chunks) > 1:
            update_tts_status(request_id, TTSStatus.CONCATENATING, "Concatenating audio chunks")
            print("Concatenating audio chunks...")
            with torch.no_grad():
                final_audio = concatenate_audio_chunks(audio_chunks, model.sr)
        else:
            final_audio = audio_chunks[0]
        
        # Convert to WAV format
        update_tts_status(request_id, TTSStatus.FINALIZING, "Converting to WAV format")
        buffer = io.BytesIO()
        
        # Ensure final_audio is on CPU for saving
        if hasattr(final_audio, 'cpu'):
            final_audio_cpu = final_audio.cpu()
        else:
            final_audio_cpu = final_audio
            
        ta.save(buffer, final_audio_cpu, model.sr, format="wav")
        buffer.seek(0)
        
        # Mark as completed
        update_tts_status(request_id, TTSStatus.COMPLETED, "Audio generation completed")
        print(f"✓ Audio generation completed. Size: {len(buffer.getvalue()):,} bytes")
        
        return buffer
        
    except Exception as e:
        # Update status with error
        update_tts_status(request_id, TTSStatus.ERROR, error_message=f"TTS generation failed: {str(e)}")
        print(f"✗ TTS generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"TTS generation failed: {str(e)}",
                    "type": "generation_error"
                }
            }
        )
    
    finally:
        # Comprehensive cleanup
        try:
            # Clear the list
            audio_chunks.clear()
            
            
        except Exception as cleanup_error:
            print(f"⚠️ Warning during cleanup: {cleanup_error}")


async def generate_speech_streaming(
    text: str,
    voice_sample_path: str,
    language_id: str = "en",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    quality_mode: str = "balanced",
    stream_chunk_size: Optional[List[int]] = None
) -> AsyncGenerator[bytes, None]:
    """Streaming function to generate speech with real-time chunk yielding"""
    
    # Start TTS request tracking
    voice_source = "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "default"
    request_id = start_tts_request(
        text=text,
        voice_source=voice_source,
        parameters={
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "voice_sample_path": voice_sample_path,
            "streaming": True,
        }
    )
    
    update_tts_status(request_id, TTSStatus.INITIALIZING, "Checking model availability (streaming)")
    
    model = get_model()
    if model is None:
        update_tts_status(request_id, TTSStatus.ERROR, error_message="Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Model not loaded", "type": "model_error"}}
        )
    
    # Always use multilingual model
    # Default to English if no language_id specified
    if not language_id:
        language_id = "en"

    
    # Validate total text length
    update_tts_status(request_id, TTSStatus.PROCESSING_TEXT, "Validating text length")
    if len(text) > Config.MAX_TOTAL_LENGTH:
        update_tts_status(request_id, TTSStatus.ERROR, 
                        error_message=f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"Input text too long. Maximum {Config.MAX_TOTAL_LENGTH} characters allowed.",
                    "type": "invalid_request_error"
                }
            }
        )

    # WAV header info for streaming
    sample_rate = model.sr
    channels = 1
    bits_per_sample = 16
    
    # Generate and yield WAV header first
    try:
        # Get parameters with defaults
        exaggeration = exaggeration if exaggeration is not None else Config.EXAGGERATION
        cfg_weight = cfg_weight if cfg_weight is not None else Config.CFG_WEIGHT
        temperature = temperature if temperature is not None else Config.TEMPERATURE
        
        # Map quality_mode to n_timesteps and set internal parameters
        quality_mapping = {"fast": 3, "balanced": 5, "quality": 10}
        effective_quality_mode = quality_mode or "balanced"
        n_timesteps = quality_mapping.get(effective_quality_mode, 5)
        
        effective_stream_chunk_size = stream_chunk_size or [20, 50, 100]
        
        # Split text using streaming-optimized chunking
        update_tts_status(request_id, TTSStatus.CHUNKING, "Splitting text for streaming")
        
        voice_source = "uploaded file" if voice_sample_path != Config.VOICE_SAMPLE_PATH else "configured sample"
        print(f"Streaming {text} with {voice_source} and parameters:")
        print(f"  - Exaggeration: {exaggeration}")
        print(f"  - CFG Weight: {cfg_weight}")
        print(f"  - Temperature: {temperature}")
        
        # Update status with chunk information
        update_tts_status(request_id, TTSStatus.GENERATING_AUDIO, "Starting streaming audio generation")
        
        # Yield a proper WAV header for streaming
        wav_header = create_wav_header(sample_rate, channels, bits_per_sample)
        yield wav_header
        
        total_samples = 0
        
        # Update progress
        update_tts_status(request_id, TTSStatus.GENERATING_AUDIO)
            
        # Use torch.no_grad() to prevent gradient accumulation
        with torch.no_grad():
            """Generator function to run in executor"""
            # Prepare streaming generation kwargs - only pass parameters explicitly provided by client
            stream_kwargs = {
                "text": text,
                "audio_prompt_path": voice_sample_path,
                "n_timesteps": n_timesteps,
                "stream_chunk_size": effective_stream_chunk_size,
            }
            
            # Only add parameters if they were explicitly provided by client
            if exaggeration is not None:
                stream_kwargs["exaggeration"] = exaggeration
            if cfg_weight is not None:
                stream_kwargs["cfg_weight"] = cfg_weight
            if temperature is not None:
                stream_kwargs["temperature"] = temperature
            
            # Always add language_id (defaulted to "en" if not specified)
            stream_kwargs["language_id"] = language_id
            
            for audio_chunk, metrics in model.generate_stream(**stream_kwargs):
                try:
                    # Check if task is cancelled (client disconnected)
                    if asyncio.current_task().cancelled():
                        print("🛑 Generation task cancelled, stopping")
                        break
                    
                    # Real-time metrics available
                    if metrics.latency_to_first_chunk:
                        print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
                    else:
                        print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if metrics.rtf else f"Chunk {metrics.chunk_count}")
                    
                    # Only process non-empty chunks
                    if audio_chunk is not None and hasattr(audio_chunk, 'numel') and audio_chunk.numel() > 0:
                        try:
                            # Convert tensor to bytes PCM for HTTP streaming
                            # Note: audio_chunk arrives already on CPU via .detach().cpu() from model
                            audio_tensor = torch.clamp(audio_chunk.detach().cpu(), -1.0, 1.0)
                            audio_tensor_int = (audio_tensor * 32767).to(torch.int16)
                            pcm_data = audio_tensor_int.numpy().tobytes()
                            yield pcm_data
                        except Exception as chunk_error:
                            print(f"  ⚠️  Error processing chunk: {chunk_error}")
                            continue
                    else:
                        if audio_chunk is None:
                            print("  ⚠️  Received None chunk, skipping...")
                        else:
                            print(f"  ⚠️  Received invalid chunk (type: {type(audio_chunk)}), skipping...")
                            
                except asyncio.CancelledError:
                    print("🔌 Client disconnection detected in generation loop, stopping")
                    raise

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Mark as completed
        update_tts_status(request_id, TTSStatus.COMPLETED, "Streaming audio generation completed")
        print(f"✓ Streaming audio generation completed. Total samples: {total_samples:,}")
        
    except Exception as e:
        # Update status with error
        update_tts_status(request_id, TTSStatus.ERROR, error_message=f"TTS streaming failed: {str(e)}")
        print(f"✗ TTS streaming failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"TTS streaming failed: {str(e)}",
                    "type": "generation_error"
                }
            }
        )
    
    finally:
        pass


@router.post(
    "/speech",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}, "audio/pcm": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate speech from text",
    description="Generate speech audio from input text. Supports voice names from the voice library or defaults to configured voice sample. Choose response_format 'wav' or 'pcm' for audio format."
)
async def text_to_speech(request: TTSRequest, http_request: Request):
    """Generate speech from text using Chatterbox TTS with voice selection support"""
    
    # Resolve voice name to file path and language
    voice_sample_path, language_id = resolve_voice_path_and_language(request.voice)
    
    if request.stream:
        # Real streaming audio generation using model.generate_stream()
        if request.response_format == "pcm":
            # Raw PCM streaming without WAV header
            async def pcm_stream():
                first_chunk = True
                try:
                    async for chunk in generate_speech_streaming(
                        text=request.input,
                        voice_sample_path=voice_sample_path,
                        language_id=language_id,
                        exaggeration=request.exaggeration,
                        cfg_weight=request.cfg_weight,
                        temperature=request.temperature,
                        quality_mode=request.quality_mode.value if request.quality_mode else "balanced",
                        stream_chunk_size=request.stream_chunk_size
                    ):
                        if first_chunk:
                            first_chunk = False
                            # Skip WAV header (first 44 bytes) for PCM format
                            if len(chunk) > 44:
                                yield chunk[44:]
                        else:
                            yield chunk
                except asyncio.CancelledError:
                    print("🔌 Client disconnected during PCM streaming, stopping TTS generation")
                    raise
            
            return StreamingResponse(
                pcm_stream(),
                media_type="audio/pcm",
                headers={
                    "Content-Disposition": "attachment; filename=speech.pcm",
                    "Transfer-Encoding": "chunked",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # WAV streaming with header
            # WAV streaming with header and disconnect detection
            async def wav_stream():
                try:
                    async for chunk in generate_speech_streaming(
                        text=request.input,
                        voice_sample_path=voice_sample_path,
                        language_id=language_id,
                        exaggeration=request.exaggeration,
                        cfg_weight=request.cfg_weight,
                        temperature=request.temperature,
                        quality_mode=request.quality_mode.value if request.quality_mode else "balanced",
                        stream_chunk_size=request.stream_chunk_size
                    ):
                        yield chunk
                except asyncio.CancelledError:
                    print("🔌 Client disconnected during WAV streaming, stopping TTS generation")
                    raise

            return StreamingResponse(
                wav_stream(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech_stream.wav",
                    "Transfer-Encoding": "chunked",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering for true streaming
                }
            )
    else:
        # Standard generation using model.generate()
        buffer = await generate_speech_internal(
            text=request.input,
            voice_sample_path=voice_sample_path,
            language_id=language_id,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            quality_mode=request.quality_mode.value if request.quality_mode else "balanced",
            stream_chunk_size=request.stream_chunk_size
        )
        
        if request.response_format == "pcm":
            # Return raw PCM data without WAV header
            audio_data = buffer.getvalue()
            if len(audio_data) > 44:  # Skip WAV header
                pcm_data = audio_data[44:]
            else:
                pcm_data = audio_data
            
            return StreamingResponse(
                io.BytesIO(pcm_data),
                media_type="audio/pcm",
                headers={"Content-Disposition": "attachment; filename=speech.pcm"}
            )
        else:
            # Return WAV data with header
            return StreamingResponse(
                io.BytesIO(buffer.getvalue()),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"}
            )


@router.post(
    "/speech/upload",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"audio/wav": {}, "audio/pcm": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Generate speech with custom voice upload or library selection",
    description="Generate speech audio from input text with voice library selection or optional custom voice file upload. Choose response_format 'wav' or 'pcm' for audio format."
)
async def text_to_speech_with_upload(
    http_request: Request,
    input: str = Form(..., description="The text to generate audio for", min_length=1, max_length=3000),
    voice: Optional[str] = Form("alloy", description="Voice name from library or OpenAI voice name (defaults to configured sample)"),
    response_format: Optional[str] = Form("wav", description="Audio format: 'wav' (with WAV header) or 'pcm' (raw PCM data)"),
    speed: Optional[float] = Form(1.0, description="Speed of speech (ignored)"),
    stream: Optional[bool] = Form(True, description="Use streaming generation (model.generate_stream) if True, standard generation (model.generate) if False"),
    exaggeration: Optional[float] = Form(None, description="Emotion intensity (0.25-2.0)", ge=0.25, le=2.0),
    cfg_weight: Optional[float] = Form(None, description="Pace control (0.0-1.0)", ge=0.0, le=1.0),
    temperature: Optional[float] = Form(None, description="Sampling temperature (0.05-5.0)", ge=0.05, le=5.0),
    quality_mode: Optional[str] = Form("balanced", description="Quality mode: 'fast', 'balanced', or 'quality'"),
    stream_chunk_size: Optional[List[int]] = Form([20, 50, 100], description="Stream chunk sizes"),
    voice_file: Optional[UploadFile] = File(None, description="Optional voice sample file for custom voice cloning")
):
    """Generate speech from text using Chatterbox TTS with optional voice file upload"""
    
    # Validate input text
    if not input or not input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "Input text cannot be empty", "type": "invalid_request_error"}}
        )
    
    input = input.strip()
    
    # Validate response_format
    if response_format not in ['wav', 'pcm']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "response_format must be 'wav' or 'pcm'", "type": "validation_error"}}
        )
    
    # Handle voice selection and file upload
    temp_voice_path = None
    voice_sample_path = Config.VOICE_SAMPLE_PATH  # Default
    language_id = "en"  # Default language
    
    # First, try to resolve voice name from library if no file uploaded
    if not voice_file:
        voice_sample_path, language_id = resolve_voice_path_and_language(voice)
    
    # If a file is uploaded, it takes priority over voice name
    if voice_file:
        try:
            # Validate the uploaded file
            validate_audio_file(voice_file)
            
            # Create temporary file for the voice sample
            file_ext = os.path.splitext(voice_file.filename.lower())[1]
            temp_voice_fd, temp_voice_path = tempfile.mkstemp(suffix=file_ext, prefix="voice_sample_")
            
            # Read and save the uploaded file
            file_content = await voice_file.read()
            with os.fdopen(temp_voice_fd, 'wb') as temp_file:
                temp_file.write(file_content)
            
            voice_sample_path = temp_voice_path
            print(f"Using uploaded voice file: {voice_file.filename} ({len(file_content):,} bytes)")
            
        except HTTPException:
            raise
        except Exception as e:
            # Clean up temp file if it was created
            if temp_voice_path and os.path.exists(temp_voice_path):
                try:
                    os.unlink(temp_voice_path)
                except:
                    pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": f"Failed to process voice file: {str(e)}",
                        "type": "file_processing_error"
                    }
                }
            )
    
    if stream:
            # Real streaming audio generation using model.generate_stream()
            if response_format == "pcm":
                # Raw PCM streaming without WAV header
                async def pcm_stream_with_cleanup():
                    try:
                        first_chunk = True
                        async for chunk in generate_speech_streaming(
                            text=input,
                            voice_sample_path=voice_sample_path,
                            language_id=language_id,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            quality_mode=quality_mode or "balanced",
                            stream_chunk_size=stream_chunk_size
                        ):
                            if first_chunk:
                                first_chunk = False
                                # Skip WAV header (first 44 bytes) for PCM format
                                if len(chunk) > 44:
                                    yield chunk[44:]
                            else:
                                yield chunk
                    except asyncio.CancelledError:
                        print("🔌 Client disconnected during PCM upload streaming, stopping TTS generation")
                        raise
                    finally:
                        # Clean up temporary voice file
                        if temp_voice_path and os.path.exists(temp_voice_path):
                            try:
                                os.unlink(temp_voice_path)
                                print(f"🗑️ Cleaned up temporary voice file: {temp_voice_path}")
                            except Exception as e:
                                print(f"⚠️ Warning: Failed to clean up temporary voice file: {e}")
                
                return StreamingResponse(
                    pcm_stream_with_cleanup(),
                    media_type="audio/pcm",
                    headers={
                        "Content-Disposition": "attachment; filename=speech.pcm",
                        "Transfer-Encoding": "chunked",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                # WAV streaming with header
                async def wav_stream_with_cleanup():
                    try:
                        async for chunk in generate_speech_streaming(
                            text=input,
                            voice_sample_path=voice_sample_path,
                            language_id=language_id,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            quality_mode=quality_mode or "balanced",
                            stream_chunk_size=stream_chunk_size
                        ):
                            yield chunk
                    except asyncio.CancelledError:
                        print("🔌 Client disconnected during WAV upload streaming, stopping TTS generation")
                        raise
                    finally:
                        # Clean up temporary voice file
                        if temp_voice_path and os.path.exists(temp_voice_path):
                            try:
                                os.unlink(temp_voice_path)
                                print(f"🗑️ Cleaned up temporary voice file: {temp_voice_path}")
                            except Exception as e:
                                print(f"⚠️ Warning: Failed to clean up temporary voice file: {e}")
                
                return StreamingResponse(
                    wav_stream_with_cleanup(),
                    media_type="audio/wav",
                    headers={
                        "Content-Disposition": "attachment; filename=speech_stream.wav",
                        "Transfer-Encoding": "chunked",
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no"  # Disable nginx buffering for true streaming
                    }
                )
    else:
            # Standard generation using model.generate()
            buffer = await generate_speech_internal(
                text=input,
                voice_sample_path=voice_sample_path,
                language_id=language_id,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                quality_mode=quality_mode or "balanced",
                stream_chunk_size=stream_chunk_size
            )
            
            if response_format == "pcm":
                # Return raw PCM data without WAV header
                audio_data = buffer.getvalue()
                if len(audio_data) > 44:  # Skip WAV header
                    pcm_data = audio_data[44:]
                else:
                    pcm_data = audio_data
                
                return StreamingResponse(
                    io.BytesIO(pcm_data),
                    media_type="audio/pcm",
                    headers={"Content-Disposition": "attachment; filename=speech.pcm"}
                )
            else:
                # Return WAV data with header
                return StreamingResponse(
                    io.BytesIO(buffer.getvalue()),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=speech.wav"}
                )
        



# Export the base router for the main app to use
__all__ = ["base_router"] 