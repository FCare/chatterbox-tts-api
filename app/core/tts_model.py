"""
TTS model initialization and management
"""

import os
import asyncio
from enum import Enum
from typing import Optional, Dict, Any
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages
    
    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None
            
            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)
            
            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')
            
            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file
        
        # Always use multilingual model for memory efficiency
        _initialization_progress = "Loading TTS model (this may take a while)..."
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        print(f"Loading Chatterbox Multilingual TTS model...")
        _model = await loop.run_in_executor(
            None,
            lambda: ChatterboxMultilingualTTS.from_pretrained(device=_device)
        )
        _is_multilingual = True
        _supported_languages = SUPPORTED_LANGUAGES.copy()
        print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        
        if _device == 'cuda':
            import torch
            print("🚀 Applying performance optimizations...")
            
            def t3_to(model, dtype):
                model.t3.to(dtype=dtype)
                model.conds.t3.to(dtype=dtype)
                torch.cuda.empty_cache()
                return model
            
            # Most new GPUs would work the fastest with this, but not all.
            _model = t3_to(_model, torch.bfloat16)
            print("✓ Performance optimizations applied")
            # Warmup avec cudagraphs pour le modèle multilingue
            warmup_text = "fast generation using cudagraphs-manual, warmup"
            
            # Warmup avec nouveaux paramètres
            warmup_params = {
                "n_timesteps": 3,  # Fast pour warmup
                "max_new_tokens": 500,
                "max_cache_len": 1500,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "top_p": 1.0,
                "stream_chunk_size": [20, 50, 100],
                "context_window": 50,
                "fade_duration": 0.02,
                "print_metrics": True,
                "t3_params": {}
            }
            # Warmup en itérant sur le générateur pour qu'il s'exécute réellement
            print("🔥 Warming up English...")
            # for _ in _model.generate_stream(warmup_text, language_id="en", **warmup_params):
            for _ in _model.generate(warmup_text, language_id="en", print_metrics=True, n_timesteps=5):
                pass  # Consommer le générateur
            print("🔥 Warming up French...")
            for _ in _model.generate_stream(warmup_text, language_id="fr", **warmup_params):
                pass  # Consommer le générateur
            print("✓ Model warmed up with cudagraphs")
            
            # Prepare conditionals cache for all voice library voices
            print("🎯 Preparing conditionals cache for voice library voices...")
            try:
                import gc
                from app.core import get_voice_library
                voice_lib = get_voice_library()
                voices = voice_lib.list_voices()
                
                if voices:
                    for voice_data in voices:
                        voice_name = voice_data["name"]
                        voice_path = voice_data["path"]
                        
                        # Skip default voice
                        if voice_path == Config.VOICE_SAMPLE_PATH:
                            continue
                            
                        print(f"  🎵 Preparing conditionals for voice: {voice_name}")
                        for exaggeration in [i/10.0 for i in range(0, 11)]:  # 0.0, 0.1, 0.2, ..., 1.0
                            try:
                                _model.prepare_conditionals_cache(voice_path, exaggeration)
                            except Exception as e:
                                print(f"    ⚠️ Warning: Failed for exaggeration={exaggeration:.1f}: {e}")
                                break  # Skip remaining exaggeration values for this voice
                        print(f"    ✓ Conditionals prepared for {voice_name}")
                else:
                    print("  ℹ️ No custom voices found in voice library")
                    
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to prepare voice library conditionals: {e}")
            
            print("✓ Voice library conditionals cache prepared")
        
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return True  # Always True since we always use multilingual model


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    return {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state
    }