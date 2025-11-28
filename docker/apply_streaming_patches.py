#!/usr/bin/env python3

import os
import glob
import re

def find_chatterbox_dir():
    """Find the chatterbox package directory in site-packages"""
    patterns = [
        '/app/.venv/lib/python3.11/site-packages/chatterbox*',
        '/app/.venv/lib/python3.11/site-packages/*chatterbox*'
    ]
    
    for pattern in patterns:
        dirs = glob.glob(pattern)
        if dirs:
            return dirs[0]
    return None

def apply_t3_streaming_patch(pkg_dir):
    """Apply streaming patch to T3 model"""
    t3_file = os.path.join(pkg_dir, 'models/t3/t3.py')
    
    if not os.path.exists(t3_file):
        print(f'❌ T3 file not found at {t3_file}')
        return False
    
    print(f'📝 Applying T3 streaming patch to {t3_file}')
    
    with open(t3_file, 'r') as f:
        content = f.read()
    
    # 1. Remove tqdm from generation loop
    content = re.sub(
        r'for i in tqdm\(range\(max_new_tokens\)[^:]*\):',
        'for i in range(max_new_tokens):',
        content
    )
    
    # 2. Add yield before predicted.append
    content = content.replace(
        'predicted.append(next_token)',
        'yield next_token.squeeze()\n            predicted.append(next_token)'
    )
    
    # 3. Remove logger statement
    content = re.sub(
        r'logger\.info\([^)]*EOS token detected[^)]*\)',
        '',
        content
    )
    
    # 4. Remove final return logic
    pattern = r'\s*# Concatenate all predicted tokens.*?return predicted_tokens'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    with open(t3_file, 'w') as f:
        f.write(content)
    
    print('✅ T3 streaming patch applied successfully')
    return True

def apply_tts_streaming_patch(pkg_dir):
    """Apply streaming patch to TTS model"""
    tts_file = os.path.join(pkg_dir, 'tts.py')
    
    if not os.path.exists(tts_file):
        print(f'❌ TTS file not found at {tts_file}')
        return False
    
    print(f'📝 Applying TTS streaming patch to {tts_file}')
    
    with open(tts_file, 'r') as f:
        content = f.read()
    
    # 1. Add chunk_size parameter to generate method
    content = re.sub(
        r'(def generate\([^)]*temperature=0\.8,)\s*\):',
        r'\1\n        chunk_size=10,\n    ):',
        content
    )
    
    # 2. Replace the entire inference section
    # Find start and end markers
    start_marker = 'with torch.inference_mode():'
    end_marker = 'return torch.from_numpy(watermarked_wav).unsqueeze(0)'
    
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos == -1 or end_pos == -1:
        print('❌ Could not find inference section markers')
        return False
    
    # Find the full end position
    end_pos = end_pos + len(end_marker)
    
    # New streaming implementation
    new_implementation = '''with torch.inference_mode():
            speech_tokens_buffer = []
            for speech_token in self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            ):
                speech_tokens_buffer.append(speech_token)
                
                if len(speech_tokens_buffer) >= chunk_size:
                    tokens_chunk = torch.stack(speech_tokens_buffer)
                    tokens_chunk = drop_invalid_tokens(tokens_chunk)
                    tokens_chunk = tokens_chunk[tokens_chunk < 6561]
                    
                    if len(tokens_chunk) > 0:
                        wav_chunk, _ = self.s3gen.inference(
                            speech_tokens=tokens_chunk.to(self.device),
                            ref_dict=self.conds.gen,
                        )
                        wav_chunk = wav_chunk.squeeze(0).detach().cpu()
                        watermarked_chunk = self.watermarker.apply_watermark(
                            wav_chunk.numpy(), sample_rate=self.sr
                        )
                        yield torch.from_numpy(watermarked_chunk)
                    
                    speech_tokens_buffer = []
            
            # Process remaining tokens
            if speech_tokens_buffer:
                tokens_chunk = torch.stack(speech_tokens_buffer)
                tokens_chunk = drop_invalid_tokens(tokens_chunk)
                tokens_chunk = tokens_chunk[tokens_chunk < 6561]
                
                if len(tokens_chunk) > 0:
                    wav_chunk, _ = self.s3gen.inference(
                        speech_tokens=tokens_chunk.to(self.device),
                        ref_dict=self.conds.gen,
                    )
                    wav_chunk = wav_chunk.squeeze(0).detach().cpu()
                    watermarked_chunk = self.watermarker.apply_watermark(
                        wav_chunk.numpy(), sample_rate=self.sr
                    )
                    yield torch.from_numpy(watermarked_chunk)'''
    
    # Replace the section
    content = content[:start_pos] + new_implementation + content[end_pos:]
    
    with open(tts_file, 'w') as f:
        f.write(content)
    
    print('✅ TTS streaming patch applied successfully')
    return True

def apply_api_streaming_patch():
    """Apply streaming patch to API endpoints"""
    speech_file = '/app/app/api/endpoints/speech.py'
    
    if not os.path.exists(speech_file):
        print(f'❌ API file not found at {speech_file}')
        return False
    
    print(f'📝 Applying API streaming patch to {speech_file}')
    
    with open(speech_file, 'r') as f:
        content = f.read()
    
    # 1. Bypass text chunking for native streaming
    old_chunking = '''update_tts_status(request_id, TTSStatus.CHUNKING, "Splitting text for streaming")
        chunks = split_text_for_streaming(
            text,
            chunk_size=streaming_settings["chunk_size"],
            strategy=streaming_settings["strategy"],
            quality=streaming_settings["quality"]
        )'''
    
    new_chunking = '''update_tts_status(request_id, TTSStatus.CHUNKING, "Splitting text for streaming")
        
        # ⭐ BYPASS chunking textuel pour streaming natif
        if len(text) <= Config.MAX_TOTAL_LENGTH:
            chunks = [text]  # Pas de chunking - streaming direct du modèle
            print(f"🚀 Streaming natif activé - pas de découpage textuel")
        else:
            # Fallback pour textes très longs
            chunks = split_text_for_streaming(
                text,
                chunk_size=streaming_settings["chunk_size"],
                strategy=streaming_settings["strategy"],
                quality=streaming_settings["quality"]
            )
            print(f"⚠️ Texte trop long, chunking de secours ({len(chunks)} chunks)")'''
    
    content = content.replace(old_chunking, new_chunking)
    
    # 2. Replace model.generate call with streaming version
    old_generate = '''# Run TTS generation in executor to avoid blocking
                audio_tensor = await loop.run_in_executor(
                    None,
                    lambda: model.generate(
                        text=chunk,
                        audio_prompt_path=voice_sample_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        **({'language_id': language_id} if is_multilingual() else {})
                    )
                )
                
                # Ensure tensor is on CPU for streaming
                if hasattr(audio_tensor, 'cpu'):
                    audio_tensor = audio_tensor.cpu()

                # Convert tensor to raw 16-bit PCM data
                # Clamp values to [-1, 1] before conversion
                audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
                audio_tensor_int = (audio_tensor * 32767).to(torch.int16)
                
                # Yield the raw audio data as bytes
                pcm_data = audio_tensor_int.numpy().tobytes()
                yield pcm_data
                
                total_samples += audio_tensor.shape[1]
                
                # Clean up this chunk
                safe_delete_tensors(audio_tensor, audio_tensor_int)
                del pcm_data'''
    
    new_generate = '''# ⭐ Streaming natif du modèle avec micro-chunks
                generate_kwargs = {
                    "text": chunk,
                    "audio_prompt_path": voice_sample_path,
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "temperature": temperature,
                    "chunk_size": 5,  # Très petits chunks pour réactivité max
                    **({'language_id': language_id} if is_multilingual() else {})
                }
                
                # ⭐ Stream chaque micro-chunk immédiatement
                audio_generator = await loop.run_in_executor(
                    None,
                    lambda: model.generate(**generate_kwargs)
                )
                
                # Yield chaque micro-chunk dès qu'il arrive
                for audio_chunk in audio_generator:
                    if hasattr(audio_chunk, 'cpu'):
                        audio_chunk = audio_chunk.cpu()

                    # Convert to PCM and yield immediately
                    audio_chunk = torch.clamp(audio_chunk, -1.0, 1.0)
                    audio_tensor_int = (audio_chunk * 32767).to(torch.int16)
                    pcm_data = audio_tensor_int.numpy().tobytes()
                    
                    # ⭐ YIELD immédiat pour streaming natif
                    print(f"📡 Micro-chunk: {len(pcm_data)} bytes")
                    yield pcm_data
                    
                    chunk_samples = audio_chunk.shape[-1] if audio_chunk.dim() > 0 else 0
                    total_samples += chunk_samples
                    
                    # Cleanup
                    safe_delete_tensors(audio_chunk, audio_tensor_int)
                    del pcm_data'''
    
    content = content.replace(old_generate, new_generate)
    
    with open(speech_file, 'w') as f:
        f.write(content)
    
    print('✅ API streaming patch applied successfully')
    return True

def main():
    print("🚀 Starting streaming patches application...")
    
    # Find chatterbox package
    pkg_dir = find_chatterbox_dir()
    if not pkg_dir:
        print('❌ No chatterbox package found in site-packages')
        return False
    
    print(f'📦 Found chatterbox package at: {pkg_dir}')
    
    # List package contents for debugging
    print('📁 Package contents:')
    for root, dirs, files in os.walk(pkg_dir):
        level = root.replace(pkg_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files per directory
            print(f'{sub_indent}{file}')
        if len(files) > 5:
            print(f'{sub_indent}... and {len(files) - 5} more files')
    
    # Apply patches
    t3_success = apply_t3_streaming_patch(pkg_dir)
    tts_success = apply_tts_streaming_patch(pkg_dir)
    api_success = apply_api_streaming_patch()
    
    if t3_success and tts_success and api_success:
        print('🎉 All streaming patches applied successfully!')
        return True
    else:
        print('⚠️ Some patches failed to apply')
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)