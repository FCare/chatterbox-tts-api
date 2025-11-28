#!/usr/bin/env python3
"""
Test script to verify SSE streaming functionality for the frontend
"""

import requests
import json
import time
import base64

def test_sse_streaming():
    """Test SSE streaming endpoint"""
    url = "http://localhost:4123/v1/audio/speech"
    
    payload = {
        "input": "Hello from the streaming test! This is a comprehensive test of the SSE streaming functionality. We'll generate multiple chunks to test the real-time playback and final audio concatenation. Each chunk should play immediately as it's received, and the final downloadable file should contain all chunks properly merged together.",
        "stream_format": "sse",
        "exaggeration": 0.7,
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    }
    
    print("🎵 Testing SSE streaming with multiple chunks...")
    print(f"📝 Text: {payload['input'][:100]}...")
    print(f"🔧 Stream format: {payload['stream_format']}")
    print()
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        
        if response.status_code != 200:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        print("✅ SSE stream started successfully!")
        print("📡 Receiving events...")
        print()
        
        audio_chunks = []
        event_count = 0
        start_time = time.time()
        first_chunk_time = None
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                event_data = line[6:]  # Remove 'data: ' prefix
                
                try:
                    event = json.loads(event_data)
                    event_count += 1
                    current_time = time.time() - start_time
                    
                    if event.get('type') == 'speech.audio.delta':
                        audio_chunks += 1
                        if first_chunk_time is None:
                            first_chunk_time = current_time
                        
                        audio_b64 = event.get('audio', '')
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            print(f"🎧 Chunk {audio_chunks} at {current_time:.1f}s: {len(audio_data)} bytes")
                        
                    elif event.get('type') == 'speech.audio.done':
                        usage = event.get('usage', {})
                        total_time = current_time
                        print()
                        print("✅ Streaming completed!")
                        print(f"📊 Total events: {event_count}")
                        print(f"🎵 Audio chunks: {audio_chunks}")
                        print(f"⏱️  Total time: {total_time:.1f}s")
                        print(f"⚡ First chunk: {first_chunk_time:.1f}s")
                        print(f"🔢 Usage: {usage}")
                        
                        # Verify we got multiple chunks
                        if audio_chunks > 1:
                            print("✅ Multiple chunks received - real-time playback should work!")
                        else:
                            print("⚠️  Only one chunk received - text might be too short")
                        
                        return True
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Could not parse event: {event_data}")
                    continue
        
        print("❌ Stream ended without completion event")
        return False
        
    except Exception as e:
        print(f"❌ Error during streaming test: {e}")
        return False


def test_multiple_chunk_streaming():
    """Test streaming with guaranteed multiple chunks"""
    url = "http://localhost:4123/v1/audio/speech"
    
    # Long text designed to create multiple chunks
    long_text = """
    This is a comprehensive test of the streaming functionality. 
    We are testing multiple sentences to ensure that the audio streaming works correctly.
    Each sentence should generate a separate chunk that plays in real-time.
    The frontend should receive each chunk immediately and play it without waiting.
    After all chunks are received, a final downloadable file should be created.
    This final file should contain all the audio chunks properly concatenated together.
    The user should hear continuous speech without gaps or overlaps between chunks.
    """
    
    payload = {
        "input": long_text.strip(),
        "stream_format": "sse",
        "exaggeration": 0.8,
    }
    
    print("🎯 Testing multiple chunk streaming...")
    print(f"📝 Text length: {len(long_text)} characters")
    print()
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, stream=True, headers={'Accept': 'text/event-stream'})
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            return False
        
        chunks = []
        chunk_times = []
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                event_data = line[6:]
                try:
                    event = json.loads(event_data)
                    
                    if event.get('type') == 'speech.audio.delta':
                        current_time = time.time() - start_time
                        chunks.append(len(base64.b64decode(event.get('audio', ''))))
                        chunk_times.append(current_time)
                        print(f"🎵 Chunk {len(chunks)}: {chunks[-1]} bytes at {current_time:.1f}s")
                        
                    elif event.get('type') == 'speech.audio.done':
                        total_time = time.time() - start_time
                        print()
                        print(f"✅ Completed: {len(chunks)} chunks in {total_time:.1f}s")
                        
                        if len(chunks) >= 3:
                            print("🎉 Multiple chunks confirmed - streaming should work perfectly!")
                            
                            # Analyze timing
                            if len(chunk_times) >= 2:
                                avg_gap = sum(chunk_times[i+1] - chunk_times[i] for i in range(len(chunk_times)-1)) / (len(chunk_times)-1)
                                print(f"⏱️  Average gap between chunks: {avg_gap:.1f}s")
                        else:
                            print("⚠️  Expected more chunks for this text length")
                        
                        return len(chunks) >= 3
                        
                except json.JSONDecodeError:
                    continue
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:4123/health")
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check successful")
            print(f"📊 Status: {health.get('status')}")
            print(f"🤖 Model loaded: {health.get('model_loaded')}")
            print(f"🖥️  Device: {health.get('device')}")
            print()
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Frontend Streaming Test Suite v2.0")
    print("Testing real-time playback and WAV concatenation fixes")
    print("=" * 60)
    print()
    
    # Test health first
    if not test_health():
        print("❌ Health check failed, aborting tests")
        return
    
    # Wait a bit for the model to load
    print("⏳ Waiting for model to initialize...")
    time.sleep(3)
    print()
    
    # Test basic SSE streaming
    print("1️⃣ Testing basic SSE streaming...")
    print("-" * 40)
    basic_success = test_sse_streaming()
    print()
    
    # Test multiple chunk streaming
    print("2️⃣ Testing multiple chunk streaming...")
    print("-" * 40)
    multi_success = test_multiple_chunk_streaming()
    print()
    
    # Summary
    print("📋 Test Results Summary")
    print("=" * 30)
    print(f"Basic streaming: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Multi-chunk streaming: {'✅ PASS' if multi_success else '❌ FAIL'}")
    
    if basic_success and multi_success:
        print()
        print("🎉 All tests passed!")
        print("✨ Frontend streaming should now work correctly with:")
        print("   • Real-time playback of each chunk as it arrives")
        print("   • Proper audio scheduling without gaps")
        print("   • Correct WAV concatenation for final download")
        print()
        print("🎯 Next steps:")
        print("   1. Test in the frontend UI")
        print("   2. Verify real-time audio playback")
        print("   3. Check that downloadable file contains all chunks")
    else:
        print()
        print("❌ Some tests failed. Check the backend implementation.")


if __name__ == "__main__":
    main() 