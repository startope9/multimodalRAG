from collections import deque
import os
import pyaudio
import threading
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
# Make sure wsagent.py is in the same directory
from wsAgent import search_query_with_tavily
import time

# Get Deepgram API key
key = os.getenv("DEEPGRAM_API_KEY", "YOUR_KEY")
deepgram = DeepgramClient(key)
dg = deepgram.listen.websocket.v("1")


latest_transcript = {
    "full_text": "",
    "queue": deque(),
    "interim": ""
}

is_searching = False

def handle_transcript(client, result):
    global latest_transcript
    text = result.channel.alternatives[0].transcript.strip()
    is_final = result.is_final
    if text and not is_searching:
        if is_final:
            latest_transcript["queue"].append(text)
            latest_transcript["interim"] = ""
            # Clear interim line and print final
            print("\r" + " " * 100 + "\r", end="", flush=True)
            print(
                f"🗣️ {latest_transcript['full_text']} {' '.join(latest_transcript['queue'])}", end="", flush=True)
        else:
            latest_transcript["interim"] = text
            # Print interim on the same line
            print("\r" + " " * 100 + "\r", end="", flush=True)
            print(
                f"🗣️ {latest_transcript['full_text']} {' '.join(latest_transcript['queue'])} {text}", end="", flush=True)

def handle_error(client, error):
    print("\n❌ Error from Deepgram:", error)

dg.on(LiveTranscriptionEvents.Transcript, handle_transcript)
dg.on(LiveTranscriptionEvents.Error, handle_error)

opts = LiveOptions(
    model="nova-3",
    language="en-US",
    encoding="linear16",
    channels=1,
    sample_rate=16000,
    endpointing=False,
    interim_results=True
)

dg.start(opts)
print("🎤 Speak now (press Enter to search)...")

RATE = 16000
CHUNK = 4096
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, rate=RATE,
                 channels=1, input=True, frames_per_buffer=CHUNK)

def stream_audio():
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            dg.send(data)
    except Exception as e:
        print("\n❌ Audio error:", e)

audio_thread = threading.Thread(target=stream_audio, daemon=True)
audio_thread.start()

try:
    while True:
        input("\n\n🔍 Press Enter to search using the full transcript...\n")

        # Allow any final/interim events to arrive
        time.sleep(0.7)  # 0.5–1.0 seconds is usually enough

        # **NEW**: include any pending interim text in the queue so it ends up in full_text
        if latest_transcript["interim"]:
            latest_transcript["queue"].append(latest_transcript["interim"])
            latest_transcript["interim"] = ""

        # Drain queue into full_text
        while latest_transcript["queue"]:
            latest_transcript["full_text"] += " " + latest_transcript["queue"].popleft()

        query = latest_transcript["full_text"].strip()
        
        # Reset transcript buffers
        latest_transcript = {
            "full_text": "",
            "queue": deque(),
            "interim": ""
        }
        if not query:
            print("⚠️ No complete speech detected yet.")
            continue

        print(f"\n🔎 Searching: {query}\n")
        is_searching=True
        result = search_query_with_tavily(query)

        print("\n🧠 Answer:", result["answer"])
        print("\n🔗 Sources:")
        for i, src in enumerate(result["sources"], 1):
            print(f"{i}. {src['title']} – {src['url']}")
            print(f"   {src['content'][:200]}...\n")

        is_searching=False

        
except KeyboardInterrupt:
    print("\n👋 Exiting...")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    dg.finish()
    print("✅ Connection closed.")
