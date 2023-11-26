import pyaudio
import wave

def record_audio(file_path, duration=30, sample_rate=44100, channels=2, chunk=1024):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording... Press Ctrl+C to stop.")

    frames = []
    try:
        for i in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
            print(f"Chunk {i}: {len(data)} bytes recorded")
    except KeyboardInterrupt:
        print("Recording stopped by user.")
    finally:
        print("Recording complete.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    output_file = "output.wav"
    record_audio(output_file)
    print(f"Audio recorded and saved to {output_file}.")
