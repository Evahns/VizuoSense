import pyaudio as pa
import keyboard
import json
import subprocess
import threading
from vosk import Model, KaldiRecognizer, SetLogLevel
from text_to_speech import speech_output as speech_output

class SpeechToTextEngine:
    def __init__(self, model_path, model_name, lang, save_textfile_dir):
        self.model_path = model_path
        self.model_name = model_name
        self.lang = lang
        self.save_textfile_dir = save_textfile_dir
        self.stop_flag = threading.Event()
        self.p = None
        self.listen_keyword_detected = threading.Event()

    def configure(self):
        model = Model(model_path=self.model_path, model_name=self.model_name, lang=self.lang)
        SetLogLevel(0)
        self.p = pa.PyAudio()
        mic_index = 3
        stream = self.p.open(format=pa.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000,
                             input_device_index=mic_index)
        stream.start_stream()

        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        rec.SetPartialWords(True)

        return stream, rec, self.p

    def listen_for_keywords(self, stream, rec,p):
        recognized_text = ""
        while not self.stop_flag.is_set():
            data = stream.read(4000, exception_on_overflow=False)
            #print("Raw Audio Data:", data)
            rec.AcceptWaveform(data)
            partial_result = rec.PartialResult()
            if partial_result:
                partial_text = json.loads(partial_result).get("partial", "")
                print("Partial Text Received:", partial_text)
                if "listen" in partial_text.lower():
                    response = "Waking up! Listening for input..."
                    speech_output(response)
                    self.listen_keyword_detected.set()
                    return "listen"
                elif "stop" in partial_text.lower():
                    response = "Stop listening detected. Stopping..."
                    speech_output(response)
                    return "stop"
                elif "write only mode" in partial_text.lower():
                    response = "Write only mode detected. Switching from speech to writing mode..."
                    speech_output(response)
                    return "write only mode"
                else:
                    response = "No keyword detected. Speak to issue an input..."
                    speech_output(response)

    def listen_for_speech_prompt(self, stream, rec,p):
        recognized_text = ""
        while not self.stop_flag.is_set():
            if not self.listen_keyword_detected.is_set():
                continue

            data = stream.read(4000, exception_on_overflow=False)
            rec.AcceptWaveform(data)
            partial_result = rec.PartialResult()

            if partial_result:
                partial_text = json.loads(partial_result).get("partial", "")
                recognized_text += partial_text
                print("Partial Result:", partial_text)

                if "open browser" in partial_text.lower():
                    response = "Opening the browser..."
                    speech_output(response)
                    subprocess.run(["start", "https://www.google.com"])
                if keyboard.is_pressed('p'):
                    response = ''' KeyboardInterrupt: Stopping real-time listening
                        recognized text being: '''
                    speech_output(response)
                    self.stop_flag.set()
                    return recognized_text

    def real_time_listen(self):
        stream, rec,p = self.configure()

        # Create threads for simultaneous tasks
        keyword_thread = threading.Thread(target=self.listen_for_keywords, args=(stream, rec,p))
        speech_prompt_thread = threading.Thread(target=self.listen_for_speech_prompt, args=(stream, rec,p))

        try:
            keyword_thread.start()
            speech_prompt_thread.start()

            keyword_thread.join()  # Wait for the keyword thread to finish
            speech_prompt_thread.join()  # Wait for the speech prompt thread to finish

        except KeyboardInterrupt:
            response = "KeyboardInterrupt: Stopping real-time listening"
            speech_output(response)

        finally:
            self.stop_flag.set()  # Set the stop flag to signal threads to stop
            keyword_thread.join()  # Wait for the keyword thread to finish
            speech_prompt_thread.join()  # Wait for the speech prompt thread to finish

            stream.stop_stream()
            stream.close()
            self.p.terminate()

def main():
    model_path = "D:\\vizuosense_mine\\STT\\Resources\\vosk-model-small-en-us-0.15"
    model_name = "vosk-model-small-en-us-0.15"
    language = "small-en-us"
    save_textfile_dir = "D:\\vizuosense_mine\\STT\\Resources\\test.txt"
    stt_engine = SpeechToTextEngine(model_path, model_name, language, save_textfile_dir)
    stt_engine.real_time_listen()

if __name__ == "__main__":
    main()
