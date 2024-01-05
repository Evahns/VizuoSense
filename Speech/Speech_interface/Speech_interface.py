import pyaudio as pa
import keyboard,json,subprocess
from vosk import Model, KaldiRecognizer, SetLogLevel
from text_to_speech import speech_output as speech_output 
class SpeechToTextEngine:
    def __init__(self, model_path, model_name, lang, save_textfile_dir):
        self.model_path = model_path
        self.model_name = model_name
        self.lang = lang
        self.save_textfile_dir = save_textfile_dir

    def configure(self):
        """
        :this is the configuaration function for the program
        :returns 
            stream: speech stream
            rec: the reconizer initializer
            recognized_text: any text detected  and recognized by the recognizer
        """
        model = Model(model_path=self.model_path, model_name=self.model_name, lang=self.lang)

        SetLogLevel(0)
        p = pa.PyAudio()
        mic_index = 3
        stream = p.open(format=pa.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000, input_device_index=mic_index)
        stream.start_stream()

        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        rec.SetPartialWords(True)

        recognized_text = ""
        return stream, rec, recognized_text

    def listen_for_keywords(self, stream):
        """
        :param: stream
        :returns:
            recognized text
        """
        stream, rec, recognized_text = self.configure()
        list_stat1 = True
        while list_stat1:
            data = stream.read(4000, exception_on_overflow=False)
            rec.AcceptWaveform(data)
            partial_result = rec.PartialResult()

            if partial_result:
                partial_text = json.loads(partial_result).get("partial", "")
                if "listen" in partial_text.lower():
                    response = "Waking up ! Listening for input..."
                    speech_output(response)
                    text_prompt = self.listen_for_speech_prompt(stream)
                    list_stat1 = False
                    return True,text_prompt
                elif "stop" in partial_text.lower():
                    response= "Stop listening detected. Stopping..."
                    speech_output(response)
                    list_stat1 = False
                    text_prompt = "stop"
                    return False,text_prompt
                elif "write only mode" in partial_text.lower():
                    response= "Write only mode detected. Switching from speech to writting mode..."
                    speech_output(response)
                    list_stat1 = False
                    text_prompt = "write only mode"
                    return False,text_prompt
                else:
                    response= "No keyword detected. Speek to issue an input..."
                    speech_output(response)
                    list_stat1 = True

    def listen_for_speech_prompt(self, stream):
        stream, rec, recognized_text = self.configure()
        list_stat2 = True
        while list_stat2:
            data = stream.read(4000, exception_on_overflow=False)
            rec.AcceptWaveform(data)
            partial_result = rec.PartialResult()

            if partial_result:
                partial_text = json.loads(partial_result).get("partial", "")
                recognized_text += partial_text 
                print("Partial Result:", partial_text)

                if "open browser" in partial_text.lower():
                    response ="Opening the browser..."
                    speech_output(response)
                    subprocess.run(["start", "https://www.google.com"])
                if keyboard.is_pressed('p'):
                    response=''' KeyboardInterrupt: Stopping real-time listening
                        recognized text being: '''
                    speech_output(response)
                    list_stat2 = False
                    return recognized_text

    def real_time_listen(self):
        """
        initialize two process:
            listening for keywords: Should be active troughout the whole speech process
            listening for speech prompts
        """
        stream, rec, recognized_text = self.configure()
        try:
            while True:
                speech_stat,speech_input = self.listen_for_keywords(stream)
                if speech_stat:
                    response= f"Recognized text: {speech_input}"
                    speech_output(response)
                    response= "going back to another session of listening. to listing keywords"
                    speech_output(response)
                elif speech_input == "write only mode":
                    response="write only mode at real time listen, switching to write only, roger that"
                    speech_output(response)
                    break
                elif speech_input == "stop":
                    response="stop keyword detected at real time listen, stopping the process. "
                    speech_output(response)
                    break


        except KeyboardInterrupt:
            response="KeyboardInterrupt: Stopping real-time listening"
            speech_output(response)

        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    model_path = "D:\\vizuosense_mine\\STT\\Resources\\vosk-model-small-en-us-0.15"
    model_name = "vosk-model-small-en-us-0.15"
    language = "small-en-us"
    save_textfile_dir = "D:\\vizuosense_mine\\STT\\Resources\\test.txt"
    stt_engine = SpeechToTextEngine(model_path, model_name, language, save_textfile_dir)
    stt_engine.real_time_listen()

if __name__ == "__main__":
    main()
