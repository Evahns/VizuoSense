
import wave
import sys
import pyaudio as pa
from vosk import Model, KaldiRecognizer, SetLogLevel

#the path to the model that is to be used for the transcription, follow instruction in the README.md file to download the model
model = Model(model_path="D:\\vizuosense_mine\\STT\Resources\\vosk-model-small-en-us-0.15", model_name="vosk-model-small-en-us-0.15", lang="small-en-us")
# You can set log level to -1 to disable debug messages
SetLogLevel(0)

'''
#Getting the audio file that is specified in the command line
 wf = wave.open(sys.argv[1], "rb")
 if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
     print("Audio file must be WAV format mono PCM.")
     sys.exit(1)
'''
#Path to the audio file that is to be transcribed
#The file has to be in .wav format at least for now
wf = wave.open("D:\\vizuosense_mine\\VizuoSense\\Resources\\test.wav", "rb")

p = pa.PyAudio()
stream = p.open(format=pa.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

#rec = KaldiRecognizer(model, wf.getframerate())
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)
rec.SetPartialWords(True)

while True:
    #data = wf.readframes(4000)
    print("start talking")
    data = stream.read(4000,exception_on_overflow=False)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)
    #     print(rec.Result())
    # else:
    #     print(rec.PartialResult())
final_result = rec.FinalResult()
final_text = json.loads(final_result)
print(final_text)

#print(rec.FinalResult())
stream.stop_stream()
stream.close()
p.terminate()