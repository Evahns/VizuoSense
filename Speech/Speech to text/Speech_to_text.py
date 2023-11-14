
import wave
import sys
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

rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())

print(rec.FinalResult())
