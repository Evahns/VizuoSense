
import wave
import sys
import pyaudio as pa
import speech_recognition as sr
from vosk import Model, KaldiRecognizer, SetLogLevel
import json

#the path to the model that is to be used for the transcription, follow instruction in the README.md file to download the model
model = Model(model_path="S:\\programs\\vosk-model-small-en-us-0.15", model_name="vosk-model-small-en-us-0.15", lang="small-en-us")
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
# wf = wave.open("S:\\vs code\\speech\\test.wav", "rb")

#  innitialize pyAudio
p = pa.PyAudio()

# open a stream with the system microphone as the input source
stream = p.open(format=pa.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

#rec = KaldiRecognizer(model, wf.getframerate())
# Initialize the KaldiRecognizer with the Vosk model
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)
rec.SetPartialWords(True)

recognized_text = ""
r = sr.Recognizer()
# Capture audio from the microphone and perform real-time speech recognition
while True:
    #data = wf.readframes(4000)
    print("start talking")

    #  read audio data from the microphone
    data = stream.read(4000,exception_on_overflow=False)

    #  check if there is no more
    if len(data) == 0:
        break

    #  Process the audio data using the KaldiRecognizer
    rec.AcceptWaveform(data)
    #     print(rec.Result())
    # else:
    #     print(rec.PartialResult())
        # Get the recognized text from the partial result
    partial_result = rec.PartialResult()
    if partial_result:
        partial_text = json.loads(partial_result).get("partial", "")
        recognized_text += partial_text 
        print("Partial Result:", partial_text)
       

# Save the recognized text to a file
output_file_path = "S:\\vs code\\speech\\recognized_text.txt"
with open(output_file_path, "w") as output_file:
    output_file.write(recognized_text)


    
#  Get the final result after the loop ends
final_result = rec.FinalResult()
final_text = json.loads(final_result)
print(final_text)

#  Stop and close the audio stream
#print(rec.FinalResult())
stream.stop_stream()
stream.close()

#  terminate Audio
p.terminate()


print(f"Recognized text saved to: {output_file_path}")
