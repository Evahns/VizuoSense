from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import logging as log

# Set up the root logger to output all messages of level INFO and above
log.basicConfig(level=log.INFO)
model_path = "C:/Users/Hp Elitebook 8470p/AppData/Local/tts/tts_models--en--ljspeech--tacotron2-DDC/model_file.pth"
config_path = "C:/Users/Hp Elitebook 8470p/AppData/Local/tts/tts_models--en--ljspeech--tacotron2-DDC/config.json"

# Load the TTS model
log.info("Loading the TTS model")
synthesizer = Synthesizer(model_path, config_path)
# Read the text file
print("Reading the text file from text file")
with open("D:/vizuosense_mine/Resources/Saves/text1.txt", "r") as f:
    text = f.read()

# Generate speech from the text
audio = synthesizer.tts(text)
log.info("Finished generating audio from the text")
sample_rate = synthesizer.tts_config['audio']['sample_rate']
# Specify the output path for the generated audio
output_path = "D:/vizuosense_mine/Resources/Saves/output.wav"

# Save the generated audio to the output path
sf.write(output_path, audio, sample_rate)
log.info("Saved the generated audio to the output path")