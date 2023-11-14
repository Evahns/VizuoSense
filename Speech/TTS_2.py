import  pyttsx3
import json, logging as log
log.basicConfig(level=log.INFO)

text_path = "D:/vizuosense_mine/STT/Resources/text_file.txt"
def text_to_speech(path):
    with open(path, "r+") as to_read:
        to_read.seek(0)
        read = to_read.read()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        log.info(read)
        print(engine.getProperty('rate'))
        engine.say(read)
        engine.runAndWait()


text_to_speech(text_path)