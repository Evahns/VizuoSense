import cv2 as cv
import time, datetime, pytesseract, os
import  pyttsx3

class OCR_ENGINE:
    def __init__(self, image_path, text_file_path):
        self.image_path = image_path
        self.text_file_path = text_file_path

    def ocr(self,image_path, text_file_path):
            # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
            # we need to convert from BGR to RGB format/mode:
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            extracted_text = pytesseract.image_to_string(image)
            # specify the text file path to save the extracted text
            with open(f"{text_file_path}\Saved_at_.txt", "w+") as read_text:
                read_text.write(extracted_text)
                write_time = str(datetime.datetime.now().date()) +' ' + str(datetime.datetime.now().hour) + ' ' +  str(datetime.datetime.now().minute) +' ' +  str(datetime.datetime.now().second)
                save_name= 'Saved_at_' + write_time +'.txt'
                read_text.seek(0)
                text_content = read_text.read()
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                #print(engine.getProperty('rate')) #printing current voice rate
                print(text_content)
                engine.say(text_content)
                engine.runAndWait()
                #renaming the file for better readability
                os.rename(f"{text_file_path}\Saved_at_.txt",
                          f"{text_file_path}\ {save_name}")

def main():
    create_text_file_path = "D:\programming\Python\Scripts and codes\Computer vision\Resources\save"
    image = "D:\programming\Python\Scripts and codes\Computer vision\Resources\Test3.png"
    ocr_engine = OCR_ENGINE(image, create_text_file_path)  # create an instance of OCR_ENGINE
    ocr_engine.ocr(image, create_text_file_path)  # call the ocr method on the instance

if __name__ == "__main__":
    main()