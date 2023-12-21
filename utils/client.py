import cv2
import requests
import base64

def convert_image_to_byte_string(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to a byte string
    _, img_buffer = cv2.imencode('.png', image)
    image_byte_string = base64.b64encode(img_buffer.tobytes()).decode('utf-8')

    return image_byte_string

def send_api_request(image_byte_string, task):
    url = 'http://your-colab-ip-address:5000/api/process_image'  # Replace with your Google Colab IP address

    if task == 'navigation':
        #Write a navigation prompt
        prompt = "You are a navigation assistant. Describe a way in which a user should navigate the environment using the image provided."
    else:
        #Write an image description prompt
        prompt = "You are a helpful assistant. Describe the contents of the image"


    data = {
        'image_byte_string': image_byte_string,
        'prompt': prompt
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print(f"Server response: {response.json()['message']}")
    else:
        print(f"Error {response.status_code}: {response.json()['error']}")

# Example usage:
image_path = 'path/to/your/image.jpg'  # Replace with the path to your image file
image_byte_string = convert_image_to_byte_string(image_path)
send_api_request(image_byte_string, task = 'description')
