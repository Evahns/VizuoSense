import os
import threading
import subprocess
from flask import Flask, jsonify, request
from pyngrok import ngrok

os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# Define Flask routes
@app.route("/")
def index():
    return "Hello from Colab!"

# API endpoint
@app.route("/api/data", methods=["GET"])
def get_data():
    data = {"message": "This is an example API endpoint", "status": "success"}
    return jsonify(data)

@app.route('/api/endpoint', methods=["POST"])
def api_call():
    # Extract data from the incoming POST request
    data = request.json

    # Replace placeholders with actual values from the request data
    llava_cli_command = ['./llava-cli', '-m', "/content/llama.cpp/ggml-model-q4_k.gguf",
                         '--mmproj',  "/content/llama.cpp/mmproj-model-f16.gguf",
                         '--image', data.get('image_path', ''),
                         '--temp', str(data.get('temperature', 0.1)),
                         '-p', data.get('image_description', '')]

    try:
        # Run the llava-cli command using subprocess
        result = subprocess.run(llava_cli_command, capture_output=True, text=True, check=True)

        # Return the command output as JSON
        return jsonify({'output': result.stdout, 'error': result.stderr}), 200
    except subprocess.CalledProcessError as e:
        # If an error occurs, return the error message and status code 500
        return jsonify({'error': str(e)}), 500


# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()