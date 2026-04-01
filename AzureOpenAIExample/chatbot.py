"""
chatbot.py
----------
A simple chatbot web server built with Flask and Azure OpenAI.

How it works:
  - Flask serves the chat page when you open the browser
  - When you send a message, it goes to the /chat route
  - The /chat route calls Azure OpenAI and streams the reply back
  - The browser receives the reply word by word (streaming)

How to run:
  python chatbot.py

Then open your browser at:
  http://127.0.0.1:5000

Requirements:
  pip install flask openai python-dotenv

Environment variables needed in your .env file:
  AZURE_OPENAI_API_KEY      - your Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT     - your Azure OpenAI endpoint URL
  AZURE_OPENAI_API_VERSION  - API version e.g. 2024-02-01
  AZURE_OPENAI_DEPLOYMENT   - your deployment name e.g. gpt-4o
"""

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from flask import Flask, render_template, request, jsonify, Response

# load the .env file so we can read our secret keys
load_dotenv()

# read the azure settings from .env file
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# create the azure openai client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# create the flask app
app = Flask(__name__)


@app.route("/")
def index():
    """
    Serves the main chat page.
    Flask will look for index.html inside the templates/ folder.
    """
    return render_template("index.html")


@app.route("/model")
def get_model():
    """
    Returns the Azure deployment name as JSON.
    The frontend calls this on page load to show the model name in the header.

    Example response:
        { "deployment": "gpt-4o" }
    """
    return jsonify({"deployment": AZURE_DEPLOYMENT})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives the conversation history from the browser and streams
    the AI reply back token by token using Server-Sent Events (SSE).

    Expected request body (JSON):
        {
            "messages": [
                { "role": "user",      "content": "Hello!" },
                { "role": "assistant", "content": "Hi there!" },
                { "role": "user",      "content": "How are you?" }
            ]
        }

    The full message history is sent each time so the AI remembers
    the whole conversation.

    Response format (SSE stream):
        data: "Hello"
        data: " there"
        data: "!"
        data: [DONE]

    Each token is JSON-encoded so that newlines and special characters
    don't break the SSE format.
    """
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    def generate():
        """Calls Azure OpenAI with streaming and yields each token as an SSE line."""
        try:
            stream = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    # json.dumps encodes the token so newlines don't break SSE
                    yield f"data: {json.dumps(token)}\n\n"

            # tell the browser the stream is finished
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps('[ERROR] ' + str(e))}\n\n"
            yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# start the server
if __name__ == "__main__":
    print("Endpoint  :", AZURE_ENDPOINT)
    print("Deployment:", AZURE_DEPLOYMENT)
    print("Open your browser at http://127.0.0.1:5000")
    app.run(debug=False, threaded=True)
