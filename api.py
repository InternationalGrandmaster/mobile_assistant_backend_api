import os
import requests
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv # To load API key from .env file

# Load environment variables from a .env file (optional but recommended)
# Create a file named .env in the same directory as api.py
# Add this line to the .env file: GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Get API Key from Environment Variable (Recommended)
api_key = os.environ.get("GEMINI_API_KEY")

# Use the appropriate Gemini model endpoint (1.5 Flash is common)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# --- API Endpoint Definition ---
@app.route('/generate', methods=['POST']) # Define endpoint path and allowed method
def generate_content():
    """
    API endpoint to receive a prompt and return Gemini's response.
    Expects JSON input: {"prompt": "Your prompt text here"}
    Returns JSON output: {"generated_text": "..."} or {"error": "..."}
    """
    # --- Security Check: Ensure API Key is configured ---
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set in the environment.")
        # Return an error response to the client
        return jsonify({"error": "API key not configured on the server"}), 500 # Internal Server Error

    # --- Input Validation: Get prompt from request ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400 # Bad Request

    data = request.get_json()
    prompt_text = data.get('prompt') # Use .get for safer access

    if not prompt_text:
        return jsonify({"error": "Missing 'prompt' key in JSON request"}), 400 # Bad Request

    # --- Prepare the call to Gemini API ---
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
      "contents": [{
        "parts":[{"text": prompt_text}] # Use the prompt received from the request
        }]
    }
    full_url = f"{GEMINI_API_URL}?key={api_key}"

    # --- Make the API Call to Gemini ---
    try:
        response = requests.post(full_url, headers=headers, json=payload, timeout=60) # Add a timeout

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response from Gemini
        response_data = response.json()

        # Extract the generated text (add robust error checking)
        try:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            # Return the successful result
            return jsonify({"generated_text": generated_text})
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing Gemini response structure: {e}")
            print(f"Full Gemini Response: {json.dumps(response_data)}")
            return jsonify({"error": "Failed to parse response from Gemini API"}), 500

    except requests.exceptions.Timeout:
        print("Error: Request to Gemini API timed out.")
        return jsonify({"error": "Gemini API request timed out"}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        # Handle connection errors, HTTP errors, etc.
        error_message = f"Error calling Gemini API: {e}"
        status_code = 502 # Bad Gateway, indicates upstream error
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                # Try to include Gemini's error message if available
                error_details = e.response.json()
                error_message = f"Gemini API Error (Status {status_code}): {json.dumps(error_details)}"
            except json.JSONDecodeError:
                error_message = f"Gemini API Error (Status {status_code}): {e.response.text}"

        print(error_message)
        # Return an error response to the client
        # Avoid exposing raw error details unless necessary for debugging client-side
        return jsonify({"error": "Failed to communicate with the backend generative API"}), status_code
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500



@app.route('/', methods=['GET'])
def index():
  """ A simple endpoint to check if the API is running. """
  return jsonify({"message": "API is running!"})

# --- Run the Flask Application ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make the API accessible from other devices on your network
    # Set debug=True for development (provides auto-reload and better error pages)
    # IMPORTANT: Turn debug=False for production deployments!
    app.run(host='0.0.0.0', port=5000, debug=True)