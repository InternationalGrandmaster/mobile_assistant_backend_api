# api.py
# This script runs a Flask web server acting as a backend API.
# It primarily receives audio files, processes them using the Gemini API
# to get transcriptions, summaries, speaker info, and action items,
# and returns the results. It also keeps the original text generation endpoint.

import os
import requests # Makes HTTP requests (to Gemini API)
import json     # Handles JSON data
import tempfile # Creates temporary files/directories safely
import mimetypes # Guesses file types (like audio/wav)
from flask import Flask, request, jsonify # Core Flask components for web app, requests, JSON responses
from dotenv import load_dotenv # Loads environment variables from a .env file for local development

# Load environment variables from .env file if it exists.
# This is mainly for local development convenience. On Render,
# environment variables are set directly in the service settings.
load_dotenv()

# Initialize the Flask web application instance
app = Flask(__name__)

# --- Configuration ---

# Get the Gemini API Key from environment variables.
# IMPORTANT: This *must* be set in the deployment environment (e.g., Render).
api_key = os.environ.get("GEMINI_API_KEY")

# Define the base URL for the Gemini API we're using
# Using v1beta as it often has the latest features like the File API
FILE_API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Specify the Gemini model to use.
# 1.5 Flash is a good balance of speed, cost, and multimodal capability.
# The "-latest" tag ensures we use the most recent stable version of this model.
MODEL_NAME = "gemini-1.5-flash-latest"

# Construct the full URLs for the Gemini API endpoints
# Endpoint for generating content (text or based on file references)
GENERATE_CONTENT_URL = f"{FILE_API_URL_BASE}/models/{MODEL_NAME}:generateContent?key={api_key}"
# Endpoint specifically for uploading files (like audio)
FILE_UPLOAD_URL = f"{FILE_API_URL_BASE}/files?key={api_key}"


# --- Helper Function: Upload File to Gemini ---

def upload_to_gemini_file_api(file_path, mime_type):
    """
    Uploads a local file to the Gemini File API.

    This is a necessary first step before asking Gemini to process
    content based on a file (like audio). It returns a dictionary
    containing the Gemini file reference (name, URI, etc.).

    Args:
        file_path (str): The local path to the file to upload.
        mime_type (str): The MIME type of the file (e.g., 'audio/wav').

    Returns:
        dict: A dictionary containing the uploaded file's info from Gemini,
              including 'name' and 'uri'.

    Raises:
        ConnectionError: If the upload request fails.
        ValueError: If the Gemini API response is unexpected.
    """
    print(f"Attempting to upload file: {file_path}, MIME: {mime_type}")
    try:
        # Open the file in binary read mode
        with open(file_path, 'rb') as f:
            # Prepare the file for a multipart/form-data upload
            # The key 'file' is expected by the Gemini File API
            files = {'file': (os.path.basename(file_path), f, mime_type)}

            # Make the POST request to upload the file
            # Increased timeout because file uploads can take longer
            upload_response = requests.post(FILE_UPLOAD_URL, files=files, timeout=120)

            # Check if the upload resulted in an HTTP error (4xx or 5xx)
            upload_response.raise_for_status()

            # Parse the JSON response from the Gemini File API
            upload_data = upload_response.json()
            # print(f"Gemini File API Response: {json.dumps(upload_data, indent=2)}") # Uncomment for debugging response structure

            # The Gemini API response structure can sometimes vary slightly.
            # Check for the expected nested 'file' object first.
            if 'file' in upload_data and 'name' in upload_data['file'] and 'uri' in upload_data['file']:
                 file_info = upload_data['file']
                 print(f"File upload successful (nested structure). Name: {file_info['name']}, URI: {file_info['uri']}")
                 return file_info # Contains 'name', 'uri', 'mimeType', etc.
            # Check if the response might be flat (name/uri directly at the root)
            elif 'name' in upload_data and upload_data['name'].startswith('files/'):
                 print(f"File upload successful (flat structure). Name: {upload_data['name']}, URI: {upload_data.get('uri')}")
                 # Return a dictionary consistent with the nested structure if possible
                 return {
                     'name': upload_data['name'],
                     'uri': upload_data.get('uri'), # URI might be optional sometimes
                     'mimeType': upload_data.get('mimeType', mime_type) # Use original if not returned
                 }
            else:
                # If we can't find the necessary info, log it and raise an error
                print(f"Unexpected File API response structure: {json.dumps(upload_data, indent=2)}")
                raise ValueError("Upload succeeded but response lacked expected 'file' details (name/uri).")

    # Handle potential network errors during upload
    except requests.exceptions.RequestException as e:
        print(f"Error during Gemini File API upload request: {e}")
        # Log response details if available, helpful for debugging API key/permission issues
        if hasattr(e, 'response') and e.response is not None:
            print(f"Upload Response Status: {e.response.status_code}")
            print(f"Upload Response Body: {e.response.text}")
        # Re-raise as a more specific error for the calling function to handle
        raise ConnectionError(f"Failed to upload file to Gemini processing service: {e}") from e
    # Catch any other unexpected exceptions during the upload process
    except Exception as e:
        print(f"An unexpected error occurred during file upload helper: {e}")
        raise # Re-raise the original exception


# --- API Endpoints ---

@app.route('/process-audio', methods=['POST'])
def process_audio_file():
    """
    Handles POST requests with an uploaded audio file.
    Orchestrates the process:
    1. Receives audio file from the request.
    2. Saves it temporarily.
    3. Uploads it to the Gemini File API.
    4. Calls Gemini's generateContent with the file reference and a detailed prompt.
    5. Returns Gemini's generated text (transcription, summary, etc.).
    """
    # --- Pre-checks ---
    # Make sure the API key is actually configured in the environment. Crucial.
    if not api_key:
        print("FATAL: /process-audio called but GEMINI_API_KEY is not set.")
        return jsonify({"error": "API key not configured on the server"}), 500

    # Check if the 'audio_file' part exists in the incoming request's files
    if 'audio_file' not in request.files:
        print("Request missing 'audio_file' part.")
        return jsonify({"error": "Missing 'audio_file' in request files"}), 400 # Bad Request

    audio_file = request.files['audio_file']

    # Check if a file was actually selected by the user
    if audio_file.filename == '':
        print("Request contains 'audio_file' but no file was selected.")
        return jsonify({"error": "No selected audio file"}), 400 # Bad Request

    # --- File Handling ---
    # Using a temporary directory is safer and helps with cleanup
    temp_dir = tempfile.gettempdir()
    # Create a temporary path using the original filename (be cautious if filenames could be malicious)
    # Consider sanitizing or generating a unique temp filename in production
    temp_path = os.path.join(temp_dir, audio_file.filename)
    uploaded_file_info = None # Will store the result from upload_to_gemini_file_api

    try:
        # Save the uploaded file stream to the temporary path
        audio_file.save(temp_path)
        print(f"Audio file temporarily saved to: {temp_path}")

        # --- Determine MIME Type ---
        # Guess the MIME type based on the file extension
        mime_type, _ = mimetypes.guess_type(temp_path)
        # If guessing fails or it's not identified as audio, try a default or log a warning
        if not mime_type or not mime_type.startswith('audio/'):
            # Consider using python-magic for more robust detection if needed,
            # but that adds dependencies (libmagic).
            mime_type = 'application/octet-stream' # A generic binary type
            print(f"Warning: Could not reliably determine audio MIME type for '{audio_file.filename}'. Using default: {mime_type}.")
        else:
            print(f"Guessed MIME type: {mime_type} for '{audio_file.filename}'")


        # --- Step 1: Upload to Gemini File API ---
        try:
             uploaded_file_info = upload_to_gemini_file_api(temp_path, mime_type)
             # We absolutely need the URI to reference the file later
             if not uploaded_file_info or 'uri' not in uploaded_file_info:
                 print("ERROR: File upload seemed successful but didn't return a usable file reference (URI).")
                 return jsonify({"error": "File uploaded but failed to get valid reference from processing service"}), 502 # Bad Gateway - upstream issue
        except Exception as upload_err:
              # If the helper function raised an error (e.g., ConnectionError)
              print(f"ERROR: Upload to Gemini File API failed. Error: {upload_err}")
              # Propagate a user-friendly error
              return jsonify({"error": f"Failed to upload audio to processing service: {upload_err}"}), 502 # Bad Gateway


        # --- Step 2: Call Gemini GenerateContent ---
        print(f"Calling Gemini GenerateContent for file URI: {uploaded_file_info['uri']}")

        # This prompt is key! It tells Gemini *what* to do with the audio.
        # Be specific about the desired output format and content.
        prompt = (
            "Please analyze the provided audio file referenced below.\n"
            "1.  Generate an accurate transcription of the conversation.\n"
            "2.  Identify the different speakers in the transcript. Label them clearly (e.g., 'Speaker A:', 'Speaker B:'). If only one speaker, note that.\n"
            "3.  Provide a concise summary of the main topics discussed in the audio.\n"
            "4.  If this audio appears to contain a meeting or discussion about tasks or actions, list any specific commitments or action items mentioned. Include who is responsible if specified.\n\n"
            "Present the results clearly, perhaps using markdown sections for TRANSCRIPT, SPEAKERS, SUMMARY, and ACTION ITEMS for easy parsing later."
        )

        # Construct the payload for the generateContent request
        payload = {
            "contents": [{
                "parts": [
                    # Reference the previously uploaded file via its URI and MIME type
                    {"fileData": {"mimeType": uploaded_file_info['mimeType'], "fileUri": uploaded_file_info['uri']}},
                    # Include the detailed text prompt
                    {"text": prompt}
                ]
            }],
            # Optional: Configure generation parameters if needed
            # "generationConfig": {
            #     "temperature": 0.7,         # Controls randomness (0=deterministic, >1=more creative)
            #     "maxOutputTokens": 8192   # Limit response length (check model limits)
            # }
        }

        # Standard headers for a JSON POST request
        headers = { 'Content-Type': 'application/json' }

        # Make the POST request to the Gemini generateContent endpoint
        # Increase timeout as audio processing + generation can take significant time
        response = requests.post(GENERATE_CONTENT_URL, headers=headers, json=payload, timeout=300) # 5 minutes timeout

        # Check for HTTP errors from the Gemini API itself
        response.raise_for_status()

        # Parse the successful JSON response from Gemini
        response_data = response.json()
        # print(f"Gemini Generate Content Response: {json.dumps(response_data, indent=2)}") # Uncomment for debugging

        # --- Step 3: Extract Results from Gemini Response ---
        try:
            # Gemini's response structure usually involves 'candidates' -> 'content' -> 'parts'
            # It's good practice to check existence safely using .get()
            candidates = response_data.get('candidates', [])
            if not candidates:
                 # This might happen if the content was blocked for safety, etc.
                 print("ERROR: No candidates found in Gemini response.")
                 # Check for block reasons if available (not shown here, see Gemini docs)
                 return jsonify({"error": "Processing service returned no content, possibly due to safety filters or an issue."}), 500

            # Assume the first candidate holds the main result
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            if not parts:
                 print("ERROR: No 'parts' found in the candidate content.")
                 return jsonify({"error": "Processing service returned empty content parts."}), 500

            # Combine the text from all 'parts' into a single string
            generated_text = "\n".join([part.get('text', '') for part in parts]).strip()

            # Check if we actually got any text back
            if not generated_text:
                 # Check why generation might have stopped (e.g., length limits, safety)
                 finish_reason = candidates[0].get('finishReason')
                 safety_ratings = candidates[0].get('safetyRatings') # Provides safety category info
                 print(f"Warning: Empty text result from Gemini. Finish Reason: {finish_reason}, Safety: {safety_ratings}")
                 # Decide if this is an error or just an empty valid result
                 if finish_reason and finish_reason != 'STOP':
                     # If it stopped for reasons like MAX_TOKENS or SAFETY, treat as error/warning
                     return jsonify({"error": f"Content generation stopped unexpectedly. Reason: {finish_reason}"}), 500
                 else:
                     # If it stopped normally but produced no text (maybe silent audio?), return empty
                     return jsonify({"processed_text": ""}) # Or maybe an informative message

            # Successfully extracted text, return it to the client
            print("Successfully processed audio and extracted text from Gemini.")
            return jsonify({"processed_text": generated_text})

        # Handle potential errors if the Gemini response structure is not as expected
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"ERROR: Failed to parse the structure of the Gemini response: {e}")
            print(f"Full Gemini Response was: {json.dumps(response_data, indent=2)}")
            return jsonify({"error": "Failed to parse response from processing service after generation"}), 500

    # --- Error Handling for GenerateContent Call ---
    except requests.exceptions.Timeout:
        print("ERROR: Request to Gemini API timed out during content generation.")
        return jsonify({"error": "Processing service request timed out during analysis"}), 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        # Handle network or HTTP errors during the generateContent call
        error_message = f"Error calling Gemini generateContent for audio: {e}"
        status_code = 502 # Bad Gateway default for upstream errors
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            # Try to get more specific error details from Gemini's response body
            try:
                error_details = e.response.json()
                error_message = f"Gemini API Error (Status {status_code}): {json.dumps(error_details)}"
            except json.JSONDecodeError: # If response isn't JSON
                error_message = f"Gemini API Error (Status {status_code}): {e.response.text}"
        print(f"ERROR: {error_message}")
        # Return a generic error to the client, hiding potentially sensitive details
        return jsonify({"error": "Failed to communicate with the backend generative API for processing"}), status_code
    # Catch any other unexpected errors during the whole process
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in /process-audio: {e}", exc_info=True) # Log traceback
        return jsonify({"error": f"An internal server error occurred during processing: {e}"}), 500
    # --- Cleanup ---
    finally:
        # Ensure the temporary file is deleted regardless of success or failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file cleaned up: {temp_path}")
            except Exception as e:
                # Log error but don't crash the response if cleanup fails
                print(f"Warning: Error removing temporary file {temp_path}: {e}")

        # Optional: Delete the file from Gemini File API using its 'name'
        # This requires making a DELETE request to FILE_API_URL_BASE + '/' + uploaded_file_info['name']
        # Consider lifecycle policies if files should expire automatically instead.
        # if uploaded_file_info and 'name' in uploaded_file_info:
        #    try:
        #        delete_url = f"{FILE_API_URL_BASE}/{uploaded_file_info['name']}?key={api_key}"
        #        delete_response = requests.delete(delete_url, timeout=30)
        #        print(f"Attempted Gemini file deletion for {uploaded_file_info['name']}, status: {delete_response.status_code}")
        #    except Exception as delete_err:
        #        print(f"Warning: Error deleting Gemini file {uploaded_file_info.get('name')}: {delete_err}")


@app.route('/generate', methods=['POST'])
def generate_content_text():
    """
    Handles POST requests with a simple text prompt.
    This is the original endpoint for basic text generation.
    """
    # --- Pre-checks ---
    if not api_key:
        print("FATAL: /generate called but GEMINI_API_KEY is not set.")
        return jsonify({"error": "API key not configured on the server"}), 500

    # --- Input Validation ---
    # Ensure request is JSON
    if not request.is_json:
        print("Request to /generate is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400
    # Get data and check for 'prompt' key
    data = request.get_json()
    prompt_text = data.get('prompt')
    if not prompt_text:
        print("Request to /generate missing 'prompt' key.")
        return jsonify({"error": "Missing 'prompt' key in JSON request"}), 400

    # --- Prepare and Call Gemini API ---
    print(f"Calling Gemini GenerateContent for text prompt: '{prompt_text[:50]}...'")
    headers = { 'Content-Type': 'application/json' }
    payload = { "contents": [{ "parts":[{"text": prompt_text}] }] }
    full_url = f"{GENERATE_CONTENT_URL}?key={api_key}" # Rebuild URL here in case key wasn't ready at startup

    try:
        # Make the request to Gemini
        response = requests.post(full_url, headers=headers, json=payload, timeout=60) # 1 minute timeout
        response.raise_for_status() # Check for HTTP errors
        response_data = response.json()

        # --- Extract Result ---
        try:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            print("Successfully generated text response.")
            return jsonify({"generated_text": generated_text})
        # Handle parsing errors if Gemini response format is unexpected
        except (KeyError, IndexError, TypeError) as e:
            print(f"ERROR: Failed to parse Gemini response structure for text prompt: {e}")
            print(f"Full Gemini Response: {json.dumps(response_data)}")
            return jsonify({"error": "Failed to parse response from processing service"}), 500

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        print("ERROR: Request to Gemini API timed out for text prompt.")
        return jsonify({"error": "Processing service request timed out"}), 504
    except requests.exceptions.RequestException as e:
        # Handle network or HTTP errors from Gemini
        error_message = f"Error calling Gemini API for text prompt: {e}"
        status_code = 502
        if hasattr(e, 'response') and e.response is not None:
            # Log details if possible
            status_code = e.response.status_code
            try: error_details = e.response.json(); error_message = f"Gemini API Error (Status {status_code}): {json.dumps(error_details)}"
            except json.JSONDecodeError: error_message = f"Gemini API Error (Status {status_code}): {e.response.text}"
        print(f"ERROR: {error_message}")
        return jsonify({"error": "Failed to communicate with the backend generative API"}), status_code
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in /generate: {e}", exc_info=True) # Log traceback
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route('/', methods=['GET'])
def index():
  """
  A simple root endpoint. Useful for quickly checking if the API
  service is running and reachable after deployment.
  """
  # Could also check if api_key is loaded here and return status
  status = "API Key Configured" if api_key else "API Key MISSING!"
  return jsonify({"message": f"Audio Processing API is running! Status: {status}"})

# --- Main execution block ---
# This part only runs when you execute the script directly (e.g., `python api.py`)
# It's used for LOCAL DEVELOPMENT ONLY. Gunicorn/Render will bypass this
# and directly use the 'app' instance defined above.
if __name__ == '__main__':
    # Check if the API key is loaded locally before starting
    if not api_key:
        print("*" * 60)
        print(" WARNING: GEMINI_API_KEY not found in environment variables or .env file.")
        print(" The API will likely fail when called.")
        print(" Ensure you have a .env file with GEMINI_API_KEY='YourKey' next to api.py")
        print("*" * 60)

    # Run the Flask development server
    # host='0.0.0.0' makes it accessible from other devices on your network
    # debug=True enables auto-reloading on code changes and detailed error pages
    # WARNING: NEVER run with debug=True in a production environment!
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)