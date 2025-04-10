from flask import Flask, render_template, request, jsonify, g
from openai import OpenAI
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import time
import uuid
from flask_cors import CORS
import traceback # For detailed error logging
from functools import lru_cache
from datetime import datetime, timedelta

# Cache configuration
CACHE_TIMEOUT = 300  # 5 minutes cache timeout
API_TIMEOUT = 5  # 5 seconds API timeout

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure file handler with rotation
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'app.log'),
    maxBytes=1024 * 1024,  # 1MB per file
    backupCount=10  # Keep up to 10 backup files
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s'
))

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s'
))

# Set up logger
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Request ID context
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, 'request_id', 'N/A')
        return True

logger.addFilter(RequestIdFilter())

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
client = OpenAI()

@app.before_request
def log_request_info():
    g.request_start_time = time.time()
    g.request_id = str(uuid.uuid4())
    logger.info(
        f'Incoming Request: {request.method} {request.path} | Headers: {request.headers}', 
        extra={'request_id': g.request_id}
    )

@app.after_request
def log_response_info(response):
    latency = time.time() - g.request_start_time
    logger.info(
        f'Outgoing Response: {response.status_code} | Latency: {latency:.4f}s', 
        extra={'request_id': g.request_id}
    )
    return response

# Cache decorator for API responses
def cache_response(timeout=CACHE_TIMEOUT):
    def decorator(func):
        @lru_cache(maxsize=100)
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)
        return cached_func
    return decorator

@cache_response()
def retrieve_all_activities(limit=3, portfolio_name=None, request_id=None):
    log_extra = {'request_id': request_id or getattr(g, 'request_id', 'N/A')}
    try:
        API_URL = os.getenv('ACTIVITIES_API_URL', 'https://eurac.goantares.uno/public/api/v1/Activities')
        AUTH_KEY = os.getenv('AUTH_KEY')
        AUTH_SECRET = os.getenv('AUTH_SECRET')

        headers = {"accept": "application/json"}
        auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)

        params = {
            "$filter": f"PortfolioName eq '{portfolio_name}'" if portfolio_name else None,
            "$top": int(limit),
            "$orderby": "CreatedDate desc"
        }

        params = {k: v for k, v in params.items() if v is not None}

        logger.info(f"Making API request to {API_URL} with params: {params}", extra=log_extra)
        response = requests.get(API_URL, headers=headers, params=params, auth=auth, timeout=API_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            filtered_data = []
            for activity in data:
                filtered_activity = {k: v for k, v in activity.items() if v is not None}
                filtered_data.append(filtered_activity)

            logger.info(f"Retrieved {len(filtered_data)} activities successfully.", extra=log_extra)
            return {"activities": filtered_data}
        else:
            error_msg = f"API request failed with status code: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - Details: {json.dumps(error_details)}"
            except:
                error_msg += f" - Response text: {response.text}"
            logger.error(error_msg, extra=log_extra)
            return {"error": error_msg}

    except Exception as e:
        error_msg = f"Error retrieving activities: {str(e)}"
        logger.error(error_msg, exc_info=True, extra=log_extra)
        return {"error": error_msg}

@cache_response()
def retrieve_assets_by_type(short_description, request_id=None):
    log_extra = {'request_id': request_id or getattr(g, 'request_id', 'N/A')}
    try:
        API_URL = os.getenv('API_URL', 'https://eurac.goantares.uno/public/api/v3/Assets')
        AUTH_KEY = os.getenv('AUTH_KEY')
        AUTH_SECRET = os.getenv('AUTH_SECRET')
        
        headers = {"accept": "application/json"}
        auth = HTTPBasicAuth(AUTH_KEY, AUTH_SECRET)
        
        params = {
            "$filter": f"ShortDescription eq '{short_description}'",
            "$top": 1000
        }

        logger.info(f"Making API request to {API_URL} with params: {params}", extra=log_extra)
        response = requests.get(API_URL, headers=headers, params=params, auth=auth, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            filtered_data = []
            for asset in data:
                filtered_asset = {
                    'SecondaryCode': asset.get('SecondaryCode'),
                    'ShortDescription': asset.get('ShortDescription'),
                    'Model': asset.get('Model'),
                    'ExternalManufacturer': asset.get('ExternalManufacturer'),
                    'Coordinates': asset.get('Coordinates')
                }
                filtered_data.append(filtered_asset)
            
            logger.info(f"Retrieved {len(filtered_data)} assets successfully.", extra=log_extra)
            return {"assets": filtered_data}
        else:
            error_msg = f"API request failed with status code: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - Details: {json.dumps(error_details)}"
            except:
                error_msg += f" - Response text: {response.text}"
            logger.error(error_msg, extra=log_extra)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error retrieving assets: {str(e)}"
        logger.error(error_msg, exc_info=True, extra=log_extra)
        return {"error": error_msg}

# Initialize or get existing assistant
def get_or_create_assistant():
    try:
        # Check if we have a valid API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            raise ValueError("Invalid or missing OpenAI API key. Please set a valid API key in your .env file.")

        # Try to get existing assistant by ID
        assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
        if assistant_id:
            try:
                logger.info(f"Retrieving existing assistant with ID: {assistant_id}")
                assistant = client.beta.assistants.retrieve(assistant_id)
                logger.info("Successfully retrieved existing assistant")
                return assistant
            except Exception as e:
                logger.warning(f"Failed to retrieve existing assistant: {str(e)}. Creating new one...")

        # If no assistant_id or retrieval failed, create new assistant
        logger.info("Creating new assistant...")
        assistant = client.beta.assistants.create(
            name="Data Retriever V2",
            instructions="""You are a data retrieval agent for an asset management system. Your role is to retrieve data based on the user query.
        
            When asked about assets:
            - Use retrieve_assets_by_type to get asset information
            - Assets have fields: SecondaryCode, ShortDescription, Model, ExternalManufacturer, and Coordinates
            - Provide clear summaries of the assets found
            
            When asked about activities:
            - Use retrieve_all_activities to get recent activities
            - If no limit is specified, use a default of 3
            - Activities are ordered by creation date
            - You can filter activities by portfolio name if specified
            
            Be specific in your responses and include relevant details from the data retrieved.
            If there's an error in data retrieval, explain it clearly to the user.""",
            model="gpt-4",
            tools=[
                {"type": "function", "function": {
                    "name": "retrieve_assets_by_type",
                    "description": "Get assets by their short description type",
                    "parameters": {"type": "object", "properties": {
                        "short_description": {"type": "string", "description": "The short description to filter assets by"}
                    }, "required": ["short_description"]}}
                },
                {"type": "function", "function": {
                    "name": "retrieve_all_activities",
                    "description": "Get recent activities with optional filtering",
                    "parameters": {"type": "object", "properties": {
                        "limit": {"type": "integer", "description": "Number of activities to retrieve (default: 3)"},
                        "portfolio_name": {"type": "string", "description": "Optional portfolio name to filter by"}
                    }, "required": []}}
                }
            ]
        )
        
        # Log the new assistant ID for future use
        logger.info(f"Created new assistant with ID: {assistant.id}")
        logger.info("Please add this ID to your .env file as OPENAI_ASSISTANT_ID")
        
        return assistant
    except Exception as e:
        logger.error(f"Failed to create OpenAI Assistant during startup: {str(e)}", exc_info=True)
        raise

assistant = get_or_create_assistant()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat/start', methods=['POST', 'OPTIONS'])
def start_chat():
    # Handles CORS preflight request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        # Allow necessary headers for CORS
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        return response

    # Use request_id generated by before_request hook
    request_id = getattr(g, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}

    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data in request", extra=log_extra)
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        message_content = data.get('message')
        if not message_content:
            logger.error("No message provided in request", extra=log_extra)
            return jsonify({'error': 'No message provided', 'status': 'error'}), 400

        thread_id = data.get('thread_id') # Can be None for first message

        # 1. Get or Create Thread
        if not thread_id:
            logger.info("Creating new thread...", extra=log_extra)
            thread = client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"New thread created: {thread_id}", extra=log_extra)
        else:
            # Optional: Validate thread_id exists? For simplicity, assume valid if provided.
            logger.info(f"Using existing thread: {thread_id}", extra=log_extra)
            # Optional: Cancel any existing runs on this thread? Might be useful.
            # try:
            #     runs = client.beta.threads.runs.list(thread_id=thread_id, limit=10)
            #     for run in runs.data:
            #         if run.status in ['queued', 'in_progress', 'requires_action']:
            #             client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            #             logger.warning(f"Cancelled previous run {run.id} on thread {thread_id}", extra=log_extra)
            # except Exception as cancel_e:
            #     logger.error(f"Error cancelling previous runs for thread {thread_id}: {cancel_e}", extra=log_extra)


        # 2. Add Message to Thread
        logger.info(f"Adding message to thread {thread_id}: {message_content}", extra=log_extra)
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )

        # 3. Create Run
        logger.info(f"Creating run for thread {thread_id} with assistant {assistant.id}", extra=log_extra)
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant.id
            # instructions="Optional override for this run" # Example
        )
        logger.info(f"Run created: {run.id} with status: {run.status}", extra=log_extra)

        # 4. Return Run ID and Thread ID Immediately
        return jsonify({
            'run_id': run.id,
            'thread_id': thread_id,
            'status': 'started' # Indicate the process has started
        }), 202 # Accepted: request accepted, processing will occur

    except Exception as e:
        error_msg = f"Error starting chat run: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg, exc_info=True, extra=log_extra)
        return jsonify({'error': f"Internal Server Error: {str(e)}", 'status': 'error'}), 500

@app.route('/api/chat/status/<run_id>', methods=['GET'])
def get_chat_status(run_id):
    # Use request_id generated by before_request hook
    request_id = getattr(g, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}

    thread_id = request.args.get('thread_id')
    if not thread_id:
        logger.error("Missing thread_id query parameter", extra=log_extra)
        return jsonify({'error': 'Missing thread_id', 'status': 'error'}), 400

    logger.info(f"Checking status for run_id: {run_id}, thread_id: {thread_id}", extra=log_extra)

    try:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        logger.debug(f"Retrieved run status: {run.status}", extra=log_extra)

        if run.status == 'completed':
            logger.info(f"Run {run.id} completed. Retrieving messages...", extra=log_extra)
            messages = client.beta.threads.messages.list(thread_id=thread_id, order='desc', limit=1) # Get newest message
            assistant_message = "No response found."
            # Find the latest assistant message in the retrieved batch
            for msg in messages.data:
                 if msg.role == 'assistant':
                    if msg.content and msg.content[0].type == 'text':
                         assistant_message = msg.content[0].text.value
                         break # Found the latest assistant message

            logger.info(f"Final assistant message retrieved.", extra=log_extra)
            return jsonify({
                'status': 'completed',
                'response': assistant_message,
                'thread_id': thread_id # Return thread_id for consistency
            })

        elif run.status == 'requires_action':
            logger.info(f"Run {run.id} requires tool action.", extra=log_extra)
            # Handle tool calls synchronously for now. This GET request might take longer.
            handle_tool_calls(run, thread_id, request_id)
            # After handling (or attempting to handle), tell the frontend to keep polling
            return jsonify({'status': 'processing_tools'})

        elif run.status in ['queued', 'in_progress']:
            logger.debug(f"Run {run.id} is {run.status}.", extra=log_extra)
            return jsonify({'status': 'processing'})

        elif run.status in ['cancelling', 'cancelled', 'failed', 'expired']:
            error_message = f"Run {run.id} ended with status: {run.status}"
            if run.last_error:
                error_message += f" - Error: {run.last_error.code}: {run.last_error.message}"
            logger.error(error_message, extra=log_extra)
            return jsonify({
                'status': 'failed',
                'error': error_message,
                'run_status': run.status # Provide specific final status
            }), 500 # Internal Server Error might be appropriate for failed runs
        
        else: # Unknown status
             logger.warning(f"Run {run.id} has unknown status: {run.status}", extra=log_extra)
             return jsonify({'status': run.status}) # Return the unknown status

    except openai.NotFoundError:
         logger.error(f"Run not found - run_id: {run_id}, thread_id: {thread_id}", extra=log_extra)
         return jsonify({'error': 'Run or Thread not found', 'status': 'error'}), 404
    except Exception as e:
        error_msg = f"Error checking run status: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg, exc_info=True, extra=log_extra)
        return jsonify({'error': f"Internal Server Error: {str(e)}", 'status': 'error'}), 500

def handle_tool_calls(run, thread_id, request_id):
    log_extra = {'request_id': request_id or getattr(g, 'request_id', 'N/A')}
    tool_outputs = []
    if run.status == 'requires_action' and run.required_action.type == 'submit_tool_outputs':
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_call_id = tool_call.id
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}", extra=log_extra)

            result = {}
            try:
                if tool_name == "retrieve_all_activities":
                    result = retrieve_all_activities(
                        limit=tool_args.get("limit", 3),
                        portfolio_name=tool_args.get("portfolio_name"),
                        request_id=request_id # Pass request_id
                    )
                elif tool_name == "retrieve_assets_by_type":
                    result = retrieve_assets_by_type(
                        short_description=tool_args["short_description"],
                        request_id=request_id # Pass request_id
                    )
                else:
                    logger.warning(f"Unknown tool call: {tool_name}", extra=log_extra)
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": json.dumps(result) # Ensure output is JSON string
                })
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                logger.error(error_msg, exc_info=True, extra=log_extra)
                tool_outputs.append({
                    "tool_call_id": tool_call_id,
                    "output": json.dumps({"error": error_msg})
                })

        # Submit tool outputs back to OpenAI
        if tool_outputs:
            try:
                logger.info("Submitting tool outputs...", extra=log_extra)
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                logger.info("Tool outputs submitted successfully.", extra=log_extra)
                return True # Indicates outputs were submitted
            except Exception as e:
                logger.error(f"Error submitting tool outputs: {e}", exc_info=True, extra=log_extra)
                # If submission fails, we might want the run to eventually fail.
                # For now, log the error and let polling continue. It might timeout/fail later.
                return False # Indicates submission failed
    return False # No tool calls required or handled

if __name__ == '__main__':
    app.run(debug=True)
    
     