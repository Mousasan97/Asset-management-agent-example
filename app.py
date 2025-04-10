from flask import Flask, render_template, request, jsonify, g
from openai import AzureOpenAI
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
import threading
import queue
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Cache configuration
CACHE_TIMEOUT = 300  # 5 minutes cache timeout
API_TIMEOUT = 5  # 5 seconds API timeout
POLLING_INTERVAL = 0.5  # 500ms polling interval
MAX_POLLING_TIME = 30  # Maximum time to poll for a response (seconds)

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
        try:
            record.request_id = getattr(g, 'request_id', 'N/A')
        except Exception:
            # If we're outside of application context, just use N/A
            record.request_id = 'N/A'
        return True

logger.addFilter(RequestIdFilter())

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Azure OpenAI client with API key authentication
try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-05-01-preview"
    )
    logger.info("Successfully initialized Azure OpenAI client")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", exc_info=True)
    raise

# Create a queue for storing run results
run_results = {}

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
        # Check if we have a valid Azure OpenAI endpoint
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise ValueError("Invalid or missing Azure OpenAI endpoint. Please set a valid endpoint in your .env file.")

        # Try to get existing assistant by ID
        assistant_id = os.getenv('AZURE_ASSISTANT_ID')
        if assistant_id:
            try:
                logger.info(f"Retrieving existing assistant with ID: {assistant_id}")
                with app.app_context():
                    assistant = client.beta.assistants.retrieve(assistant_id)
                    logger.info("Successfully retrieved existing assistant")
                    return assistant
            except Exception as e:
                logger.warning(f"Failed to retrieve existing assistant: {str(e)}. Creating new one...")

        # If no assistant_id or retrieval failed, create new assistant
        logger.info("Creating new assistant...")
        with app.app_context():
            assistant = client.beta.assistants.create(
                model="gpt-4o",  # replace with model deployment name
                name="Asset management agent",
                instructions="""You are a data retrieval agent, having access to an Antares platform, your role is to retrieve the data based on the user query. The data you 
                have access to is:
                -Retrieve assets based on the asset name.
                -Retrieve activities
                """,
                tools=[
                    {"type":"file_search"},
                    {"type":"function","function":{"name":"retrieve_assets_by_type","description":"Retrieves the available assets by the type of the asset, while reading from Antares platform","parameters":{"type":"object","properties":{"short_description":{"type":"string","description":"The type of the asset to retrieve"}},"required":["short_description"]}}},
                    {"type":"function","function":{"name":"retrieve_all_activities","description":"Retrieves the most recent n activities on assets while reading from Antares platform","parameters":{"type":"object","properties":{"limit":{"type":"string","description":"The number of activities to retrieve"}},"required":["limit"]}}}
                ],
                tool_resources={"file_search":{"vector_store_ids":[]}},
                temperature=1,
                top_p=1
            )
            
            # Log the new assistant ID for future use
            logger.info(f"Created new assistant with ID: {assistant.id}")
            logger.info("Please add this ID to your .env file as AZURE_ASSISTANT_ID")
            
            return assistant
    except Exception as e:
        logger.error(f"Failed to create Azure OpenAI Assistant during startup: {str(e)}", exc_info=True)
        raise

# Create the assistant at startup
with app.app_context():
    assistant = get_or_create_assistant()

@app.route('/')
def home():
    return render_template('index.html')

# Function to poll for run completion in a separate thread
def poll_run_completion(thread_id, run_id, request_id):
    log_extra = {'request_id': request_id}
    start_time = time.time()
    
    while time.time() - start_time < MAX_POLLING_TIME:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            
            if run.status == 'completed':
                logger.info(f"Run {run.id} completed. Retrieving messages...", extra=log_extra)
                messages = client.beta.threads.messages.list(thread_id=thread_id, order='desc', limit=1)
                assistant_message = "No response found."
                
                for msg in messages.data:
                    if msg.role == 'assistant':
                        if msg.content and msg.content[0].type == 'text':
                            assistant_message = msg.content[0].text.value
                            break
                
                logger.info(f"Final assistant message retrieved.", extra=log_extra)
                run_results[run_id] = {
                    'status': 'completed',
                    'response': assistant_message,
                    'thread_id': thread_id
                }
                return
                
            elif run.status == 'requires_action':
                logger.info(f"Run {run.id} requires tool action.", extra=log_extra)
                handle_tool_calls(run, thread_id, request_id)
                # Continue polling after handling tool calls
                
            elif run.status in ['queued', 'in_progress']:
                logger.debug(f"Run {run.id} is {run.status}.", extra=log_extra)
                time.sleep(POLLING_INTERVAL)
                
            elif run.status in ['cancelling', 'cancelled', 'failed', 'expired']:
                error_message = f"Run {run.id} ended with status: {run.status}"
                if run.last_error:
                    error_message += f" - Error: {run.last_error.code}: {run.last_error.message}"
                logger.error(error_message, extra=log_extra)
                run_results[run_id] = {
                    'status': 'failed',
                    'error': error_message,
                    'run_status': run.status
                }
                return
                
        except Exception as e:
            logger.error(f"Error polling run status: {str(e)}", exc_info=True, extra=log_extra)
            run_results[run_id] = {
                'status': 'failed',
                'error': f"Error polling run status: {str(e)}"
            }
            return
            
        time.sleep(POLLING_INTERVAL)
    
    # If we've reached the maximum polling time
    logger.warning(f"Run {run_id} timed out after {MAX_POLLING_TIME} seconds", extra=log_extra)
    run_results[run_id] = {
        'status': 'timeout',
        'error': f"Run timed out after {MAX_POLLING_TIME} seconds"
    }

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    request_id = getattr(g, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}
    
    try:
        data = request.get_json()
        message_content = data.get('message', '')
        thread_id = data.get('thread_id')
        
        if not message_content:
            logger.error("No message content provided", extra=log_extra)
            return jsonify({'error': 'No message content provided', 'status': 'error'}), 400

        # 1. Get or Create Thread
        if not thread_id:
            logger.info("Creating new thread...", extra=log_extra)
            thread = client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"New thread created: {thread_id}", extra=log_extra)
        else:
            logger.info(f"Using existing thread: {thread_id}", extra=log_extra)

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
        )
        logger.info(f"Run created: {run.id} with status: {run.status}", extra=log_extra)
        
        # 4. Start polling in a separate thread
        polling_thread = threading.Thread(
            target=poll_run_completion,
            args=(thread_id, run.id, request_id)
        )
        polling_thread.daemon = True
        polling_thread.start()

        # 5. Return Run ID and Thread ID Immediately
        return jsonify({
            'run_id': run.id,
            'thread_id': thread_id,
            'status': 'started'
        }), 202

    except Exception as e:
        error_msg = f"Error starting chat run: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg, exc_info=True, extra=log_extra)
        return jsonify({'error': f"Internal Server Error: {str(e)}", 'status': 'error'}), 500

@app.route('/api/chat/status/<run_id>', methods=['GET'])
def get_chat_status(run_id):
    request_id = getattr(g, 'request_id', 'N/A')
    log_extra = {'request_id': request_id}

    thread_id = request.args.get('thread_id')
    if not thread_id:
        logger.error("Missing thread_id query parameter", extra=log_extra)
        return jsonify({'error': 'Missing thread_id', 'status': 'error'}), 400

    logger.info(f"Checking status for run_id: {run_id}, thread_id: {thread_id}", extra=log_extra)

    # Check if we have a result for this run
    if run_id in run_results:
        result = run_results[run_id]
        # Remove the result from the queue to free up memory
        del run_results[run_id]
        return jsonify(result)
    
    # If no result yet, return processing status
    return jsonify({'status': 'processing'})

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
    
     