from flask import Flask, request, jsonify, Response, stream_with_context
from anthropic import AnthropicVertex
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, Iterator
from dotenv import load_dotenv
from functools import wraps
import uuid
from contextvars import ContextVar
import time

# Create a context variable for request_id
request_id_var = ContextVar('request_id', default='main')

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

# Set up logging
log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# File handler with rotation
file_handler = RotatingFileHandler(
    'logs/flask_app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(RequestIdFilter())

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)
console_handler.addFilter(RequestIdFilter())

# Setup root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Generate and set request ID for this context
        req_id = str(uuid.uuid4())[:8]
        token = request_id_var.set(req_id)
        
        try:
            # Log request details
            logger.info("\n%s\nRequest Details:", "="*50)
            logger.info(f"Endpoint: {request.endpoint}")
            logger.info(f"Method: {request.method}")
            logger.info(f"URL: {request.url}")
            logger.info(f"Headers:\n{dict(request.headers)}")
            
            # Log request data based on content type
            if request.is_json:
                logger.info(f"JSON Data:\n{json.dumps(request.json, indent=2)}")
            elif request.form:
                logger.info(f"Form Data: {dict(request.form)}")
            elif request.data:
                logger.info(f"Raw Data: {request.data.decode()}")
                
            logger.info(f"Query Parameters: {dict(request.args)}")
            logger.info(f"{'='*50}\n")
            
            return f(*args, **kwargs)
        finally:
            request_id_var.reset(token)
            
    return decorated_function

load_dotenv()

app = Flask(__name__)

# Load configuration from environment
PORT = int(os.getenv('PORT', 3456))
PROJECT_ID = os.getenv('PROJECT_ID', 'meta-agents')
LOCATION = os.getenv('LOCATION', 'us-east5')
MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-7-sonnet@20250219')
HAIKU_MODEL = os.getenv('CLAUDE_HAIKU_MODEL', 'claude-3-5-haiku@20241022')

# Map of supported models
SUPPORTED_MODELS = {
    'claude-3-sonnet-20240229': MODEL,
    'claude-3-7-sonnet-20250219': MODEL,
    'claude-3-haiku-20240307': HAIKU_MODEL,
    'claude-3-5-haiku-20241022': HAIKU_MODEL
}

class VertexClaudeClient:
    def __init__(
        self,
        project_id: str = "meta-agents",
        location: str = "us-east5",
        model: str = None,
        service_account_file: str = "./service-account.json"
    ):
        self.project_id = project_id
        self.location = location
        self.model = model or MODEL
        self.service_account_file = service_account_file
        self.client = self._initialize_client()
        self.tools = []

    def _initialize_client(self) -> AnthropicVertex:
        logger.info(f"Initializing Vertex AI client with project_id={self.project_id}, location={self.location}")
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        return AnthropicVertex(
            region=self.location,
            project_id=self.project_id,
            access_token=credentials.token
        )

    def refresh_token(self) -> None:
        logger.info("Refreshing Vertex AI token")
        self.client = self._initialize_client()

    def add_tools(self, tools: list) -> None:
        logger.info(f"Adding tools: {tools}")
        self.tools = tools

    def create_message(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"Creating message with prompt: {prompt[:100]}...")
        kwargs = {
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            **{k: v for k, v in kwargs.items() if k not in ['messages', 'system']}
        }
        if system_prompt:
            kwargs["system"] = system_prompt
            logger.info(f"Using system prompt: {system_prompt[:100]}...")
        if self.tools:
            kwargs["tools"] = self.tools
            logger.info(f"Using tools: {self.tools}")
        
        if stream:
            logger.info("Using streaming mode")
            return self.client.messages.stream(**kwargs)
        
        logger.info("Using non-streaming mode")
        return self.client.messages.create(**kwargs)

# Initialize the Claude client with default values
claude_client = VertexClaudeClient()

@app.route('/v1/messages', methods=['POST'])
@log_request
def create_message():
    try:
        data = request.json
        logger.info(f"Processing request data: {json.dumps(data, indent=2)}")
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Handle model selection
        requested_model = data.get('model')
        if requested_model in SUPPORTED_MODELS:
            data['model'] = SUPPORTED_MODELS[requested_model]
            logger.info(f"Using mapped model: {data['model']} for requested model: {requested_model}")
        else:
            logger.warning(f"Requested model '{requested_model}' not in supported models. Using default model '{MODEL}'.")
            data['model'] = MODEL
            
        stream = data.get('stream', False)
        
        # Extract message content from the messages array
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        # Normalize message content format
        first_message = messages[0]
        if isinstance(first_message.get('content'), str):
            content = first_message['content']
        elif isinstance(first_message.get('content'), list):
            # Extract text from content array
            content = ' '.join(item['text'] for item in first_message['content'] if item.get('type') == 'text')
        else:
            return jsonify({'error': 'Invalid message content format'}), 400
        
        # Remove fields that are handled separately
        kwargs = {k: v for k, v in data.items() if k not in ['messages', 'system', 'stream']}
        
        if stream:
            logger.info("Processing streaming request")
            def generate():
                message_id = f"msg_{int(time.time() * 1000)}"
                content_block_index = 0
                
                # Send message_start event
                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": MODEL,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }
                }
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                
                # Send content_block_start event
                block_start = {
                    "type": "content_block_start",
                    "index": content_block_index,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    }
                }
                yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                
                # Stream the content
                stream_manager = claude_client.create_message(
                    prompt=content,
                    system_prompt=data.get('system'),
                    stream=True,
                    **kwargs
                )
                
                accumulated_text = ""
                with stream_manager as stream:
                    for chunk in stream.text_stream:
                        # Send content delta
                        delta = {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {
                                "type": "text_delta",
                                "text": chunk
                            }
                        }
                        accumulated_text += chunk
                        logger.debug(f"Streaming chunk: {delta}")
                        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                
                # Send content_block_stop event
                block_stop = {
                    "type": "content_block_stop",
                    "index": content_block_index
                }
                yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
                
                # Send message_delta event
                message_delta = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "content": [{
                            "type": "text",
                            "text": accumulated_text
                        }]
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 150}
                }
                yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
                
                # Send message_stop event
                message_stop = {
                    "type": "message_stop"
                }
                yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                }
            )
        
        logger.info("Processing non-streaming request")
        message = claude_client.create_message(
            prompt=content,
            system_prompt=data.get('system'),
            **kwargs
        )
        
        # Convert Message object to JSON-serializable format
        response = {
            'id': message.id,
            'content': [{'type': block.type, 'text': block.text} for block in message.content],
            'model': message.model,
            'role': message.role,
            'stop_reason': message.stop_reason,
            'stop_sequence': message.stop_sequence,
            'type': message.type,
            'usage': {
                'input_tokens': message.usage.input_tokens,
                'output_tokens': message.usage.output_tokens,
                'cache_creation_input_tokens': message.usage.cache_creation_input_tokens,
                'cache_read_input_tokens': message.usage.cache_read_input_tokens
            }
        }
        
        logger.info(f"Sending response: {json.dumps(response, indent=2)}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@log_request
def health_check():
    logger.info("Health check request received")
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    logger.info(f"Starting server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT) 