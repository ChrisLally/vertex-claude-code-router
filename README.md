# Claude Code Router with Vertex AI

This is an expanded version of the [original Claude Code Router](README_ORIGINAL.md) from [musistudio/claude-code-reverse](https://github.com/musistudio/claude-code-reverse) that adds support for Google Cloud Vertex AI. Our implementation provides a Python-based server that routes Claude Code requests to Vertex AI's Claude models, supporting both Claude 3 Sonnet and Haiku variants.

## Key Features

- Python Flask server implementation
- Google Cloud Vertex AI integration
- Support for Claude 3 models (Sonnet 3.7 and Haiku 3.5)
- Detailed request logging (which you can disable)
- Streaming and non-streaming responses
- Environment-based configuration

## Prerequisites

- Python 3.8+
- Google Cloud Platform account with Vertex AI API enabled
- Service account with appropriate permissions
- Claude Code CLI tool

## Getting Started

1. Clone this repository:
```shell
git clone https://github.com/chrislally/vertex-claude-code-router.git
cd claude-code-router
```

2. Install Python dependencies:
```shell
pip install -r requirements.txt
```

3. Set up your service account:
- Create a service account in Google Cloud Console
- Download the service account key file
- Place it in the project root as `service-account.json`

4. Configure your environment:
- Copy `.env.example` to `.env`
- Update the following key settings:
```
PROJECT_ID=your-gcp-project-id
LOCATION=your-preferred-region
CLAUDE_MODEL=claude-3-7-sonnet-20250219
CLAUDE_HAIKU_MODEL=claude-3-5-haiku-20241022
```

5. Start the server:
```shell
python app.py
```

6. Configure Claude Code CLI:
```shell
export DISABLE_PROMPT_CACHING=1
export ANTHROPIC_AUTH_TOKEN="test"
export ANTHROPIC_BASE_URL="http://127.0.0.1:3456"
export API_TIMEOUT_MS=600000
claude
```

## Environment Configuration

The server can be configured using environment variables. See `.env.example` for a complete list of supported configurations. Key variables include:

- `PROJECT_ID`: Your Google Cloud project ID
- `LOCATION`: Vertex AI API region
- `CLAUDE_MODEL`: Default Claude model (Sonnet)
- `CLAUDE_HAIKU_MODEL`: Default Haiku model
- `PORT`: Server port (default: 3456)

## Logging

The server maintains detailed logs in the `logs` directory. Each request is tracked with a unique ID and includes:
- Request details (endpoint, method, URL)
- Headers and request data
- Model selection and routing decisions
- Response information

## Original Project

This implementation builds upon the original Claude Code Router project, which supports multiple model routing strategies. For information about the original implementation using OpenAI and other models, see the [original README](README_ORIGINAL.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same terms as the original Claude Code Router project.