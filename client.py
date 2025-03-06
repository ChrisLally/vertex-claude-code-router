
from anthropic import AnthropicVertex
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from typing import List, Optional, Any, Dict, Iterator

# ClaudeClient (unchanged from your original)
class ClaudeClient:
    def __init__(
        self,
        project_id: str = "meta-agents",
        location: str = "us-east5",
        model: str = "claude-3-7-sonnet@20250219",
        service_account_file: str = "./service-account.json",
    ):
        self.project_id = project_id
        self.location = location
        self.model = model
        self.service_account_file = service_account_file
        self.client = self._initialize_client()
        self.tools = []

    def _initialize_client(self) -> AnthropicVertex:
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
        self.client = self._initialize_client()

    def add_tools(self, tools: List[Dict[str, Any]]) -> None:
        self.tools = tools

    def create_message(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ):
        kwargs = {
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.tools:
            kwargs["tools"] = self.tools
        return self.client.messages.create(**kwargs)

    def stream_message(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> Iterator[Any]:
        kwargs = {
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.tools:
            kwargs["tools"] = self.tools
        
        stream_manager = self.client.messages.stream(**kwargs)
        with stream_manager as stream:
            for text in stream.text_stream:
                yield type('MessageChunk', (), {'text': text})()

# Usage example
if __name__ == "__main__":
    client = ClaudeClient()
    response = client.create_message("Hello, how are you?")
    print(response)