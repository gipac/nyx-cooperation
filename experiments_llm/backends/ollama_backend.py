"""
Ollama Backend for NYX Experiments
Connects to Ollama API (local or remote)
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from .base_backend import BaseLLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLMBackend):
    """
    Ollama LLM Backend

    Connects to Ollama API for model inference.
    Supports both local (localhost:11434) and remote Ollama instances.

    Args:
        model_name: Name of Ollama model (e.g., 'mistral-7b-instruct')
        api_url: Ollama API URL (default: http://localhost:11434)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
    """

    def __init__(self,
                 model_name: str,
                 api_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 timeout: int = 30):

        super().__init__(model_name, temperature, max_tokens, timeout)
        self.api_url = api_url.rstrip('/')
        self.generate_endpoint = f"{self.api_url}/api/generate"
        self.tags_endpoint = f"{self.api_url}/api/tags"

        logger.info(f"Ollama backend initialized: {model_name} @ {api_url}")

    def is_available(self) -> bool:
        """Check if Ollama API is reachable"""
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            response.raise_for_status()

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]

            logger.info(f"Found {len(models)} Ollama models")
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate response using Ollama

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            **kwargs: Additional Ollama parameters

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        # Build full prompt with system instruction
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Prepare request payload
        # Add random seed to prevent any caching
        import random
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
                "seed": random.randint(1, 1000000),  # Force unique inference each time
                "num_ctx": 512,  # Small context to prevent cache reuse
            }
        }

        try:
            # Send request to Ollama
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            data = response.json()
            response_text = data.get('response', '').strip()

            # Extract tokens (approximate if not provided)
            tokens_used = data.get('eval_count', len(response_text.split()))

            response_time = time.time() - start_time

            # Create response object
            llm_response = LLMResponse(
                text=response_text,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                temperature=self.temperature,
                metadata={
                    'total_duration': data.get('total_duration', 0),
                    'load_duration': data.get('load_duration', 0),
                    'prompt_eval_count': data.get('prompt_eval_count', 0),
                    'eval_count': data.get('eval_count', 0),
                }
            )

            # Update statistics
            self._update_statistics(llm_response)

            logger.debug(f"Generated response in {response_time:.2f}s, {tokens_used} tokens")

            return llm_response

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timeout after {self.timeout}s")
            raise TimeoutError(f"Ollama generation timeout: {self.timeout}s")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_batch(self,
                      prompts: List[str],
                      system_prompt: Optional[str] = None,
                      **kwargs) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts (sequential)

        Note: Ollama doesn't support native batching, so this is sequential
        """
        responses = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            # Ollama doesn't have a dedicated model info endpoint
            # We can infer from the model name
            return {
                'name': self.model_name,
                'backend': 'ollama',
                'api_url': self.api_url,
                'available': self.is_available()
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'name': self.model_name, 'error': str(e)}


if __name__ == "__main__":
    # Test Ollama backend
    logging.basicConfig(level=logging.INFO)

    print("Testing Ollama Backend")
    print("=" * 50)

    backend = OllamaBackend("mistral-7b-instruct")

    # Check availability
    print(f"Ollama available: {backend.is_available()}")

    if backend.is_available():
        # List models
        models = backend.list_models()
        print(f"Available models: {models}")

        # Test generation
        prompt = "You have 3 apples. If you give 1 away, how many do you have left? Answer with just the number."
        response = backend.generate(prompt)

        print(f"\nTest prompt: {prompt}")
        print(f"Response: {response.text}")
        print(f"Time: {response.response_time:.2f}s")
        print(f"Tokens: {response.tokens_used}")

        # Statistics
        stats = backend.get_statistics()
        print(f"\nStatistics: {stats}")
    else:
        print("\n⚠️  Ollama not available. Make sure Ollama is running:")
        print("   Windows: Start Ollama app")
        print("   Linux: systemctl start ollama")
        print("   Docker: docker run -d -p 11434:11434 ollama/ollama")
