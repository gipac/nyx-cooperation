"""
HuggingFace Backend for NYX Experiments
Direct transformer inference without caching
"""

import time
import logging
from typing import Dict, List, Optional, Any
from .base_backend import BaseLLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class HuggingFaceBackend(BaseLLMBackend):
    """
    HuggingFace Transformers Backend

    Loads models directly from HuggingFace or local directory.
    Provides full control over inference - NO hidden caching.

    Args:
        model_name: HuggingFace model name or local path
        device: 'cuda' or 'cpu' (default: auto)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        load_in_8bit: Use 8-bit quantization (saves memory)
    """

    def __init__(self,
                 model_name: str,
                 device: str = 'auto',
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 timeout: int = 30,
                 load_in_8bit: bool = False):

        super().__init__(model_name, temperature, max_tokens, timeout)
        self.device = device
        self.load_in_8bit = load_in_8bit

        # Will be initialized on first use (lazy loading)
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        logger.info(f"HuggingFace backend initialized: {model_name}")

    def _initialize_model(self):
        """Lazy load model on first use"""
        if self.model is not None:
            return

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline
            )
            import torch

            logger.info(f"Loading model {self.model_name}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optional quantization
            load_kwargs = {'trust_remote_code': True}

            if self.load_in_8bit:
                load_kwargs['load_in_8bit'] = True
                load_kwargs['device_map'] = 'auto'
            else:
                # Determine device
                if self.device == 'auto':
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                load_kwargs['device_map'] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )

            logger.info(f"✅ Model loaded on {self.device}")

        except ImportError:
            logger.error("transformers library not installed!")
            logger.error("Install with: pip install transformers torch accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def is_available(self) -> bool:
        """Check if backend is available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

    def list_models(self) -> List[str]:
        """List available models (not applicable for HF)"""
        return [self.model_name]

    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> LLMResponse:
        """
        Generate response using HuggingFace model

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            **kwargs: Generation parameters

        Returns:
            LLMResponse object
        """
        # Initialize model on first call
        self._initialize_model()

        start_time = time.time()

        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Generation parameters
        gen_kwargs = {
            'max_new_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'do_sample': True,  # Enable sampling for non-deterministic output
            'top_p': 0.9,
            'top_k': 50,
            'pad_token_id': self.tokenizer.eos_token_id,
            'return_full_text': False,  # Only return generated text
        }

        try:
            # Generate with pipeline
            outputs = self.pipeline(
                full_prompt,
                **gen_kwargs
            )

            # Extract generated text
            response_text = outputs[0]['generated_text'].strip()

            # Count tokens (approximate)
            tokens_used = len(self.tokenizer.encode(response_text))

            response_time = time.time() - start_time

            # Create response object
            llm_response = LLMResponse(
                text=response_text,
                model=self.model_name,
                tokens_used=tokens_used,
                response_time=response_time,
                temperature=self.temperature,
                metadata={
                    'device': str(self.device),
                    'load_in_8bit': self.load_in_8bit
                }
            )

            # Update statistics
            self._update_statistics(llm_response)

            logger.debug(f"Generated in {response_time:.2f}s, {tokens_used} tokens")

            return llm_response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"HuggingFace generation failed: {e}")

    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

            logger.info("Model unloaded from memory")


if __name__ == "__main__":
    # Test HuggingFace backend
    logging.basicConfig(level=logging.INFO)

    print("Testing HuggingFace Backend")
    print("="*60)

    # Test with small model (if available)
    model_name = "gpt2"  # Small model for testing

    backend = HuggingFaceBackend(model_name, device='cpu')

    if not backend.is_available():
        print("❌ HuggingFace not available!")
        print("   Install: pip install transformers torch")
        exit(1)

    print("✅ HuggingFace available")

    # Test generation
    print("\n🧪 Testing generation...")
    prompt = "Complete this: COOPERATE or"

    response = backend.generate(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.text}")
    print(f"Time: {response.response_time:.2f}s")
    print(f"Tokens: {response.tokens_used}")

    print("\n✅ HuggingFace backend test complete!")
