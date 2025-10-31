#!/usr/bin/env python3
"""
Test Model Bias - Preliminary Testing
Measures cooperation bias in different LLM models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any

from experiments_llm.backends.ollama_backend import OllamaBackend
from experiments_llm.llm_agents.llm_nyx_agent import LLMNYXAgent
from nyx.agents import CooperationDecision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results_llm/model_bias_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_model_cooperation_bias(model_name: str,
                                 ollama_url: str = "http://localhost:11434",
                                 trials: int = 20,
                                 temperatures: List[float] = [0.0, 0.5, 0.7, 1.0]) -> Dict[str, Any]:
    """
    Test cooperation bias for a specific model

    Args:
        model_name: Name of Ollama model
        ollama_url: Ollama API URL
        trials: Number of trials per temperature
        temperatures: Temperature values to test

    Returns:
        Dictionary with bias results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Model: {model_name}")
    logger.info(f"{'='*60}")

    results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'temperatures': {}
    }

    for temp in temperatures:
        logger.info(f"\nTesting temperature: {temp}")

        # Create backend
        backend = OllamaBackend(
            model_name=model_name,
            api_url=ollama_url,
            temperature=temp,
            max_tokens=20
        )

        # Test different consciousness levels
        for consciousness_bits in [0, 1, 2]:
            logger.info(f"  Consciousness bits: {consciousness_bits}")

            agent = LLMNYXAgent(
                agent_id=f"test_agent_{temp}_{consciousness_bits}",
                llm_backend=backend,
                consciousness_bits=consciousness_bits,
                temperature=temp
            )

            cooperations = 0
            defections = 0
            failures = 0

            for trial in range(trials):
                try:
                    # Simulate some history for consciousness > 0
                    if consciousness_bits > 0 and trial > 0:
                        # Add fake history
                        agent.update_consciousness_state(
                            benefits_received=float(trial % 3),
                            costs_incurred=1.0
                        )

                    decision = agent.make_cooperation_decision()

                    if decision == CooperationDecision.COOPERATE:
                        cooperations += 1
                    elif decision == CooperationDecision.DEFECT:
                        defections += 1

                except Exception as e:
                    failures += 1
                    logger.warning(f"    Trial {trial} failed: {e}")

            # Calculate bias
            total_valid = cooperations + defections
            cooperation_rate = cooperations / total_valid if total_valid > 0 else 0

            temp_key = f"temp_{temp}"
            if temp_key not in results['temperatures']:
                results['temperatures'][temp_key] = {}

            results['temperatures'][temp_key][f"{consciousness_bits}_bit"] = {
                'cooperations': cooperations,
                'defections': defections,
                'failures': failures,
                'cooperation_rate': cooperation_rate,
                'trials': trials
            }

            logger.info(f"    Cooperation rate: {cooperation_rate:.1%} "
                       f"({cooperations}/{total_valid} cooperate, {failures} failures)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test LLM cooperation bias")
    parser.add_argument("--models", nargs="+",
                       default=["mistral-7b-instruct", "llama-3.2-3b-instruct", "phi-3.5-mini"],
                       help="Models to test")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama API URL")
    parser.add_argument("--trials", type=int, default=20,
                       help="Trials per temperature")
    parser.add_argument("--output", default="results_llm/model_bias_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("NYX MODEL BIAS TESTING")
    logger.info("="*60)
    logger.info(f"Models: {args.models}")
    logger.info(f"Trials per temperature: {args.trials}")
    logger.info(f"Ollama URL: {args.ollama_url}")

    # Test Ollama availability
    test_backend = OllamaBackend("test", api_url=args.ollama_url)
    if not test_backend.is_available():
        logger.error("❌ Ollama not available!")
        logger.error(f"   Make sure Ollama is running at {args.ollama_url}")
        logger.error("   Windows: Start Ollama app")
        logger.error("   Linux: systemctl start ollama")
        return 1

    available_models = test_backend.list_models()
    logger.info(f"Available models: {available_models}")

    # Test each model
    all_results = []

    for model in args.models:
        if model not in available_models:
            logger.warning(f"⚠️  Model '{model}' not found in Ollama. Skipping.")
            logger.warning(f"   Pull with: ollama pull {model}")
            continue

        try:
            results = test_model_cooperation_bias(
                model_name=model,
                ollama_url=args.ollama_url,
                trials=args.trials
            )
            all_results.append(results)

        except Exception as e:
            logger.error(f"Failed to test {model}: {e}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY - Cooperation Bias by Model")
    logger.info("="*60)

    for result in all_results:
        model = result['model']
        logger.info(f"\n{model}:")

        for temp_key, temp_data in result['temperatures'].items():
            temp = temp_key.replace('temp_', '')
            logger.info(f"  Temperature {temp}:")

            for bits_key, bits_data in temp_data.items():
                bits = bits_key.replace('_bit', '')
                coop_rate = bits_data['cooperation_rate']
                logger.info(f"    {bits}-bit consciousness: {coop_rate:.1%} cooperation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
