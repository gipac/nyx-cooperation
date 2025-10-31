#!/usr/bin/env python3
"""
NYX Reconstruction - Windows Standalone Script
Run this directly on Windows PowerShell where Ollama is running

Usage:
    python run_nyx_windows.py --fast
    python run_nyx_windows.py --test-bias
    python run_nyx_windows.py --phase 1
"""

import sys
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import NYX modules
from experiments_llm.backends.ollama_backend import OllamaBackend
from experiments_llm.llm_agents.llm_nyx_agent import LLMNYXAgent
from experiments_llm.llm_agents.interaction_runner import (
    run_llm_interaction_cycle,
    create_llm_agent_population
)
from nyx import predict_cooperation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nyx_windows_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_ollama_connection(model_name='mistral-7b-instruct'):
    """Test Ollama connection"""
    print("\n" + "="*60)
    print("Testing Ollama Connection")
    print("="*60)

    backend = OllamaBackend(model_name, api_url='http://localhost:11434')

    if not backend.is_available():
        print("❌ Ollama not available!")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama app is running")
        print("  2. Check if model is installed: ollama list")
        print("  3. Pull model: ollama pull mistral-7b-instruct")
        return False

    print("✅ Ollama connected!")

    models = backend.list_models()
    print(f"\n📦 Available models: {len(models)}")
    for model in models[:5]:
        print(f"  - {model}")

    # Quick test
    print("\n🧪 Testing model response...")
    try:
        response = backend.generate(
            "Reply with only one word: COOPERATE or DEFECT",
            max_tokens=10
        )
        print(f"   Response: {response.text}")
        print(f"   Time: {response.response_time:.2f}s")
        print("\n✅ All systems operational!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def quick_bias_test(model_name='mistral-7b-instruct', trials=10):
    """Quick bias test"""
    print("\n" + "="*60)
    print(f"Quick Bias Test: {model_name}")
    print("="*60)

    backend = OllamaBackend(model_name, api_url='http://localhost:11434', temperature=0.7)

    results = {}
    for consciousness_bits in [0, 1, 2]:
        print(f"\nTesting {consciousness_bits}-bit consciousness...")

        agent = LLMNYXAgent(
            agent_id=f"test_{consciousness_bits}",
            llm_backend=backend,
            consciousness_bits=consciousness_bits,
            temperature=0.7
        )

        cooperations = 0
        for i in range(trials):
            if consciousness_bits > 0 and i > 0:
                agent.update_consciousness_state(float(i % 3), 1.0)

            from nyx.agents import CooperationDecision
            decision = agent.make_cooperation_decision()
            if decision == CooperationDecision.COOPERATE:
                cooperations += 1

        rate = cooperations / trials
        results[consciousness_bits] = rate
        print(f"  Cooperation rate: {rate:.1%} ({cooperations}/{trials})")

    print("\n" + "="*60)
    print("Bias Test Complete")
    print("="*60)
    for bits, rate in results.items():
        print(f"  {bits}-bit: {rate:.1%}")

    return results


def run_single_phase(phase_num, model_name='mistral-7b-instruct', fast_mode=True):
    """Run a single phase"""
    from experiments_llm.run_historical_reconstruction import NYXHistoricalReconstruction

    print("\n" + "="*60)
    print(f"Running Phase {phase_num}")
    print("="*60)

    reconstruction = NYXHistoricalReconstruction(
        model_name=model_name,
        ollama_url='http://localhost:11434',
        fast_mode=fast_mode,
        save_data=True
    )

    phase_methods = {
        0: reconstruction.phase_0_baseline,
        1: reconstruction.phase_1_single_bit_theory,
        2: reconstruction.phase_2_multibit_consciousness,
        3: reconstruction.phase_3_memory_effect,
        4: reconstruction.phase_4_network_effect,
        5: reconstruction.phase_5_formula_synthesis,
        6: reconstruction.phase_6_80_20_validation
    }

    result = phase_methods[phase_num]()

    # Save result
    output_file = f"results_llm/phase_{phase_num}_result.json"
    os.makedirs('results_llm', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✅ Phase {phase_num} complete!")
    print(f"   Results saved to: {output_file}")

    return result


def run_full_reconstruction(model_name='mistral-7b-instruct', fast_mode=True):
    """Run all 7 phases"""
    from experiments_llm.run_historical_reconstruction import NYXHistoricalReconstruction

    print("\n" + "="*70)
    print("NYX HISTORICAL RECONSTRUCTION - FULL RUN")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Mode: {'FAST' if fast_mode else 'FULL'}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    reconstruction = NYXHistoricalReconstruction(
        model_name=model_name,
        ollama_url='http://localhost:11434',
        fast_mode=fast_mode,
        save_data=True
    )

    try:
        results = reconstruction.run_all_phases()

        print("\n" + "="*70)
        print("🎉 RECONSTRUCTION COMPLETE!")
        print("="*70)

        summary = results['summary']
        print(f"\nDuration: {summary['total_duration_hours']:.2f} hours")
        print(f"\n✅ Key Discoveries Validated:")
        for discovery, validated in summary['key_discoveries'].items():
            symbol = "✓" if validated else "✗"
            print(f"  {symbol} {discovery}")

        return results

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="NYX Reconstruction - Windows Runner")
    parser.add_argument("--model", default="mistral-7b-instruct",
                       help="Ollama model name")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode (fewer cycles)")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test Ollama connection only")
    parser.add_argument("--test-bias", action="store_true",
                       help="Quick bias test (10 trials)")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                       help="Run specific phase only")
    parser.add_argument("--trials", type=int, default=10,
                       help="Trials for bias test")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     NYX: Mathematical Laws of AI Cooperation                     ║
║     LLM Historical Reconstruction                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Test connection
    if args.test_connection:
        return 0 if test_ollama_connection(args.model) else 1

    # Quick bias test
    if args.test_bias:
        quick_bias_test(args.model, args.trials)
        return 0

    # Test connection first
    if not test_ollama_connection(args.model):
        return 1

    # Run specific phase
    if args.phase is not None:
        run_single_phase(args.phase, args.model, args.fast)
        return 0

    # Run full reconstruction
    print("\n⚠️  Starting full reconstruction!")
    print(f"   Estimated time: {'2-3 hours' if args.fast else '6-8 hours'}")
    print("\n   Press Ctrl+C to cancel (you have 5 seconds)...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 1

    results = run_full_reconstruction(args.model, args.fast)

    return 0 if results else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
