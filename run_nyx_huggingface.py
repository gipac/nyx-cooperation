#!/usr/bin/env python3
"""
NYX Experiments with HuggingFace Backend
Direct transformer inference - NO Ollama caching issues

Usage:
    python run_nyx_huggingface.py --model microsoft/phi-2 --test-bias
    python run_nyx_huggingface.py --model-path D:/NYX_PROJECT/models/mistral-7b --test-bias
"""

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import logging
from experiments_llm.backends.huggingface_backend import HuggingFaceBackend
from experiments_llm.llm_agents.llm_nyx_agent import LLMNYXAgent
from nyx.agents import CooperationDecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_huggingface_setup(model_name, device='auto', load_in_8bit=False):
    """Test if HuggingFace backend works"""
    print("\n" + "="*60)
    print("Testing HuggingFace Setup")
    print("="*60)

    try:
        backend = HuggingFaceBackend(
            model_name=model_name,
            device=device,
            temperature=1.5,
            load_in_8bit=load_in_8bit
        )

        if not backend.is_available():
            print("❌ HuggingFace not available!")
            print("\nInstall with:")
            print("  pip install transformers torch accelerate")
            return False

        print("✅ HuggingFace available!")
        print(f"📦 Loading model: {model_name}")
        print("   (This may take 1-5 minutes on first load...)")

        # Test generation
        response = backend.generate(
            "Reply with only one word: COOPERATE or DEFECT",
            max_tokens=10
        )

        print(f"\n🧪 Test generation:")
        print(f"   Response: {response.text}")
        print(f"   Time: {response.response_time:.2f}s")
        print(f"   Device: {backend.device}")

        print("\n✅ HuggingFace backend ready!")
        return True

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def bias_test_huggingface(model_name, device='auto', load_in_8bit=False, trials=20, temperature=1.5):
    """Run bias test with HuggingFace backend"""
    print("\n" + "="*60)
    print(f"Bias Test: {model_name} (temp={temperature})")
    print("="*60)

    backend = HuggingFaceBackend(
        model_name=model_name,
        device=device,
        temperature=temperature,
        load_in_8bit=load_in_8bit
    )

    results = {}

    for consciousness_bits in [0, 1, 2]:
        print(f"\nTesting {consciousness_bits}-bit consciousness...")
        print(f"  Running {trials} trials (will take 30-120 seconds)...")

        agent = LLMNYXAgent(
            agent_id=f"hf_agent_{consciousness_bits}",
            llm_backend=backend,
            consciousness_bits=consciousness_bits,
            temperature=temperature
        )

        cooperations = 0

        import time
        start = time.time()

        for i in range(trials):
            if i % 5 == 0 and i > 0:
                elapsed = time.time() - start
                per_trial = elapsed / i
                remaining = per_trial * (trials - i)
                print(f"    Progress: {i}/{trials} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

            # Add fake history for consciousness > 0
            if consciousness_bits > 0 and i > 0:
                agent.update_consciousness_state(float(i % 3), 1.0)

            decision = agent.make_cooperation_decision()

            if decision == CooperationDecision.COOPERATE:
                cooperations += 1

        elapsed = time.time() - start
        rate = cooperations / trials
        results[consciousness_bits] = rate

        print(f"  ✓ Cooperation rate: {rate:.1%} ({cooperations}/{trials})")
        print(f"    Total time: {elapsed:.1f}s ({elapsed/trials:.2f}s per trial)")

    print("\n" + "="*60)
    print("Bias Test Complete")
    print("="*60)
    for bits, rate in results.items():
        print(f"  {bits}-bit: {rate:.1%}")

    # Analyze results
    print("\n📊 Analysis:")
    if results[0] < 0.3:
        print("  ✓ 0-bit baseline is LOW (good!)")
    else:
        print(f"  ⚠ 0-bit baseline is HIGH ({results[0]:.1%})")

    jump = results[1] - results[0]
    if jump > 0.4:
        print(f"  ✓ Single Bit Theory VALIDATED! Jump: {jump:.1%}")
    else:
        print(f"  ⚠ Single Bit jump is small: {jump:.1%}")

    if results[2] > results[1]:
        print(f"  ✓ 2-bit > 1-bit (as expected)")
    else:
        print(f"  ⚠ 2-bit NOT > 1-bit")

    return results


def main():
    parser = argparse.ArgumentParser(description="NYX with HuggingFace Backend")
    parser.add_argument("--model", default="microsoft/phi-2",
                       help="HuggingFace model name")
    parser.add_argument("--model-path",
                       help="Local model path (e.g., D:/NYX_PROJECT/models/mistral-7b)")
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"],
                       help="Device to use (default: cpu for compatibility)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Use 8-bit quantization (saves memory)")
    parser.add_argument("--test-setup", action="store_true",
                       help="Test HuggingFace setup only")
    parser.add_argument("--test-bias", action="store_true",
                       help="Run bias test")
    parser.add_argument("--trials", type=int, default=20,
                       help="Number of trials for bias test")
    parser.add_argument("--temperature", type=float, default=1.5,
                       help="LLM temperature")

    args = parser.parse_args()

    # Use model-path if provided, otherwise use model name
    model = args.model_path if args.model_path else args.model

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     NYX with HuggingFace Backend                                 ║
║     Direct Transformer Inference (No Ollama Cache)               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print(f"Model: {model}")
    print(f"Device: {args.device}")
    print(f"8-bit: {args.load_in_8bit}")

    # Test setup
    if args.test_setup:
        success = test_huggingface_setup(model, args.device, args.load_in_8bit)
        return 0 if success else 1

    # Bias test
    if args.test_bias:
        bias_test_huggingface(model, args.device, args.load_in_8bit, args.trials, args.temperature)
        return 0

    # Default: run setup test
    print("\nNo action specified. Running setup test...")
    success = test_huggingface_setup(model, args.device, args.load_in_8bit)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
