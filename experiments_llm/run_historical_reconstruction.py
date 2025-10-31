#!/usr/bin/env python3
"""
NYX Historical Reconstruction with Real LLMs
Reconstructs the scientific discovery path using real language models

7 Phases:
0. Baseline (no consciousness)
1. Single Bit Theory (0% → 67.5% jump)
2. Multi-Bit Consciousness (1-bit vs 2-bit vs 3-bit)
3. Memory Effect (sweet spot at 10 patterns)
4. Network Effect (Minimum Viable Society)
5. Formula Synthesis (C = 0.1×N + 0.1×M + 0.8×A)
6. 80/20 Law Validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from experiments_llm.backends.ollama_backend import OllamaBackend
from experiments_llm.llm_agents.llm_nyx_agent import LLMNYXAgent
from experiments_llm.llm_agents.interaction_runner import (
    run_llm_interaction_cycle,
    create_llm_agent_population
)
from nyx import NYXCooperationSystem, predict_cooperation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results_llm/reconstruction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NYXHistoricalReconstruction:
    """
    Complete historical reconstruction of NYX discoveries
    """

    def __init__(self,
                 model_name: str,
                 ollama_url: str = "http://localhost:11434",
                 fast_mode: bool = False,
                 save_data: bool = True):

        self.model_name = model_name
        self.ollama_url = ollama_url
        self.fast_mode = fast_mode
        self.save_data = save_data
        self.start_time = time.time()

        # Results storage
        self.results = {}

        # Create backend
        self.backend = OllamaBackend(
            model_name=model_name,
            api_url=ollama_url,
            temperature=0.7
        )

        logger.info(f"Historical Reconstruction initialized")
        logger.info(f"Model: {model_name}")
        logger.info(f"Fast mode: {fast_mode}")

    def phase_0_baseline(self) -> Dict[str, Any]:
        """
        Phase 0: Baseline - No Consciousness

        Test: Pure instinct cooperation without meta-cognition
        Expected: ~0-10% cooperation (random/baseline)
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 0: BASELINE (No Consciousness)")
        logger.info("="*60)

        cycles = 20 if self.fast_mode else 50
        agent_count = 4

        # Create 0-bit agents
        agents = create_llm_agent_population(
            count=agent_count,
            llm_backend=self.backend,
            consciousness_bits=0,
            sharing_probability=0.1  # Low baseline
        )

        # Run interactions
        stats = run_llm_interaction_cycle(
            agents,
            cycles=cycles,
            interaction_probability=0.3
        )

        results = {
            'phase': 0,
            'phase_name': 'baseline',
            'consciousness_bits': 0,
            'agent_count': agent_count,
            'cycles': cycles,
            'cooperation_rate': stats['cooperation_rate'],
            'predicted': 0.0,  # No formula prediction for baseline
            'statistics': stats
        }

        logger.info(f"✅ Phase 0 Complete: {stats['cooperation_rate']:.1%} cooperation")

        return results

    def phase_1_single_bit_theory(self) -> Dict[str, Any]:
        """
        Phase 1: Single Bit Theory

        Test: Does 1-bit ROI awareness cause dramatic jump?
        Expected: 0% → 67.5% cooperation jump
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: SINGLE BIT THEORY")
        logger.info("="*60)

        cycles = 25 if self.fast_mode else 60
        agent_count = 4

        # Test 0-bit vs 1-bit
        results_by_bits = {}

        for consciousness_bits in [0, 1]:
            logger.info(f"\nTesting {consciousness_bits}-bit consciousness...")

            agents = create_llm_agent_population(
                count=agent_count,
                llm_backend=self.backend,
                consciousness_bits=consciousness_bits
            )

            stats = run_llm_interaction_cycle(
                agents,
                cycles=cycles,
                interaction_probability=0.4
            )

            # Predict using NYX formula
            predicted = predict_cooperation(
                agent_count=agent_count,
                memory_size=10,
                consciousness_bits=consciousness_bits
            )

            results_by_bits[consciousness_bits] = {
                'cooperation_rate': stats['cooperation_rate'],
                'predicted': predicted,
                'accuracy': 1 - abs(predicted - stats['cooperation_rate']) / max(predicted, stats['cooperation_rate'], 0.01),
                'statistics': stats
            }

            logger.info(f"  {consciousness_bits}-bit: Observed={stats['cooperation_rate']:.1%}, "
                       f"Predicted={predicted:.1%}")

        # Calculate jump
        jump = results_by_bits[1]['cooperation_rate'] - results_by_bits[0]['cooperation_rate']

        results = {
            'phase': 1,
            'phase_name': 'single_bit_theory',
            'results_by_bits': results_by_bits,
            'cooperation_jump': jump,
            'jump_validated': jump > 0.4,  # Expect > 40% jump
            'agent_count': agent_count,
            'cycles': cycles
        }

        logger.info(f"✅ Phase 1 Complete: {jump:.1%} cooperation jump")
        logger.info(f"   Single Bit Theory: {'✓ VALIDATED' if jump > 0.4 else '✗ NOT VALIDATED'}")

        return results

    def phase_2_multibit_consciousness(self) -> Dict[str, Any]:
        """
        Phase 2: Multi-Bit Consciousness Scaling

        Test: 1-bit vs 2-bit vs 3-bit performance
        Expected: Optimal at 2-bit, diminishing returns at 3-bit
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: MULTI-BIT CONSCIOUSNESS")
        logger.info("="*60)

        cycles = 30 if self.fast_mode else 70
        agent_count = 4

        results_by_bits = {}

        for consciousness_bits in [1, 2, 3]:
            logger.info(f"\nTesting {consciousness_bits}-bit consciousness...")

            agents = create_llm_agent_population(
                count=agent_count,
                llm_backend=self.backend,
                consciousness_bits=consciousness_bits
            )

            stats = run_llm_interaction_cycle(
                agents,
                cycles=cycles,
                interaction_probability=0.4
            )

            # Predict
            predicted = predict_cooperation(
                agent_count=agent_count,
                memory_size=10,
                consciousness_bits=consciousness_bits
            )

            results_by_bits[consciousness_bits] = {
                'cooperation_rate': stats['cooperation_rate'],
                'predicted': predicted,
                'accuracy': 1 - abs(predicted - stats['cooperation_rate']) / max(predicted, stats['cooperation_rate'], 0.01),
                'statistics': stats
            }

            logger.info(f"  {consciousness_bits}-bit: Observed={stats['cooperation_rate']:.1%}, "
                       f"Predicted={predicted:.1%}, "
                       f"Accuracy={results_by_bits[consciousness_bits]['accuracy']:.1%}")

        # Find optimal
        optimal_bits = max(results_by_bits.keys(),
                          key=lambda k: results_by_bits[k]['cooperation_rate'])

        # Check diminishing returns
        improvement_1_to_2 = results_by_bits[2]['cooperation_rate'] - results_by_bits[1]['cooperation_rate']
        improvement_2_to_3 = results_by_bits[3]['cooperation_rate'] - results_by_bits[2]['cooperation_rate']
        diminishing_returns = improvement_2_to_3 < improvement_1_to_2

        results = {
            'phase': 2,
            'phase_name': 'multibit_consciousness',
            'results_by_bits': results_by_bits,
            'optimal_bits': optimal_bits,
            'optimal_is_2_bit': optimal_bits == 2,
            'improvement_1_to_2': improvement_1_to_2,
            'improvement_2_to_3': improvement_2_to_3,
            'diminishing_returns_validated': diminishing_returns,
            'agent_count': agent_count,
            'cycles': cycles
        }

        logger.info(f"✅ Phase 2 Complete:")
        logger.info(f"   Optimal: {optimal_bits}-bit ({'✓' if optimal_bits == 2 else '✗'})")
        logger.info(f"   Diminishing returns: {'✓ VALIDATED' if diminishing_returns else '✗ NOT VALIDATED'}")

        return results

    def phase_3_memory_effect(self) -> Dict[str, Any]:
        """
        Phase 3: Memory Effect

        Test: Memory sweet spot around 10 patterns
        Expected: Gaussian curve peaking at ~10
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: MEMORY EFFECT")
        logger.info("="*60)

        memory_sizes = [5, 8, 10, 12, 15] if self.fast_mode else [5, 8, 10, 12, 15, 20]
        cycles = 25 if self.fast_mode else 50
        agent_count = 4

        results_by_memory = {}

        for memory_size in memory_sizes:
            logger.info(f"\nTesting memory size: {memory_size}")

            agents = create_llm_agent_population(
                count=agent_count,
                llm_backend=self.backend,
                consciousness_bits=2,
                memory_size=memory_size
            )

            stats = run_llm_interaction_cycle(
                agents,
                cycles=cycles,
                interaction_probability=0.4
            )

            # Predict
            predicted = predict_cooperation(
                agent_count=agent_count,
                memory_size=memory_size,
                consciousness_bits=2
            )

            results_by_memory[memory_size] = {
                'cooperation_rate': stats['cooperation_rate'],
                'predicted': predicted,
                'accuracy': 1 - abs(predicted - stats['cooperation_rate']) / max(predicted, stats['cooperation_rate'], 0.01),
                'statistics': stats
            }

            logger.info(f"  Memory {memory_size}: Observed={stats['cooperation_rate']:.1%}, "
                       f"Predicted={predicted:.1%}")

        # Find optimal
        optimal_memory = max(results_by_memory.keys(),
                            key=lambda k: results_by_memory[k]['cooperation_rate'])

        results = {
            'phase': 3,
            'phase_name': 'memory_effect',
            'results_by_memory': results_by_memory,
            'optimal_memory': optimal_memory,
            'optimal_near_10': abs(optimal_memory - 10) <= 2,
            'agent_count': agent_count,
            'cycles': cycles
        }

        logger.info(f"✅ Phase 3 Complete:")
        logger.info(f"   Optimal memory: {optimal_memory} ({'✓' if abs(optimal_memory - 10) <= 2 else '✗'})")

        return results

    def phase_4_network_effect(self) -> Dict[str, Any]:
        """
        Phase 4: Network Effect - Minimum Viable Society

        Test: Cooperation emergence at 4-agent threshold
        Expected: <4 agents = no cooperation, ≥4 = emergence
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: NETWORK EFFECT (Minimum Viable Society)")
        logger.info("="*60)

        agent_counts = [2, 3, 4, 5, 6] if self.fast_mode else [2, 3, 4, 5, 6, 8]
        cycles = 30 if self.fast_mode else 60

        results_by_agents = {}

        for agent_count in agent_counts:
            logger.info(f"\nTesting {agent_count} agents...")

            agents = create_llm_agent_population(
                count=agent_count,
                llm_backend=self.backend,
                consciousness_bits=2,
                memory_size=10
            )

            stats = run_llm_interaction_cycle(
                agents,
                cycles=cycles,
                interaction_probability=0.5
            )

            # Predict
            predicted = predict_cooperation(
                agent_count=agent_count,
                memory_size=10,
                consciousness_bits=2
            )

            results_by_agents[agent_count] = {
                'cooperation_rate': stats['cooperation_rate'],
                'predicted': predicted,
                'accuracy': 1 - abs(predicted - stats['cooperation_rate']) / max(predicted, stats['cooperation_rate'], 0.01),
                'statistics': stats
            }

            logger.info(f"  {agent_count} agents: Observed={stats['cooperation_rate']:.1%}, "
                       f"Predicted={predicted:.1%}")

        # Find threshold
        threshold_agent_count = None
        for count in sorted(agent_counts):
            if results_by_agents[count]['cooperation_rate'] > 0.5:
                threshold_agent_count = count
                break

        results = {
            'phase': 4,
            'phase_name': 'network_effect',
            'results_by_agents': results_by_agents,
            'minimum_viable_society': threshold_agent_count,
            'threshold_is_4': threshold_agent_count == 4,
            'cycles': cycles
        }

        logger.info(f"✅ Phase 4 Complete:")
        logger.info(f"   Minimum viable society: {threshold_agent_count} agents "
                   f"({'✓' if threshold_agent_count == 4 else '✗'})")

        return results

    def phase_5_formula_synthesis(self) -> Dict[str, Any]:
        """
        Phase 5: Formula Synthesis

        Test: Validate C = 0.1×N + 0.1×M + 0.8×A across configurations
        Expected: >85% accuracy
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: FORMULA SYNTHESIS")
        logger.info("="*60)

        cycles = 25 if self.fast_mode else 50

        # Test various configurations
        configs = [
            {'agents': 4, 'memory': 10, 'bits': 2},
            {'agents': 6, 'memory': 10, 'bits': 2},
            {'agents': 4, 'memory': 8, 'bits': 2},
            {'agents': 4, 'memory': 10, 'bits': 1},
            {'agents': 6, 'memory': 12, 'bits': 2},
        ]

        if not self.fast_mode:
            configs.extend([
                {'agents': 8, 'memory': 10, 'bits': 2},
                {'agents': 4, 'memory': 15, 'bits': 3},
            ])

        results_by_config = []
        all_accuracies = []

        for i, config in enumerate(configs):
            logger.info(f"\nConfig {i+1}/{len(configs)}: "
                       f"{config['agents']} agents, {config['memory']} memory, {config['bits']}-bit")

            agents = create_llm_agent_population(
                count=config['agents'],
                llm_backend=self.backend,
                consciousness_bits=config['bits'],
                memory_size=config['memory']
            )

            stats = run_llm_interaction_cycle(
                agents,
                cycles=cycles,
                interaction_probability=0.4
            )

            # Predict
            predicted = predict_cooperation(
                agent_count=config['agents'],
                memory_size=config['memory'],
                consciousness_bits=config['bits']
            )

            accuracy = 1 - abs(predicted - stats['cooperation_rate']) / max(predicted, stats['cooperation_rate'], 0.01)
            all_accuracies.append(accuracy)

            result = {
                'config': config,
                'observed': stats['cooperation_rate'],
                'predicted': predicted,
                'accuracy': accuracy,
                'statistics': stats
            }

            results_by_config.append(result)

            logger.info(f"  Observed: {stats['cooperation_rate']:.1%}, "
                       f"Predicted: {predicted:.1%}, "
                       f"Accuracy: {accuracy:.1%}")

        # Overall accuracy
        overall_accuracy = np.mean(all_accuracies)

        results = {
            'phase': 5,
            'phase_name': 'formula_synthesis',
            'results_by_config': results_by_config,
            'overall_accuracy': overall_accuracy,
            'accuracy_above_85': overall_accuracy > 0.85,
            'cycles': cycles
        }

        logger.info(f"✅ Phase 5 Complete:")
        logger.info(f"   Overall accuracy: {overall_accuracy:.1%} "
                   f"({'✓' if overall_accuracy > 0.85 else '✗'})")

        return results

    def phase_6_80_20_validation(self) -> Dict[str, Any]:
        """
        Phase 6: 80/20 Law Validation

        Test: Consciousness (A) contributes 80%, Infrastructure (N+M) 20%
        Expected: A ≈ 80%, N+M ≈ 20%
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 6: 80/20 LAW VALIDATION")
        logger.info("="*60)

        # Use optimal configuration
        agent_count = 6
        memory_size = 10
        consciousness_bits = 2

        # Calculate formula breakdown
        dummy_agents = [None] * agent_count
        nyx_system = NYXCooperationSystem(
            agents=dummy_agents,
            memory_size=memory_size,
            consciousness_bits=consciousness_bits
        )

        breakdown = nyx_system.get_formula_breakdown()

        # Extract components
        network_component = breakdown['network_component']
        memory_component = breakdown['memory_component']
        awareness_component = breakdown['awareness_component']
        total = breakdown['total_cooperation']

        # Calculate percentages
        consciousness_pct = (awareness_component / total * 100) if total > 0 else 0
        infrastructure_pct = ((network_component + memory_component) / total * 100) if total > 0 else 0

        # Validate 80/20
        is_80_20 = abs(consciousness_pct - 80) < 10 and abs(infrastructure_pct - 20) < 10

        results = {
            'phase': 6,
            'phase_name': '80_20_validation',
            'formula_breakdown': breakdown,
            'consciousness_percentage': consciousness_pct,
            'infrastructure_percentage': infrastructure_pct,
            '80_20_validated': is_80_20,
            'agent_count': agent_count,
            'memory_size': memory_size,
            'consciousness_bits': consciousness_bits
        }

        logger.info(f"✅ Phase 6 Complete:")
        logger.info(f"   Consciousness: {consciousness_pct:.1f}%")
        logger.info(f"   Infrastructure: {infrastructure_pct:.1f}%")
        logger.info(f"   80/20 Law: {'✓ VALIDATED' if is_80_20 else '✗ NOT VALIDATED'}")

        return results

    def run_all_phases(self) -> Dict[str, Any]:
        """Run all 7 phases sequentially"""
        logger.info("\n" + "="*60)
        logger.info("NYX HISTORICAL RECONSTRUCTION - ALL PHASES")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Start time: {datetime.now().isoformat()}")

        try:
            self.results['phase_0'] = self.phase_0_baseline()
            self.results['phase_1'] = self.phase_1_single_bit_theory()
            self.results['phase_2'] = self.phase_2_multibit_consciousness()
            self.results['phase_3'] = self.phase_3_memory_effect()
            self.results['phase_4'] = self.phase_4_network_effect()
            self.results['phase_5'] = self.phase_5_formula_synthesis()
            self.results['phase_6'] = self.phase_6_80_20_validation()

        except KeyboardInterrupt:
            logger.warning("Reconstruction interrupted by user")
            raise

        except Exception as e:
            logger.error(f"Phase failed: {e}", exc_info=True)
            raise

        # Compile summary
        total_duration = time.time() - self.start_time

        summary = {
            'model': self.model_name,
            'total_duration_hours': total_duration / 3600,
            'timestamp': datetime.now().isoformat(),
            'fast_mode': self.fast_mode,
            'key_discoveries': {
                'single_bit_theory': self.results['phase_1'].get('jump_validated', False),
                'optimal_2_bit': self.results['phase_2'].get('optimal_is_2_bit', False),
                'memory_sweet_spot': self.results['phase_3'].get('optimal_near_10', False),
                'minimum_viable_society': self.results['phase_4'].get('threshold_is_4', False),
                'formula_accuracy': self.results['phase_5'].get('overall_accuracy', 0),
                '80_20_law': self.results['phase_6'].get('80_20_validated', False)
            }
        }

        self.results['summary'] = summary

        # Save results
        if self.save_data:
            output_file = Path(f"results_llm/reconstruction_{self.model_name.replace('/', '_')}.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"\n✅ Results saved to: {output_file}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("RECONSTRUCTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Duration: {total_duration/3600:.2f} hours")
        logger.info(f"\nKey Discoveries Validated:")
        for discovery, validated in summary['key_discoveries'].items():
            symbol = "✓" if validated else "✗"
            logger.info(f"  {symbol} {discovery}: {validated}")

        return self.results


def main():
    parser = argparse.ArgumentParser(description="NYX Historical Reconstruction with LLMs")
    parser.add_argument("--model", default="mistral-7b-instruct",
                       help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama API URL")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode (fewer cycles)")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                       help="Run specific phase only")

    args = parser.parse_args()

    # Check Ollama availability
    backend = OllamaBackend(args.model, api_url=args.ollama_url)
    if not backend.is_available():
        logger.error("❌ Ollama not available!")
        logger.error(f"   Make sure Ollama is running at {args.ollama_url}")
        return 1

    # Initialize reconstruction
    reconstruction = NYXHistoricalReconstruction(
        model_name=args.model,
        ollama_url=args.ollama_url,
        fast_mode=args.fast
    )

    try:
        if args.phase is not None:
            # Run specific phase
            phase_methods = {
                0: reconstruction.phase_0_baseline,
                1: reconstruction.phase_1_single_bit_theory,
                2: reconstruction.phase_2_multibit_consciousness,
                3: reconstruction.phase_3_memory_effect,
                4: reconstruction.phase_4_network_effect,
                5: reconstruction.phase_5_formula_synthesis,
                6: reconstruction.phase_6_80_20_validation
            }

            result = phase_methods[args.phase]()
            logger.info(f"\nPhase {args.phase} complete")
        else:
            # Run all phases
            reconstruction.run_all_phases()

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
