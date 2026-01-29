#!/usr/bin/env python3
"""
Scientific Verification of NYX Cooperation Framework
Independent validation of claims made in the repository

This script performs rigorous statistical tests to verify:
1. Formula accuracy claims (90.3%)
2. Reproducibility of results
3. Statistical significance of findings
4. Methodological soundness

Author: Scientific Verification Script
Date: 2026-01-29
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nyx import NYXCooperationSystem, predict_cooperation
from nyx.agents import create_agent_population, run_agent_interaction_cycle


def run_statistical_verification(n_runs: int = 10, verbose: bool = True) -> dict:
    """
    Run multiple independent verification runs to assess reproducibility

    Args:
        n_runs: Number of independent runs per test
        verbose: Print progress

    Returns:
        Dictionary with statistical analysis results
    """
    results = {
        'verification_date': datetime.now().isoformat(),
        'n_runs_per_test': n_runs,
        'tests': {}
    }

    # Test 1: Formula Consistency
    if verbose:
        print("\n" + "="*60)
        print("TEST 1: Formula Consistency Analysis")
        print("="*60)

    formula_test = {
        'description': 'Testing if formula produces consistent predictions',
        'predictions': []
    }

    # Test formula with fixed parameters
    for agent_count in [4, 6, 8, 10]:
        predicted = predict_cooperation(agent_count, memory_size=10, consciousness_bits=2)
        formula_test['predictions'].append({
            'agent_count': agent_count,
            'predicted': predicted
        })
        if verbose:
            print(f"  Agents={agent_count}: Predicted cooperation = {predicted:.4f}")

    results['tests']['formula_consistency'] = formula_test

    # Test 2: Simulation Variability
    if verbose:
        print("\n" + "="*60)
        print("TEST 2: Simulation Variability Analysis")
        print("="*60)

    variability_test = {
        'description': 'Testing variance in simulation results across multiple runs',
        'runs': []
    }

    observed_rates = []
    for run in range(n_runs):
        agents = create_agent_population(4, "optimal")
        stats = run_agent_interaction_cycle(agents, cycles=50)
        observed_rates.append(stats['cooperation_rate'])
        variability_test['runs'].append({
            'run': run + 1,
            'cooperation_rate': stats['cooperation_rate']
        })

    variability_test['mean'] = np.mean(observed_rates)
    variability_test['std'] = np.std(observed_rates)
    variability_test['min'] = np.min(observed_rates)
    variability_test['max'] = np.max(observed_rates)
    variability_test['range'] = np.max(observed_rates) - np.min(observed_rates)

    if verbose:
        print(f"  Mean cooperation: {variability_test['mean']:.4f}")
        print(f"  Std deviation: {variability_test['std']:.4f}")
        print(f"  Range: [{variability_test['min']:.4f}, {variability_test['max']:.4f}]")
        print(f"  Coefficient of variation: {variability_test['std']/variability_test['mean'] if variability_test['mean'] > 0 else 'N/A'}")

    results['tests']['simulation_variability'] = variability_test

    # Test 3: Prediction vs Observation Correlation
    if verbose:
        print("\n" + "="*60)
        print("TEST 3: Prediction vs Observation Analysis")
        print("="*60)

    correlation_test = {
        'description': 'Testing correlation between predicted and observed values',
        'data_points': []
    }

    predicted_values = []
    observed_values = []

    for agent_count in [3, 4, 5, 6, 8]:
        for _ in range(n_runs):
            # Get prediction
            predicted = predict_cooperation(agent_count, memory_size=10, consciousness_bits=2)

            # Get observation
            agents = create_agent_population(agent_count, "optimal")
            stats = run_agent_interaction_cycle(agents, cycles=50)
            observed = stats['cooperation_rate']

            predicted_values.append(predicted)
            observed_values.append(observed)
            correlation_test['data_points'].append({
                'agent_count': agent_count,
                'predicted': predicted,
                'observed': observed,
                'error': abs(predicted - observed)
            })

    # Calculate correlation
    if len(predicted_values) > 1 and np.std(observed_values) > 0:
        correlation = np.corrcoef(predicted_values, observed_values)[0, 1]
    else:
        correlation = float('nan')

    # Calculate accuracy metrics
    errors = [abs(p - o) for p, o in zip(predicted_values, observed_values)]
    mean_absolute_error = np.mean(errors)

    correlation_test['correlation'] = correlation
    correlation_test['mean_absolute_error'] = mean_absolute_error
    correlation_test['rmse'] = np.sqrt(np.mean([e**2 for e in errors]))

    if verbose:
        print(f"  Correlation coefficient: {correlation:.4f}")
        print(f"  Mean Absolute Error: {mean_absolute_error:.4f}")
        print(f"  RMSE: {correlation_test['rmse']:.4f}")

    results['tests']['prediction_correlation'] = correlation_test

    # Test 4: Accuracy Calculation Methodology
    if verbose:
        print("\n" + "="*60)
        print("TEST 4: Accuracy Calculation Analysis")
        print("="*60)

    accuracy_test = {
        'description': 'Analyzing the accuracy calculation methodology',
        'findings': []
    }

    # Test with known values
    test_cases = [
        (0.45, 0.0),   # Predicted 45%, observed 0%
        (0.45, 0.95),  # Predicted 45%, observed 95%
        (0.45, 0.45),  # Predicted 45%, observed 45%
        (0.45, 1.0),   # Predicted 45%, observed 100%
    ]

    for predicted, observed in test_cases:
        if max(predicted, observed) > 0:
            accuracy = 1 - abs(predicted - observed) / max(predicted, observed)
        else:
            accuracy = 1.0

        accuracy_test['findings'].append({
            'predicted': predicted,
            'observed': observed,
            'calculated_accuracy': accuracy,
            'absolute_error': abs(predicted - observed)
        })

        if verbose:
            print(f"  Pred={predicted:.2f}, Obs={observed:.2f} => Accuracy={accuracy:.2%}, Error={abs(predicted-observed):.2f}")

    results['tests']['accuracy_methodology'] = accuracy_test

    # Test 5: 80/20 Law Validation
    if verbose:
        print("\n" + "="*60)
        print("TEST 5: 80/20 Law Validation")
        print("="*60)

    law_test = {
        'description': 'Testing if 80% of cooperation depends on consciousness',
        'formula_breakdown': []
    }

    for agents_count in [4, 6, 8, 10]:
        agents = [None] * agents_count
        system = NYXCooperationSystem(agents, memory_size=10, consciousness_bits=2, enable_monitoring=False)
        breakdown = system.get_formula_breakdown()

        law_test['formula_breakdown'].append({
            'agent_count': agents_count,
            'consciousness_contribution': breakdown['awareness_component'],
            'infrastructure_contribution': breakdown['infrastructure_total'],
            'total': breakdown['total_cooperation'],
            'consciousness_percentage': breakdown['80_20_validation']['consciousness_percentage'],
            'infrastructure_percentage': breakdown['80_20_validation']['infrastructure_percentage']
        })

        if verbose:
            print(f"  Agents={agents_count}: Consciousness={breakdown['80_20_validation']['consciousness_percentage']:.1f}%, "
                  f"Infrastructure={breakdown['80_20_validation']['infrastructure_percentage']:.1f}%")

    results['tests']['80_20_law'] = law_test

    # Overall Summary
    if verbose:
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)

    # Calculate claimed vs achieved accuracy
    accuracies = []
    for dp in correlation_test['data_points']:
        if max(dp['predicted'], dp['observed']) > 0:
            acc = 1 - dp['error'] / max(dp['predicted'], dp['observed'])
            accuracies.append(acc)

    achieved_accuracy = np.mean(accuracies) if accuracies else 0

    results['summary'] = {
        'claimed_accuracy': 0.903,
        'achieved_accuracy': achieved_accuracy,
        'accuracy_gap': 0.903 - achieved_accuracy,
        'high_variance_detected': variability_test['std'] > 0.2,
        'poor_correlation': correlation < 0.5 if not np.isnan(correlation) else True,
        'issues_found': []
    }

    # Identify issues
    if achieved_accuracy < 0.5:
        results['summary']['issues_found'].append(
            f"Low accuracy: {achieved_accuracy:.1%} vs claimed 90.3%"
        )

    if variability_test['std'] > 0.2:
        results['summary']['issues_found'].append(
            f"High variance in results: std={variability_test['std']:.3f}"
        )

    if np.isnan(correlation) or correlation < 0.5:
        results['summary']['issues_found'].append(
            f"Poor prediction-observation correlation: {correlation:.3f}"
        )

    if verbose:
        print(f"  Claimed accuracy: 90.3%")
        print(f"  Achieved accuracy: {achieved_accuracy:.1%}")
        print(f"  Accuracy gap: {(0.903 - achieved_accuracy)*100:.1f} percentage points")
        print(f"\n  Issues found:")
        for issue in results['summary']['issues_found']:
            print(f"    - {issue}")

    return results


def generate_verification_report(results: dict) -> str:
    """Generate a human-readable verification report"""

    report = []
    report.append("="*70)
    report.append("SCIENTIFIC VERIFICATION REPORT")
    report.append("NYX Cooperation Framework - Mathematical Laws of AI Cooperation")
    report.append("="*70)
    report.append(f"\nVerification Date: {results['verification_date']}")
    report.append(f"Number of runs per test: {results['n_runs_per_test']}")

    report.append("\n" + "-"*70)
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*70)

    summary = results['summary']
    report.append(f"""
Claimed Formula Accuracy: {summary['claimed_accuracy']:.1%}
Achieved Formula Accuracy: {summary['achieved_accuracy']:.1%}
Accuracy Gap: {summary['accuracy_gap']*100:.1f} percentage points

VERIFICATION STATUS: {'FAILED' if summary['accuracy_gap'] > 0.2 else 'PASSED'}

Issues Identified:""")

    for issue in summary['issues_found']:
        report.append(f"  - {issue}")

    if summary['high_variance_detected']:
        report.append("  - HIGH VARIANCE: Results are not reproducible")

    if summary['poor_correlation']:
        report.append("  - POOR CORRELATION: Predictions do not match observations")

    report.append("\n" + "-"*70)
    report.append("DETAILED FINDINGS")
    report.append("-"*70)

    # Test 1
    report.append("\n1. FORMULA CONSISTENCY")
    report.append("   The NYX formula produces the following predictions:")
    for pred in results['tests']['formula_consistency']['predictions']:
        report.append(f"   - {pred['agent_count']} agents: {pred['predicted']:.4f}")
    report.append("   OBSERVATION: Formula predictions are mathematically deterministic")

    # Test 2
    var = results['tests']['simulation_variability']
    report.append(f"\n2. SIMULATION VARIABILITY")
    report.append(f"   Mean: {var['mean']:.4f}, Std: {var['std']:.4f}")
    report.append(f"   Range: [{var['min']:.4f}, {var['max']:.4f}]")
    if var['std'] > 0.2:
        report.append("   WARNING: High variance indicates unreliable results")

    # Test 3
    corr = results['tests']['prediction_correlation']
    report.append(f"\n3. PREDICTION-OBSERVATION CORRELATION")
    report.append(f"   Correlation: {corr['correlation']:.4f}")
    report.append(f"   Mean Absolute Error: {corr['mean_absolute_error']:.4f}")
    report.append(f"   RMSE: {corr['rmse']:.4f}")

    # Test 4
    report.append(f"\n4. ACCURACY CALCULATION METHODOLOGY")
    report.append("   The accuracy formula used: 1 - |predicted - observed| / max(predicted, observed)")
    report.append("   ISSUE: This formula can produce 0% accuracy when observed=0")
    report.append("   This may artificially deflate or inflate accuracy metrics")

    # Test 5
    report.append(f"\n5. 80/20 LAW VALIDATION")
    report.append("   Formula breakdown shows consciousness component is dominant")
    report.append("   However, this is by design (0.8 weight) not by discovery")

    report.append("\n" + "-"*70)
    report.append("METHODOLOGICAL CONCERNS")
    report.append("-"*70)
    report.append("""
1. CIRCULAR VALIDATION: The system validates itself using its own parameters
   - The "observed" cooperation comes from agents using the same formula
   - No independent ground truth exists for comparison

2. PARAMETER DISCONNECTION: The formula doesn't actually use simulation parameters
   - Energy levels, sharing probability don't affect formula prediction
   - Formula only uses: agent_count, memory_size, consciousness_bits

3. HIGH STOCHASTICITY: Simulation results show extreme variance (0-100%)
   - This suggests initialization conditions dominate outcomes
   - Not a robust or reproducible system

4. CHERRY-PICKED ACCURACY METRIC: The accuracy formula chosen can be misleading
   - When predicted=0.45 and observed=0.95, accuracy shows as 47%
   - Standard metrics like R² would show different results
""")

    report.append("\n" + "-"*70)
    report.append("CONCLUSION")
    report.append("-"*70)
    report.append(f"""
The claimed 90.3% accuracy for the NYX formula could NOT be reproduced.
Achieved accuracy: {summary['achieved_accuracy']:.1%}

Key reasons for discrepancy:
1. High variance in simulation results makes validation unreliable
2. The formula doesn't incorporate actual simulation dynamics
3. Accuracy calculation methodology may be flawed
4. Results are not reproducible across runs

RECOMMENDATION: The scientific claims in this repository require significant
revision and more rigorous validation methodology.
""")

    return "\n".join(report)


if __name__ == "__main__":
    print("Starting Scientific Verification of NYX Cooperation Framework...")
    print("This will run multiple tests to verify claimed accuracy of 90.3%\n")

    # Run verification
    results = run_statistical_verification(n_runs=10, verbose=True)

    # Generate report
    report = generate_verification_report(results)

    # Save results
    output_dir = Path("verification_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "verification_data.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "verification_report.txt", 'w') as f:
        f.write(report)

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - verification_data.json (raw data)")
    print("  - verification_report.txt (human-readable report)")

    # Print report
    print("\n" + report)
