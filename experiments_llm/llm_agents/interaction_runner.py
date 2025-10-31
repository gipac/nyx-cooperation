"""
Interaction Runner for LLM Agents
Manages multi-agent cooperation cycles
"""

import random
import logging
import numpy as np
from typing import List, Dict, Any
from .llm_nyx_agent import LLMNYXAgent
from nyx.agents import CooperationDecision

logger = logging.getLogger(__name__)


def run_llm_interaction_cycle(agents: List[LLMNYXAgent],
                               cycles: int = 100,
                               interaction_probability: float = 0.3,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Run interaction cycles between LLM agents

    Args:
        agents: List of LLM NYX agents
        cycles: Number of interaction cycles
        interaction_probability: Probability of any two agents interacting
        verbose: Print progress

    Returns:
        Dictionary with interaction statistics
    """
    logger.info(f"Running {cycles} LLM interaction cycles with {len(agents)} agents")

    total_interactions = 0
    total_cooperations = 0
    mutual_cooperations = 0
    mutual_defections = 0
    exploitation_events = 0  # One cooperates, one defects

    for cycle in range(cycles):
        if verbose and cycle % 10 == 0:
            logger.info(f"Cycle {cycle}/{cycles}")

        # Randomly pair agents
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if random.random() < interaction_probability:
                    agent_a, agent_b = agents[i], agents[j]

                    # Record pre-interaction state
                    decision_a_pre = agent_a.make_cooperation_decision()
                    decision_b_pre = agent_b.make_cooperation_decision()

                    # Agents interact
                    benefits_a, costs_a = agent_a.interact_with_agent(agent_b)
                    benefits_b, costs_b = agent_b.interact_with_agent(agent_a)

                    total_interactions += 2

                    # Count cooperation types
                    if decision_a_pre == CooperationDecision.COOPERATE:
                        total_cooperations += 1
                    if decision_b_pre == CooperationDecision.COOPERATE:
                        total_cooperations += 1

                    # Analyze interaction type
                    if (decision_a_pre == CooperationDecision.COOPERATE and
                        decision_b_pre == CooperationDecision.COOPERATE):
                        mutual_cooperations += 1
                    elif (decision_a_pre == CooperationDecision.DEFECT and
                          decision_b_pre == CooperationDecision.DEFECT):
                        mutual_defections += 1
                    else:
                        exploitation_events += 1

    # Calculate statistics
    cooperation_rate = total_cooperations / total_interactions if total_interactions > 0 else 0
    avg_roi = np.mean([agent.calculate_cooperation_roi() for agent in agents])
    avg_energy = np.mean([agent.energy for agent in agents])

    # LLM statistics
    total_llm_calls = sum(agent.llm_calls for agent in agents)
    total_llm_failures = sum(agent.llm_failures for agent in agents)

    stats = {
        'cooperation_rate': cooperation_rate,
        'total_interactions': total_interactions,
        'total_cooperations': total_cooperations,
        'mutual_cooperations': mutual_cooperations,
        'mutual_defections': mutual_defections,
        'exploitation_events': exploitation_events,
        'average_roi': avg_roi,
        'average_energy': avg_energy,
        'agents_analyzed': len(agents),
        'llm_calls': total_llm_calls,
        'llm_failures': total_llm_failures,
        'llm_failure_rate': total_llm_failures / total_llm_calls if total_llm_calls > 0 else 0
    }

    logger.info(f"Cycle complete: {cooperation_rate:.1%} cooperation, "
               f"{total_llm_calls} LLM calls, "
               f"{total_llm_failures} failures")

    return stats


def create_llm_agent_population(count: int,
                                 llm_backend,
                                 consciousness_bits: int = 2,
                                 **kwargs) -> List[LLMNYXAgent]:
    """
    Create a population of LLM agents

    Args:
        count: Number of agents
        llm_backend: LLM backend to use
        consciousness_bits: Consciousness level
        **kwargs: Additional agent parameters

    Returns:
        List of LLM NYX agents
    """
    agents = []

    for i in range(count):
        agent = LLMNYXAgent(
            agent_id=f"llm_agent_{i}",
            llm_backend=llm_backend,
            consciousness_bits=consciousness_bits,
            **kwargs
        )
        agents.append(agent)

    logger.info(f"Created {len(agents)} LLM agents with {consciousness_bits}-bit consciousness")

    return agents
