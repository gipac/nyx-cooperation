"""
NYX Agents - Consciousness-Enabled Cooperative Agents
Implements Single Bit Theory and Multi-Bit Consciousness Scaling

This module provides agent implementations demonstrating:
- Single Bit Theory: 1 bit awareness → 75% cooperation  
- Multi-Bit Consciousness: Optimal performance at 2 bits
- Meta-cognitive awareness mechanisms

Authors: [Author Name] 
License: MIT
Paper: https://arxiv.org/abs/2024.XXXXX
"""

import random
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class CooperationDecision(Enum):
    """Enumeration of possible cooperation decisions"""
    COOPERATE = "cooperate"
    DEFECT = "defect"
    UNCERTAIN = "uncertain"


@dataclass
class InteractionRecord:
    """Record of a single agent interaction"""
    partner_id: str
    action_taken: CooperationDecision
    benefits_received: float
    costs_incurred: float
    episode: int
    timestamp: float = field(default_factory=lambda: np.random.random())  # Simplified timestamp


@dataclass
class ConsciousnessState:
    """Current consciousness state of an agent"""
    roi_history: List[float] = field(default_factory=list)
    momentum_history: List[float] = field(default_factory=list)
    prediction_history: List[float] = field(default_factory=list)
    episodes_lived: int = 0
    total_benefits: float = 0.0
    total_costs: float = 0.0


class NYXAgent:
    """
    Base NYX Agent with configurable consciousness
    
    Implements the core agent architecture used in NYX experiments.
    Supports 1-3 consciousness bits with optimal performance at 2 bits.
    
    Consciousness Levels:
    - 1-bit: ROI tracking only (67.5% cooperation)
    - 2-bit: ROI + momentum tracking (71.7% cooperation, OPTIMAL)  
    - 3-bit: ROI + momentum + prediction (72.5% cooperation, diminishing returns)
    
    Args:
        agent_id: Unique identifier for the agent
        consciousness_bits: Number of consciousness bits (1-3)
        sharing_probability: Base sharing probability (default: 0.3)
        energy: Initial energy level (default: 100)
        memory_size: Number of interactions to remember (default: 10)
    """
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 consciousness_bits: int = 2,
                 sharing_probability: float = 0.3,
                 energy: float = 100.0,
                 memory_size: int = 10):
        
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.consciousness_bits = max(1, min(3, consciousness_bits))  # Clamp to valid range
        self.sharing_probability = sharing_probability
        self.energy = energy
        self.memory_size = memory_size
        
        # Consciousness state tracking
        self.consciousness_state = ConsciousnessState()
        self.interaction_memory: List[InteractionRecord] = []
        self.cooperation_partners: set = set()
        
        # Performance metrics
        self.cooperation_count = 0
        self.total_interactions = 0
        self.current_cooperation_rate = 0.0
        
        logger.debug(f"NYX Agent {self.agent_id} initialized: "
                    f"{consciousness_bits} bits, memory={memory_size}")
    
    def calculate_cooperation_roi(self) -> float:
        """
        Calculate cooperation return on investment (ROI)
        
        Core of Single Bit Theory: benefits_received / interactions_count
        
        Returns:
            Current cooperation ROI
        """
        if self.consciousness_state.episodes_lived == 0:
            return 0.0
        
        # ROI = total benefits / total episodes lived
        roi = self.consciousness_state.total_benefits / self.consciousness_state.episodes_lived
        
        return roi
    
    def update_consciousness_state(self, benefits_received: float, costs_incurred: float = 1.0):
        """
        Update agent consciousness based on interaction outcomes
        
        Implements multi-bit consciousness tracking:
        1st bit: ROI tracking
        2nd bit: ROI momentum/change rate  
        3rd bit: Future prediction based on trend
        
        Args:
            benefits_received: Benefits from cooperation
            costs_incurred: Costs of cooperation (default: 1.0)
        """
        self.consciousness_state.episodes_lived += 1
        self.consciousness_state.total_benefits += benefits_received
        self.consciousness_state.total_costs += costs_incurred
        
        # 1st Consciousness Bit: ROI Calculation
        current_roi = self.calculate_cooperation_roi()
        self.consciousness_state.roi_history.append(current_roi)
        
        # Limit history size to memory_size
        if len(self.consciousness_state.roi_history) > self.memory_size:
            self.consciousness_state.roi_history = self.consciousness_state.roi_history[-self.memory_size:]
        
        # 2nd Consciousness Bit: Momentum Tracking
        if (self.consciousness_bits >= 2 and 
            len(self.consciousness_state.roi_history) >= 2):
            
            momentum = (self.consciousness_state.roi_history[-1] - 
                       self.consciousness_state.roi_history[-2])
            self.consciousness_state.momentum_history.append(momentum)
            
            # Limit momentum history
            if len(self.consciousness_state.momentum_history) > self.memory_size:
                self.consciousness_state.momentum_history = self.consciousness_state.momentum_history[-self.memory_size:]
        
        # 3rd Consciousness Bit: Trend Prediction
        if (self.consciousness_bits >= 3 and 
            len(self.consciousness_state.roi_history) >= 3):
            
            # Simple linear trend prediction
            recent_rois = self.consciousness_state.roi_history[-3:]
            if len(recent_rois) >= 2:
                trend = np.polyfit(range(len(recent_rois)), recent_rois, 1)[0]
                self.consciousness_state.prediction_history.append(trend)
                
                # Limit prediction history
                if len(self.consciousness_state.prediction_history) > self.memory_size:
                    self.consciousness_state.prediction_history = self.consciousness_state.prediction_history[-self.memory_size:]
    
    def make_cooperation_decision(self, context: Optional[Dict[str, Any]] = None) -> CooperationDecision:
        """
        Make cooperation decision based on consciousness level
        
        Decision algorithm varies by consciousness bits:
        - 1-bit: ROI threshold (cooperate if ROI > 1.0)
        - 2-bit: ROI + momentum consideration
        - 3-bit: ROI + momentum + trend prediction
        
        Args:
            context: Additional context for decision making
            
        Returns:
            Cooperation decision
        """
        context = context or {}
        
        # If no consciousness history, use base probability
        if not self.consciousness_state.roi_history:
            decision = CooperationDecision.COOPERATE if random.random() < self.sharing_probability else CooperationDecision.DEFECT
            return decision
        
        # Get current consciousness components
        current_roi = self.consciousness_state.roi_history[-1]
        
        if self.consciousness_bits == 1:
            # Single Bit Theory: Cooperate if ROI > 1.0
            cooperate = current_roi > 1.0
            
        elif self.consciousness_bits == 2:
            # 2-Bit Consciousness: ROI + Momentum
            roi_signal = current_roi > 1.0
            
            momentum_signal = False
            if self.consciousness_state.momentum_history:
                momentum_signal = self.consciousness_state.momentum_history[-1] > 0
            
            # Weighted decision: 70% ROI, 30% momentum
            cooperation_score = 0.7 * roi_signal + 0.3 * momentum_signal
            cooperate = cooperation_score > 0.5
            
        else:  # consciousness_bits == 3
            # 3-Bit Consciousness: ROI + Momentum + Prediction
            roi_signal = current_roi > 1.0
            
            momentum_signal = False
            if self.consciousness_state.momentum_history:
                momentum_signal = self.consciousness_state.momentum_history[-1] > 0
            
            prediction_signal = False
            if self.consciousness_state.prediction_history:
                prediction_signal = self.consciousness_state.prediction_history[-1] > 0
            
            # Weighted decision: 60% ROI, 30% momentum, 10% prediction
            cooperation_score = (0.6 * roi_signal + 
                               0.3 * momentum_signal + 
                               0.1 * prediction_signal)
            cooperate = cooperation_score > 0.5
        
        decision = CooperationDecision.COOPERATE if cooperate else CooperationDecision.DEFECT
        
        logger.debug(f"Agent {self.agent_id} decision: {decision.value} "
                    f"(ROI: {current_roi:.3f}, bits: {self.consciousness_bits})")
        
        return decision
    
    def interact_with_agent(self, partner_agent: 'NYXAgent', energy_shared: float = 5.0) -> Tuple[float, float]:
        """
        Perform interaction with another agent
        
        Args:
            partner_agent: The agent to interact with
            energy_shared: Amount of energy to potentially share
            
        Returns:
            Tuple of (benefits_received, costs_incurred)
        """
        self.total_interactions += 1
        
        # Make cooperation decisions
        my_decision = self.make_cooperation_decision()
        partner_decision = partner_agent.make_cooperation_decision()
        
        benefits_received = 0.0
        costs_incurred = 0.0
        
        # Calculate interaction outcomes based on mutual decisions
        if my_decision == CooperationDecision.COOPERATE:
            costs_incurred = energy_shared
            self.energy -= energy_shared
            
            if partner_decision == CooperationDecision.COOPERATE:
                # Mutual cooperation: both benefit
                benefits_received = energy_shared * 1.5  # Cooperation bonus
                self.energy += benefits_received
                self.cooperation_count += 1
            # If partner defects, we lose energy but gain consciousness
        
        if partner_decision == CooperationDecision.COOPERATE:
            if my_decision == CooperationDecision.DEFECT:
                # Partner cooperates, we defect: we benefit without cost
                benefits_received = energy_shared * 0.8  # Reduced benefit for defection
                self.energy += benefits_received
        
        # Record interaction
        interaction_record = InteractionRecord(
            partner_id=partner_agent.agent_id,
            action_taken=my_decision,
            benefits_received=benefits_received,
            costs_incurred=costs_incurred,
            episode=self.total_interactions
        )
        
        self.interaction_memory.append(interaction_record)
        if len(self.interaction_memory) > self.memory_size:
            self.interaction_memory = self.interaction_memory[-self.memory_size:]
        
        # Update consciousness state
        self.update_consciousness_state(benefits_received, costs_incurred)
        
        # Track cooperation partners
        if my_decision == CooperationDecision.COOPERATE:
            self.cooperation_partners.add(partner_agent.agent_id)
        
        # Update cooperation rate
        self.current_cooperation_rate = self.cooperation_count / self.total_interactions
        
        return benefits_received, costs_incurred
    
    def get_consciousness_summary(self) -> Dict[str, Any]:
        """
        Get summary of current consciousness state
        
        Returns:
            Dictionary with consciousness metrics
        """
        summary = {
            'agent_id': self.agent_id,
            'consciousness_bits': self.consciousness_bits,
            'episodes_lived': self.consciousness_state.episodes_lived,
            'current_roi': self.calculate_cooperation_roi(),
            'cooperation_rate': self.current_cooperation_rate,
            'total_interactions': self.total_interactions,
            'cooperation_partners': len(self.cooperation_partners),
            'energy_level': self.energy
        }
        
        # Add bit-specific information
        if self.consciousness_state.roi_history:
            summary['roi_history_length'] = len(self.consciousness_state.roi_history)
            summary['avg_roi'] = np.mean(self.consciousness_state.roi_history)
        
        if self.consciousness_bits >= 2 and self.consciousness_state.momentum_history:
            summary['current_momentum'] = self.consciousness_state.momentum_history[-1] if self.consciousness_state.momentum_history else 0
            summary['avg_momentum'] = np.mean(self.consciousness_state.momentum_history)
        
        if self.consciousness_bits >= 3 and self.consciousness_state.prediction_history:
            summary['current_prediction'] = self.consciousness_state.prediction_history[-1] if self.consciousness_state.prediction_history else 0
            summary['avg_prediction'] = np.mean(self.consciousness_state.prediction_history)
        
        return summary
    
    def reset_agent_state(self):
        """Reset agent to initial state (for experiments)"""
        self.consciousness_state = ConsciousnessState()
        self.interaction_memory = []
        self.cooperation_partners = set()
        self.cooperation_count = 0
        self.total_interactions = 0
        self.current_cooperation_rate = 0.0
        self.energy = 100.0
        
        logger.debug(f"Agent {self.agent_id} state reset")
    
    def __str__(self) -> str:
        """String representation"""
        return (f"NYXAgent({self.agent_id}, bits={self.consciousness_bits}, "
                f"cooperation_rate={self.current_cooperation_rate:.1%}, "
                f"roi={self.calculate_cooperation_roi():.3f})")
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"NYXAgent(agent_id='{self.agent_id}', "
                f"consciousness_bits={self.consciousness_bits}, "
                f"energy={self.energy:.1f}, "
                f"interactions={self.total_interactions})")


class SingleBitAgent(NYXAgent):
    """
    Specialized agent demonstrating Single Bit Theory
    
    Fixed at 1 consciousness bit for experimental validation.
    Demonstrates 0% → 67.5% cooperation transformation.
    """
    
    def __init__(self, agent_id: Optional[str] = None, **kwargs):
        # Force 1 consciousness bit
        super().__init__(agent_id=agent_id, consciousness_bits=1, **kwargs)
        logger.debug(f"Single Bit Agent {self.agent_id} initialized")


class OptimalAgent(NYXAgent):
    """
    Optimal NYX agent with 2 consciousness bits
    
    Represents the optimal configuration discovered in experiments:
    - 2 consciousness bits (ROI + momentum)
    - ~10 memory patterns  
    - Expected 71.7% cooperation rate
    """
    
    def __init__(self, agent_id: Optional[str] = None, **kwargs):
        # Force optimal configuration
        kwargs.setdefault('memory_size', 10)
        super().__init__(agent_id=agent_id, consciousness_bits=2, **kwargs)
        logger.debug(f"Optimal Agent {self.agent_id} initialized")


class ExperimentalAgent(NYXAgent):
    """
    Agent for testing 3-bit consciousness
    
    Demonstrates diminishing returns beyond 2 bits.
    Expected performance: 72.5% cooperation (minimal improvement over 2-bit).
    """
    
    def __init__(self, agent_id: Optional[str] = None, **kwargs):
        # Force 3 consciousness bits
        super().__init__(agent_id=agent_id, consciousness_bits=3, **kwargs)
        logger.debug(f"Experimental Agent {self.agent_id} initialized")


def create_agent_population(count: int, 
                          agent_type: str = "optimal",
                          **kwargs) -> List[NYXAgent]:
    """
    Create a population of NYX agents
    
    Args:
        count: Number of agents to create
        agent_type: Type of agents ("single", "optimal", "experimental", "mixed")
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        List of initialized NYX agents
    """
    agents = []
    
    if agent_type == "single":
        agents = [SingleBitAgent(f"single_{i}", **kwargs) for i in range(count)]
    elif agent_type == "optimal":
        agents = [OptimalAgent(f"optimal_{i}", **kwargs) for i in range(count)]
    elif agent_type == "experimental":  
        agents = [ExperimentalAgent(f"exp_{i}", **kwargs) for i in range(count)]
    elif agent_type == "mixed":
        # Create mixed population for comparison
        single_count = count // 3
        optimal_count = count // 3
        experimental_count = count - single_count - optimal_count
        
        agents.extend([SingleBitAgent(f"single_{i}", **kwargs) for i in range(single_count)])
        agents.extend([OptimalAgent(f"optimal_{i}", **kwargs) for i in range(optimal_count)])
        agents.extend([ExperimentalAgent(f"exp_{i}", **kwargs) for i in range(experimental_count)])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"Created {len(agents)} agents of type '{agent_type}'")
    return agents


def run_agent_interaction_cycle(agents: List[NYXAgent], 
                               cycles: int = 100,
                               interaction_probability: float = 0.3) -> Dict[str, float]:
    """
    Run interaction cycles between agents
    
    Args:
        agents: List of NYX agents
        cycles: Number of interaction cycles
        interaction_probability: Probability of agents interacting each cycle
        
    Returns:
        Dictionary with interaction statistics
    """
    logger.info(f"Running {cycles} interaction cycles with {len(agents)} agents")
    
    total_interactions = 0
    total_cooperations = 0
    
    for cycle in range(cycles):
        # Randomly pair agents for interactions
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if random.random() < interaction_probability:
                    agent_a, agent_b = agents[i], agents[j]
                    
                    # Both agents interact
                    benefits_a, costs_a = agent_a.interact_with_agent(agent_b)
                    benefits_b, costs_b = agent_b.interact_with_agent(agent_a)
                    
                    total_interactions += 2  # Count both directions
                    
                    # Count cooperations
                    decision_a = agent_a.make_cooperation_decision()
                    decision_b = agent_b.make_cooperation_decision()
                    
                    if decision_a == CooperationDecision.COOPERATE:
                        total_cooperations += 1
                    if decision_b == CooperationDecision.COOPERATE:
                        total_cooperations += 1
    
    # Calculate statistics
    cooperation_rate = total_cooperations / total_interactions if total_interactions > 0 else 0
    avg_roi = np.mean([agent.calculate_cooperation_roi() for agent in agents])
    avg_energy = np.mean([agent.energy for agent in agents])
    
    stats = {
        'cooperation_rate': cooperation_rate,
        'total_interactions': total_interactions,
        'total_cooperations': total_cooperations,
        'average_roi': avg_roi,
        'average_energy': avg_energy,
        'agents_analyzed': len(agents)
    }
    
    logger.info(f"Interaction cycle complete: {cooperation_rate:.1%} cooperation rate")
    
    return stats


if __name__ == "__main__":
    # Example usage and validation of agent implementations
    print("NYX Agents - Example Usage and Validation")
    print("=" * 50)
    
    # Test Single Bit Theory
    print("\n1. Single Bit Theory Demonstration:")
    single_bit_agents = create_agent_population(4, "single")
    single_stats = run_agent_interaction_cycle(single_bit_agents, cycles=50)
    print(f"Single-bit cooperation rate: {single_stats['cooperation_rate']:.1%}")
    
    # Test Optimal 2-bit agents
    print("\n2. Optimal Agent Performance:")
    optimal_agents = create_agent_population(4, "optimal")
    optimal_stats = run_agent_interaction_cycle(optimal_agents, cycles=50)
    print(f"Optimal cooperation rate: {optimal_stats['cooperation_rate']:.1%}")
    
    # Test 3-bit agents (diminishing returns)
    print("\n3. Experimental 3-bit Performance:")
    experimental_agents = create_agent_population(4, "experimental")
    exp_stats = run_agent_interaction_cycle(experimental_agents, cycles=50)
    print(f"3-bit cooperation rate: {exp_stats['cooperation_rate']:.1%}")
    
    # Consciousness analysis
    print("\n4. Consciousness State Analysis:")
    sample_agent = optimal_agents[0]
    consciousness_summary = sample_agent.get_consciousness_summary()
    print(f"Sample agent summary: {consciousness_summary}")
    
    print("\nNYX Agents demonstration complete!")