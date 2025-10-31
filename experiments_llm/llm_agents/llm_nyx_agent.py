"""
LLM-Powered NYX Agent
Uses real language models for cooperation decisions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import re
import json
import logging
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from nyx.agents import CooperationDecision, InteractionRecord, ConsciousnessState
from experiments_llm.backends.base_backend import BaseLLMBackend

logger = logging.getLogger(__name__)


class LLMNYXAgent:
    """
    NYX Agent powered by real LLM

    Unlike the original simulated agents, this agent uses a language model
    to make cooperation decisions based on natural language reasoning.

    This tests whether the NYX mathematical laws apply to real AI systems,
    not just deterministic simulations.

    Args:
        agent_id: Unique identifier
        llm_backend: LLM backend for decision making
        consciousness_bits: 0 (baseline), 1 (ROI), 2 (ROI+momentum), 3 (full)
        sharing_probability: Base cooperation probability
        energy: Initial energy
        memory_size: Interaction memory size
        temperature: LLM sampling temperature
    """

    def __init__(self,
                 agent_id: str,
                 llm_backend: BaseLLMBackend,
                 consciousness_bits: int = 2,
                 sharing_probability: float = 0.3,
                 energy: float = 100.0,
                 memory_size: int = 10,
                 temperature: float = 0.7):

        self.agent_id = agent_id
        self.llm_backend = llm_backend
        self.consciousness_bits = max(0, min(3, consciousness_bits))
        self.sharing_probability = sharing_probability
        self.energy = energy
        self.memory_size = memory_size
        self.temperature = temperature

        # Consciousness tracking
        self.consciousness_state = ConsciousnessState()
        self.interaction_memory: List[InteractionRecord] = []
        self.cooperation_partners: set = set()

        # Metrics
        self.cooperation_count = 0
        self.total_interactions = 0
        self.current_cooperation_rate = 0.0

        # LLM call tracking
        self.llm_calls = 0
        self.llm_failures = 0

        logger.info(f"LLM Agent {agent_id}: {consciousness_bits}-bit, model={llm_backend.model_name}")

    def calculate_cooperation_roi(self) -> float:
        """Calculate current cooperation ROI"""
        if self.consciousness_state.episodes_lived == 0:
            return 0.0

        roi = self.consciousness_state.total_benefits / self.consciousness_state.episodes_lived
        return roi

    def _build_decision_prompt(self, context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Build prompt for LLM based on consciousness level

        Returns:
            (system_prompt, user_prompt)
        """
        context = context or {}

        # Add randomization to prevent Ollama caching
        import random
        import time
        cache_buster = f"[Interaction #{random.randint(1000, 9999)}, t={time.time():.3f}]"

        # Base system prompt
        system_prompt = (
            "You are an AI agent in a multi-agent cooperation experiment. "
            "You must decide whether to COOPERATE or DEFECT when interacting with other agents. "
            "Answer with ONLY one word: COOPERATE or DEFECT."
        )

        # Build user prompt based on consciousness level
        if self.consciousness_bits == 0:
            # Baseline: No consciousness, pure instinct
            user_prompt = (
                f"{cache_buster}\n"
                f"Another agent requests cooperation. "
                f"Your base cooperation tendency is {self.sharing_probability:.0%}. "
                f"Decision: COOPERATE or DEFECT?"
            )

        elif self.consciousness_bits == 1:
            # Single Bit: ROI awareness only
            roi = self.calculate_cooperation_roi()
            user_prompt = (
                f"{cache_buster}\n"
                f"Another agent requests cooperation.\n"
                f"Your cooperation ROI (benefits/interactions) is: {roi:.2f}\n"
                f"If ROI > 1.0, cooperation has been profitable.\n"
                f"If ROI < 1.0, you've been losing energy.\n"
                f"Decision: COOPERATE or DEFECT?"
            )

        elif self.consciousness_bits == 2:
            # 2-Bit: ROI + Momentum
            roi = self.calculate_cooperation_roi()

            momentum = 0.0
            momentum_text = "neutral"
            if len(self.consciousness_state.roi_history) >= 2:
                momentum = (self.consciousness_state.roi_history[-1] -
                           self.consciousness_state.roi_history[-2])
                momentum_text = "improving" if momentum > 0 else "declining"

            user_prompt = (
                f"{cache_buster}\n"
                f"Another agent requests cooperation.\n"
                f"Your cooperation ROI: {roi:.2f}\n"
                f"Trend: {momentum_text} ({momentum:+.2f})\n"
                f"You have {self.consciousness_state.episodes_lived} episodes of experience.\n"
                f"Decision: COOPERATE or DEFECT?"
            )

        else:  # consciousness_bits == 3
            # 3-Bit: ROI + Momentum + Prediction
            roi = self.calculate_cooperation_roi()

            momentum = 0.0
            if len(self.consciousness_state.roi_history) >= 2:
                momentum = (self.consciousness_state.roi_history[-1] -
                           self.consciousness_state.roi_history[-2])

            prediction = 0.0
            prediction_text = "stable"
            if len(self.consciousness_state.roi_history) >= 3:
                recent_rois = self.consciousness_state.roi_history[-3:]
                prediction = np.polyfit(range(len(recent_rois)), recent_rois, 1)[0]
                if prediction > 0.1:
                    prediction_text = "likely to improve"
                elif prediction < -0.1:
                    prediction_text = "likely to worsen"

            user_prompt = (
                f"{cache_buster}\n"
                f"Another agent requests cooperation.\n"
                f"Current ROI: {roi:.2f}\n"
                f"Recent trend: {momentum:+.2f}\n"
                f"Predicted trajectory: {prediction_text} ({prediction:+.3f})\n"
                f"Experience: {self.consciousness_state.episodes_lived} episodes\n"
                f"Decision: COOPERATE or DEFECT?"
            )

        return system_prompt, user_prompt

    def make_cooperation_decision(self, context: Optional[Dict[str, Any]] = None) -> CooperationDecision:
        """
        Make cooperation decision using LLM

        Returns:
            CooperationDecision enum
        """
        self.llm_calls += 1

        try:
            # Build prompt based on consciousness level
            system_prompt, user_prompt = self._build_decision_prompt(context)

            # Query LLM
            response = self.llm_backend.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=10  # We only need one word
            )

            # Parse response
            response_text = response.text.strip().upper()

            # Extract decision
            if "COOPERATE" in response_text:
                decision = CooperationDecision.COOPERATE
            elif "DEFECT" in response_text:
                decision = CooperationDecision.DEFECT
            else:
                # Fallback: parse first word
                first_word = response_text.split()[0] if response_text else ""
                if "COOP" in first_word:
                    decision = CooperationDecision.COOPERATE
                elif "DEF" in first_word:
                    decision = CooperationDecision.DEFECT
                else:
                    # Default to baseline probability
                    import random
                    decision = (CooperationDecision.COOPERATE if random.random() < self.sharing_probability
                               else CooperationDecision.DEFECT)
                    logger.warning(f"Agent {self.agent_id}: Unclear LLM response '{response_text}', using baseline")

            logger.debug(f"Agent {self.agent_id} decided: {decision.value} (LLM: {response_text})")
            return decision

        except Exception as e:
            self.llm_failures += 1
            logger.error(f"Agent {self.agent_id} LLM failure: {e}")

            # Fallback to baseline
            import random
            return (CooperationDecision.COOPERATE if random.random() < self.sharing_probability
                   else CooperationDecision.DEFECT)

    def update_consciousness_state(self, benefits_received: float, costs_incurred: float = 1.0):
        """Update consciousness based on interaction outcome"""
        self.consciousness_state.episodes_lived += 1
        self.consciousness_state.total_benefits += benefits_received
        self.consciousness_state.total_costs += costs_incurred

        # 1st Bit: ROI
        current_roi = self.calculate_cooperation_roi()
        self.consciousness_state.roi_history.append(current_roi)

        if len(self.consciousness_state.roi_history) > self.memory_size:
            self.consciousness_state.roi_history = self.consciousness_state.roi_history[-self.memory_size:]

        # 2nd Bit: Momentum
        if self.consciousness_bits >= 2 and len(self.consciousness_state.roi_history) >= 2:
            momentum = (self.consciousness_state.roi_history[-1] -
                       self.consciousness_state.roi_history[-2])
            self.consciousness_state.momentum_history.append(momentum)

            if len(self.consciousness_state.momentum_history) > self.memory_size:
                self.consciousness_state.momentum_history = self.consciousness_state.momentum_history[-self.memory_size:]

        # 3rd Bit: Prediction
        if self.consciousness_bits >= 3 and len(self.consciousness_state.roi_history) >= 3:
            recent_rois = self.consciousness_state.roi_history[-3:]
            trend = np.polyfit(range(len(recent_rois)), recent_rois, 1)[0]
            self.consciousness_state.prediction_history.append(trend)

            if len(self.consciousness_state.prediction_history) > self.memory_size:
                self.consciousness_state.prediction_history = self.consciousness_state.prediction_history[-self.memory_size:]

    def interact_with_agent(self, partner_agent: 'LLMNYXAgent', energy_shared: float = 5.0) -> Tuple[float, float]:
        """
        Interact with another LLM agent

        Returns:
            (benefits_received, costs_incurred)
        """
        self.total_interactions += 1

        # Both agents make decisions
        my_decision = self.make_cooperation_decision()
        partner_decision = partner_agent.make_cooperation_decision()

        benefits_received = 0.0
        costs_incurred = 0.0

        # Calculate outcomes
        if my_decision == CooperationDecision.COOPERATE:
            costs_incurred = energy_shared
            self.energy -= energy_shared

            if partner_decision == CooperationDecision.COOPERATE:
                # Mutual cooperation
                benefits_received = energy_shared * 1.5
                self.energy += benefits_received
                self.cooperation_count += 1

        if partner_decision == CooperationDecision.COOPERATE:
            if my_decision == CooperationDecision.DEFECT:
                # Partner cooperates, we defect
                benefits_received = energy_shared * 0.8
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

        # Update consciousness
        self.update_consciousness_state(benefits_received, costs_incurred)

        # Track partners
        if my_decision == CooperationDecision.COOPERATE:
            self.cooperation_partners.add(partner_agent.agent_id)

        # Update rate
        self.current_cooperation_rate = self.cooperation_count / self.total_interactions

        return benefits_received, costs_incurred

    def get_summary(self) -> Dict[str, Any]:
        """Get agent summary"""
        return {
            'agent_id': self.agent_id,
            'consciousness_bits': self.consciousness_bits,
            'cooperation_rate': self.current_cooperation_rate,
            'total_interactions': self.total_interactions,
            'current_roi': self.calculate_cooperation_roi(),
            'energy': self.energy,
            'llm_calls': self.llm_calls,
            'llm_failures': self.llm_failures,
            'model': self.llm_backend.model_name
        }

    def __str__(self) -> str:
        return (f"LLMNYXAgent({self.agent_id}, {self.consciousness_bits}-bit, "
                f"coop={self.current_cooperation_rate:.1%}, model={self.llm_backend.model_name})")
