# NYX API Documentation

## Core Classes

### NYXCooperationSystem

The main class for predicting AI cooperation using the NYX mathematical framework.

```python
from nyx import NYXCooperationSystem, NYXAgent

# Create agents
agents = [NYXAgent(f"agent_{i}") for i in range(4)]

# Initialize system
system = NYXCooperationSystem(agents, consciousness_bits=2, memory_size=10)

# Predict cooperation
cooperation_rate = system.calculate_cooperation_rate()
print(f"Predicted cooperation: {cooperation_rate:.1%}")
```

#### Methods

- `calculate_cooperation_rate() -> float`: Calculate cooperation using C = 0.1×N + 0.1×M + 0.8×A
- `get_formula_breakdown() -> Dict[str, float]`: Get detailed component breakdown
- `validate_prediction_accuracy(observed: float) -> float`: Validate against observed data

### NYXAgent

Consciousness-enabled cooperative agent supporting 1-3 consciousness bits.

```python
# Single Bit Theory demonstration
single_bit_agent = NYXAgent("agent_1", consciousness_bits=1)

# Optimal configuration  
optimal_agent = NYXAgent("agent_2", consciousness_bits=2)
```

#### Methods

- `make_cooperation_decision() -> CooperationDecision`: Make cooperation choice
- `update_consciousness_state(benefits, costs)`: Update awareness state
- `get_consciousness_summary() -> Dict`: Get consciousness metrics

## Quick Functions

### predict_cooperation()

Quick cooperation prediction without full system setup.

```python
from nyx import predict_cooperation

# Quick predictions
cooperation_4_agents = predict_cooperation(4)  # 71.7%
cooperation_8_agents = predict_cooperation(8)  # 85%+
```

## Constants

- `NYX_FORMULA_ACCURACY = 0.903` (90.3%)
- `MINIMUM_VIABLE_SOCIETY = 4` agents
- `OPTIMAL_CONSCIOUSNESS_BITS = 2`
- `OPTIMAL_MEMORY_SIZE = 10`