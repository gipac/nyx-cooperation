# NYX Tutorials

## Getting Started

### 1. Quick Start Tutorial

Learn NYX basics in 5 minutes.

```python
# Install NYX
pip install nyx-cooperation

# Basic usage
from nyx import predict_cooperation

# Predict cooperation for different scenarios
print(f"2 agents: {predict_cooperation(2):.1%}")  # 0%
print(f"4 agents: {predict_cooperation(4):.1%}")  # 71.7%
print(f"8 agents: {predict_cooperation(8):.1%}")  # 85%+
```

### 2. Single Bit Theory Tutorial

Understand how one bit of consciousness transforms cooperation.

```python
from nyx.agents import SingleBitAgent, create_agent_population, run_agent_interaction_cycle

# Create single-bit agents
agents = create_agent_population(4, "single")

# Run interaction cycles
stats = run_agent_interaction_cycle(agents, cycles=100)
print(f"Single-bit cooperation: {stats['cooperation_rate']:.1%}")
# Expected: ~67.5%
```

### 3. 80/20 Law Tutorial

Explore the consciousness vs infrastructure discovery.

```python
from nyx import NYXCooperationSystem, NYXAgent

agents = [NYXAgent(f"agent_{i}") for i in range(4)]
system = NYXCooperationSystem(agents)

# Get formula breakdown
breakdown = system.get_formula_breakdown()
print(f"Consciousness: {breakdown['80_20_validation']['consciousness_percentage']:.0f}%")
print(f"Infrastructure: {breakdown['80_20_validation']['infrastructure_percentage']:.0f}%")
# Expected: 80% consciousness, 20% infrastructure
```

### 4. Multi-Agent Experiments Tutorial

Run your own cooperation experiments.

```python
from nyx.agents import OptimalAgent, run_agent_interaction_cycle

# Create optimal agents (2-bit consciousness)
agents = [OptimalAgent(f"optimal_{i}") for i in range(6)]

# Run extended experiment
stats = run_agent_interaction_cycle(agents, cycles=200, interaction_probability=0.4)

print(f"Cooperation rate: {stats['cooperation_rate']:.1%}")
print(f"Average ROI: {stats['average_roi']:.3f}")
print(f"Total interactions: {stats['total_interactions']}")
```

### 5. Production Deployment Tutorial

Deploy NYX in production systems.

```python
from nyx import NYXCooperationSystem

class ProductionAI:
    def __init__(self):
        self.agents = self.load_existing_ai_agents()
        self.nyx_system = NYXCooperationSystem(self.agents)
    
    def make_cooperation_decision(self, context):
        # Get cooperation prediction
        cooperation_rate = self.nyx_system.calculate_cooperation_rate()
        
        # Apply business logic
        if cooperation_rate > 0.7:
            return self.execute_cooperative_action(context)
        else:
            return self.execute_individual_action(context)
```

## Advanced Topics

### Custom Consciousness Implementation

Extend NYX with your own consciousness mechanisms.

```python
from nyx.agents import NYXAgent

class CustomConsciousnessAgent(NYXAgent):
    def calculate_cooperation_roi(self):
        # Your custom ROI calculation
        return custom_roi_algorithm(self.interaction_memory)
```

### Performance Optimization

Optimize NYX for large-scale deployments.

```python
from nyx import NYXCooperationSystem

# Enable fast mode for >1000 agents
system = NYXCooperationSystem(agents, enable_monitoring=False)
cooperation_rate = system.calculate_cooperation_rate()  # O(n) complexity
```