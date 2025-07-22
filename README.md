# Drug Shortage Multi-Agent Simulation Framework

A comprehensive LLM-based multi-agent simulation system for modeling pharmaceutical supply chain dynamics and evaluating FDA policy interventions during drug shortages.

## 🎯 Overview

This framework simulates the complex interactions between manufacturers, healthcare buyers, and FDA regulators during drug shortage events. Each agent uses Large Language Models (LLMs) to make realistic decisions based on market conditions, regulatory signals, and strategic objectives.

### Key Features

* **🤖 LLM-Powered Agents** : Manufacturers, buyers, and FDA regulators with realistic decision-making
* **📊 Market Dynamics** : Supply-demand allocation, capacity investments, and disruption modeling
* **🏛️ Policy Testing** : Compare reactive vs proactive FDA intervention strategies
* **📈 Comprehensive Evaluation** : Track shortage resolution, costs, and stakeholder outcomes
* **🔧 Modular Design** : Easily customizable agent behaviors and prompt templates

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Lemutisme/Sortage_Management.git
cd Sortage_Management
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from simulator import SimulationCoordinator
from configs import SimulationConfig

# Configure simulation
config = SimulationConfig(
    n_manufacturers=4,
    n_periods=4, 
    disruption_probability=0.05,
    api_key="your-openai-api-key"  # or set environment variable
)

# Run simulation
async def main():
    coordinator = SimulationCoordinator(config)
    results = await coordinator.run_simulation()
    print(f"Peak shortage: {results['summary_metrics']['peak_shortage_percentage']:.1%}")

asyncio.run(main())
```

## 📁 Repository Structure

```
drug-shortage-simulation/
├── src/
│   ├── simulator.py           # Main simulation coordinator
│   ├── base.py               # Base agent framework  
│   ├── manufacturer.py       # Manufacturer agent implementation
│   ├── buyer.py             # Buyer consortium agent
│   ├── fda.py               # FDA regulatory agent
│   ├── Environment.py       # Market environment and coordination
│   ├── configs.py           # Configuration and data structures
│   ├── prompts.py           # Modular prompt template system
│   └── main.py              # Example usage and testing
├── prompts/
│   ├── README.md            # Detailed prompt documentation
│   └── templates/           # Individual prompt template files
├── data/
│   ├── FDA_final.csv        # Historical FDA shortage data
│   └── ASHP_Detailed_data.csv # ASHP narrative reports
├── experiments/             # Example experimental scenarios
├── tests/                   # Unit tests and validation
├── requirements.txt
└── README.md
```

## 🎭 Agent Architecture

Each agent follows a two-stage decision pipeline:

### Stage 1: Collector & Analyst

* **Input** : Raw market context and agent state
* **Process** : Extract structured state variables using LLM
* **Output** : JSON with market conditions and internal state

### Stage 2: Decision Maker

* **Input** : Structured state analysis from Stage 1
* **Process** : Make strategic decisions using LLM reasoning
* **Output** : JSON with decision and detailed reasoning

### Agent Types

| Agent                  | Role                            | Objective                           | Key Decisions              |
| ---------------------- | ------------------------------- | ----------------------------------- | -------------------------- |
| **Manufacturer** | Pharmaceutical company CEO      | Maximize profit while managing risk | Capacity investment levels |
| **Buyer**        | Healthcare consortium CPO       | Ensure availability, minimize costs | Purchase quantities        |
| **FDA**          | Regulatory shortage coordinator | Minimize patient impact             | Public announcement timing |

## 🔧 LLM Integration

### Supported Providers

The framework supports multiple LLM providers:

```python
# OpenAI (recommended for JSON reliability)
config = SimulationConfig(
    llm_model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Anthropic Claude
config = SimulationConfig(
    llm_model="claude-3-sonnet-20240229", 
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Azure OpenAI
config = SimulationConfig(
    llm_model="gpt-4",
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint="https://your-resource.openai.azure.com/"
)
```

### Environment Variables

For security, set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export AZURE_API_KEY="your-key-here"
```

## 📝 Prompt Customization

### Using the Modular Prompt System

```python
from prompts import PromptManager

# Load prompt templates
pm = PromptManager()

# Get formatted prompts for any agent
system_prompt, user_prompt, expected_keys = pm.get_prompt(
    agent_type="manufacturer",  # or "buyer", "fda"
    stage="collector_analyst",  # or "decision_maker"
    **context_variables
)
```

### Customizing Agent Behavior

Create variant prompt templates for different experimental conditions:

```python
# Risk-averse manufacturer
def _manufacturer_conservative_prompts():
    return """
    You are a conservative pharmaceutical manufacturer.
    Prioritize risk management over aggressive expansion.
    Only invest in capacity when shortage risk is confirmed and high.
    """

# Aggressive buyer 
def _buyer_aggressive_prompts():
    return """
    You are a proactive healthcare buyer.
    Build significant safety stock when any shortage signals appear.
    Patient safety is absolute priority regardless of cost.
    """
```

### Prompt Template Structure

All prompts follow this consistent structure:

```python
{
    "system_template": "Role definition and objectives",
    "user_template": "Context + decision framework + JSON schema", 
    "expected_keys": ["required", "json", "keys"]
}
```

## 🧪 Experimental Scenarios

### Baseline Experiments

```python
# Test different market structures
configs = [
    SimulationConfig(n_manufacturers=3),  # Concentrated market
    SimulationConfig(n_manufacturers=5),  # Competitive market
]

# Test disruption sensitivity  
configs = [
    SimulationConfig(disruption_probability=0.01),  # Low disruption
    SimulationConfig(disruption_probability=0.10),  # High disruption
]
```

### Policy Comparisons

```python
# Compare FDA intervention strategies
results_reactive = await run_simulation(fda_mode="reactive")
results_proactive = await run_simulation(fda_mode="proactive")

# Analyze policy effectiveness
compare_resolution_times(results_reactive, results_proactive)
```

## 📊 Evaluation Metrics

### Key Performance Indicators

* **Shortage Resolution** : Time from onset to market clearing
* **Peak Shortage Severity** : Maximum unmet demand percentage
* **Total Economic Cost** : Buyer costs + manufacturer profits
* **Policy Effectiveness** : Impact of FDA interventions

### Analysis Functions

```python
from analysis import analyze_results, compare_scenarios

# Convert results to DataFrame
df = analyze_results(simulation_results)

# Generate summary metrics
metrics = {
    "avg_shortage_duration": df['shortage_periods'].mean(),
    "peak_shortage": df['shortage_pct'].max(), 
    "total_cost": simulation_results['buyer_total_cost']
}
```

## 🔬 Research Applications

### Validation Against Historical Data

```python
# Load historical shortage events
historical_events = load_fda_shortage_data("data/FDA_final.csv")

# Run simulation with historical parameters
for event in historical_events:
    config = create_config_from_event(event)
    simulated_outcome = await run_simulation(config)
  
    # Compare simulated vs actual resolution time
    validate_prediction(simulated_outcome, event.actual_resolution)
```

### Policy Counterfactuals

```python
# Test "what if" scenarios
scenarios = [
    {"name": "No FDA Intervention", "fda_mode": "silent"},
    {"name": "Early Warning", "fda_mode": "proactive"},  
    {"name": "Reactive Only", "fda_mode": "reactive"}
]

# Compare outcomes across policy regimes
results = {}
for scenario in scenarios:
    results[scenario["name"]] = await run_simulation(**scenario)
  
analyze_policy_impact(results)
```

## 🤝 Contributing

### Adding New Agent Types

1. Inherit from `BaseAgent`
2. Implement `collect_and_analyze()` and `decide()` methods
3. Add prompt templates to `prompts.py`
4. Update configuration options

### Extending Market Dynamics

1. Modify `Environment.calculate_market_outcome()`
2. Add new state variables to data structures
3. Update agent context creation
4. Test with existing scenarios

### Research Datasets

We welcome contributions of:

* Historical drug shortage datasets
* Manufacturer capacity/investment data
* FDA announcement archives
* Validation benchmarks

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{Sortage_Management,
  title={Drug Shortage Multi-Agent Simulation Framework},
  author={Clause DZ},
  year={2025},
  url={https://github.com/your-username/drug-shortage-simulation}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## 🆘 Support

* **Documentation** : See `/prompts/README.md` for detailed prompt engineering guide
* **Issues** : Report bugs and feature requests via GitHub Issues
* **Discussions** : Join our GitHub Discussions for research collaboration

## 🔮 Roadmap

* [ ] Proactive FDA agent implementation
* [ ] Multi-drug market extensions
* [ ] International supply chain modeling
* [ ] Machine learning outcome prediction
* [ ] Real-time data integration
* [ ] Interactive visualization dashboard

---

**Built for pharmaceutical supply chain research and policy analysis.**
