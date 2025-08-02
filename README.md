
# ShortageSim: LLM-based Multi-Agent Simulation for Drug Shortage Management

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

ShortageSim is a comprehensive multi-agent simulation framework that models pharmaceutical supply chain dynamics during drug shortage events. By leveraging Large Language Models (LLMs) to power agent decision-making, the system captures realistic responses to regulatory signals and market conditions under information asymmetry.

### 🎯 Research Motivation

Drug shortages regularly disrupt patient care and impose major costs on health systems worldwide. While the FDA issues alerts about potential shortages, the effectiveness of these interventions remains poorly understood due to:

* **Information Asymmetry** : FDA cannot observe individual manufacturers' inventory levels or buyers' procurement plans
* **Strategic Behavior** : Alerts may trigger stockpiling, potentially exacerbating shortages
* **Complex Interactions** : Multiple stakeholders with conflicting objectives

ShortageSim addresses these challenges by simulating realistic agent behaviors and evaluating policy interventions.

## 🚀 Key Features

* **🤖 LLM-Powered Agents** : Manufacturers, buyers, and FDA regulators with sophisticated decision-making
* **📊 Realistic Market Dynamics** : Supply disruptions, capacity investments, and demand allocation
* **🏛️ Policy Evaluation** : Test reactive vs proactive FDA intervention strategies
* **📈 Ground Truth Validation** : Calibrated against historical FDA shortage data
* **📝 Comprehensive Logging** : Detailed tracking of all decisions and market states
* **🔧 Modular Architecture** : Easily extensible for new agent types and behaviors

## 📦 Installation

### Setup

```bash
cd Sortage_Management

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or add your key to keys/openai.txt
```

### Quick Test

```bash
# Run setup test to verify installation
python src/test_setup.py

# Run a single example simulation
python src/main.py
```

## 🎮 Usage

### Basic Simulation

```python
import asyncio
from src.simulator import SimulationCoordinator
from src.configs import SimulationConfig

async def run_simulation():
    # Configure simulation parameters
    config = SimulationConfig(
        n_manufacturers=4,          # Number of competing manufacturers
        n_periods=4,               # Simulation horizon (quarters)
        disruption_probability=0.05,  # 5% chance of disruption per period
        disruption_magnitude=0.2,     # 20% capacity reduction when disrupted
        llm_temperature=0.3          # LLM creativity parameter
    )
  
    # Run simulation
    coordinator = SimulationCoordinator(config)
    results = await coordinator.run_simulation()
  
    # Display key metrics
    metrics = results['summary_metrics']
    print(f"Peak shortage: {metrics['peak_shortage_percentage']:.1%}")
    print(f"Resolution time: {metrics['total_shortage_periods']} periods")
    print(f"FDA interventions: {metrics['fda_intervention_rate']:.1%}")
  
    return results

# Execute
asyncio.run(run_simulation())
```

### Running Experiments

```bash
# Single simulation with disruption
python src/main.py

# Comparative study across scenarios
python src/main.py comparative

# Ground truth validation experiments
python src/main.py gt_experiment_dic    # For discontinued cases
python src/main.py gt_experiment_nodic  # For non-discontinued cases

# Policy effectiveness test
python src/main.py policy
```

## 🏗️ System Architecture

<img src="figures/system_overview.png" alt="System Architecture" width="100%">
### Core Components

1. **Environment Module** : Manages market dynamics, disruptions, and state transitions
2. **Agent System** : LLM-powered decision makers (manufacturers, buyers, FDA)
3. **Information Broker** : Controls inter-agent communication and enforces information asymmetry
4. **Simulation Controller** : Orchestrates execution and comprehensive logging

### Agent Decision Pipeline

Each agent follows a two-stage LLM pipeline:

```
Stage 1: Collector & Analyst
├── Input: Raw market context and signals
├── Process: Extract structured state via LLM
└── Output: JSON with analyzed market conditions

Stage 2: Decision Maker  
├── Input: Structured analysis from Stage 1
├── Process: Strategic decision-making via LLM
└── Output: Action + detailed reasoning
```

## 📊 Market Mechanics

### Disruption Modeling

* **Probability** : λ = 0.05 per manufacturer per period
* **Magnitude** : δ = 20% capacity reduction
* **Duration** : Uniform{1,2,3,4} periods
* **Recovery** : Gradual capacity restoration

### Supply-Demand Allocation

1. Equal initial allocation: `D_t / N` per manufacturer
2. Disrupted firms produce: `min(capacity, allocation)`
3. Unfilled demand redistributed to healthy firms
4. Market shortage calculated as: `max(0, D_t - total_supply)`

### Agent Objectives

| Agent                  | Role                  | Objective                             | Key Decisions                         |
| ---------------------- | --------------------- | ------------------------------------- | ------------------------------------- |
| **Manufacturer** | Pharmaceutical CEO    | Maximize profit while managing risk   | Capacity investment (0-30% expansion) |
| **Buyer**        | Healthcare consortium | Minimize costs (purchase + stockout)  | Order quantity adjustment             |
| **FDA**          | Regulatory agency     | Minimize shortage duration & severity | Issue public announcements            |

## 📈 Evaluation Metrics

### Primary Metrics

1. **FDA Intervention Percentage (FIP)** : Fraction of periods with FDA announcements
2. **Resolution-Lag Percentage (RLP)** : Timing accuracy vs ground truth

```
   RLP = 100 × (t_sim - t_GT) / t_GT
```

### Performance Results

| Dataset  | Avg FIP (%) | Avg RLP (%)      | Description                     |
| -------- | ----------- | ---------------- | ------------------------------- |
| FDA-Disc | 79.1        | **1.40**   | Discontinued manufacturer cases |
| FDA-NR   | 37.5        | **-22.70** | No disclosed reason cases       |

## 📁 Repository Structure

```
Sortage_Management/
├── src/
│   ├── simulator.py         # Main simulation coordinator
│   ├── base.py             # Base agent class with LLM integration
│   ├── manufacturer.py      # Manufacturer agent implementation
│   ├── buyer.py            # Buyer consortium agent
│   ├── fda.py              # FDA regulatory agent
│   ├── Environment.py      # Market environment and dynamics
│   ├── configs.py          # Configuration and data structures
│   ├── prompts.py          # LLM prompt templates
│   ├── logger.py           # Comprehensive logging system
│   ├── main.py             # Entry point and experiments
│   └── test_setup.py       # Installation verification
├── data/
│   ├── FDA_final.csv       # Historical FDA shortage database
│   ├── ASHP_Detailed_data.csv  # ASHP narrative reports
│   ├── GT_Disc.csv         # Ground truth - discontinued cases
│   └── GT_NoDisc.csv       # Ground truth - no disclosed reason
├── prompts/                # Detailed prompt documentation
├── gt_evaluation/          # Ground truth experiment results
├── analysis_exports/       # Simulation analysis outputs
├── figures/               # Visualization and diagrams
├── experiments_logs/       # Detailed experiment logs
├── requirements.txt
└── README.md
```

## 🔬 Experimental Framework

### Ground Truth Validation

The framework includes comprehensive validation against 51 historical FDA shortage events:

```python
# Load ground truth data
df = pd.read_csv("data/GT_Disc.csv")

# Run validation experiments
results = await run_gt_experiments(
    df, 
    n_simulations=3,      # Multiple runs per case
    export_dir="gt_evaluation"
)
```

### Customizing Agent Behaviors

Modify agent prompts in `src/prompts.py`:

```python
def get_manufacturer_prompts():
    return {
        "system_template": """You are the CEO of a pharmaceutical company...""",
        "user_template": """Current market state: {market_context}...""",
        "expected_keys": ["investment_decision", "reasoning"]
    }
```

## 📊 Logging and Analysis

The framework includes comprehensive logging at multiple levels:

```python
simulation_logs/
└── session_20250102_143022/
    ├── simulation_log.json      # Complete event log
    ├── market_states.json       # Period-by-period states
    ├── agent_decisions.json     # All agent decisions
    └── summary_metrics.json     # Aggregate results
```

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

* [ ] Proactive FDA agent implementation
* [ ] Multi-drug market extensions
* [ ] International supply chain modeling
* [ ] Alternative LLM backends (Claude, Llama, etc.)
* [ ] Interactive visualization dashboard

Please see our [contributing guidelines](https://claude.ai/chat/CONTRIBUTING.md) for details.

## 📚 Citation

If you use ShortageSim in your research, please cite:

```bibtex
@software{shortagesim2025,
  title={ShortageSim: LLM-based Multi-Agent Simulation for Drug Shortage Management},
  author={Your Name},
  year={2025},
  url={https://github.com/Lemutisme/Sortage_Management}
}
```

## 📄 Related Publications

* Naumov, S., Noh, I. J., & Zhao, H. (2025). Evaluating quality reward and other interventions to mitigate US drug shortages.  *Journal of Operations Management* , 71(3), 335–372.
* Qian, Cheng, et al. (2025). ModelingAgent: Bridging LLMs and Mathematical Modeling for Real-World Challenges.  *arXiv preprint arXiv:2505.15068* .

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://claude.ai/chat/LICENSE) file for details.

## 🙏 Acknowledgments

* FDA Drug Shortage Database for historical data
* ASHP for detailed shortage reports
* OpenAI for GPT API access

---

**For questions or support, please open an issue or contact the maintainers.**
