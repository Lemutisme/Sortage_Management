"""
Drug Shortage Multi-Agent Simulation System
==========================================

A comprehensive simulation framework for modeling pharmaceutical supply chain
dynamics with LLM-based agents representing manufacturers, buyers, and regulators.
"""

from typing import Dict, Any

import pandas as pd
import asyncio
import openai

from simulator import SimulationCoordinator
from configs import SimulationConfig

# =============================================================================
# Setup and Installation Guide
# =============================================================================

"""
To set up real LLM API calls, follow these steps:

1. Install required dependencies:
   pip install openai anthropic azure-openai pandas numpy

2. Set up your API configuration:

For OpenAI:
   config = SimulationConfig(
       llm_model="gpt-4",
       api_key="sk-your-openai-api-key-here"
   )

For Anthropic Claude:
   config = SimulationConfig(
       llm_model="claude-3-sonnet-20240229",
       api_key="your-anthropic-api-key-here"
   )

For Azure OpenAI:
   config = SimulationConfig(
       llm_model="gpt-4",
       api_key="your-azure-api-key-here",
       azure_endpoint="https://your-resource-name.openai.azure.com/"
   )

3. Update the _make_llm_call method above with your chosen provider
4. Set environment variables for security:
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   
   Then in code:
   import os
   config.api_key = os.getenv("OPENAI_API_KEY")
"""


# =============================================================================
# Example Usage and Testing
# =============================================================================

async def run_example_simulation():
    """Example of how to run the simulation."""
    
    # Configure simulation
    config = SimulationConfig(
        n_manufacturers=4,
        n_periods=4,
        disruption_probability=0.05,
        disruption_magnitude=0.2,
        llm_temperature=0.3,
        # Uncomment and set your API key:
        # api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create and run simulation
    coordinator = SimulationCoordinator(config)
    results = await coordinator.run_simulation()
    
    # Print summary
    print("\n=== Simulation Results ===")
    print(f"Peak shortage: {results['summary_metrics']['peak_shortage_percentage']:.1%}")
    print(f"Shortage periods: {results['summary_metrics']['total_shortage_periods']}/{config.n_periods}")
    print(f"Total buyer cost: {results['buyer_total_cost']:.3f}")
    print(f"Total manufacturer profit: {results['summary_metrics']['total_manufacturer_profit']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert results to DataFrame for analysis."""
    market_data = []
    
    for state in results["market_trajectory"]:
        market_data.append({
            "period": state["period"],
            "demand": state["total_demand"], 
            "supply": state["total_supply"],
            "shortage": state["shortage_amount"],
            "shortage_pct": state["shortage_percentage"],
            "disrupted_count": len(state["disrupted_manufacturers"]),
            "fda_announcement": bool(state["fda_announcement"])
        })
    
    return pd.DataFrame(market_data)


if __name__ == "__main__":
    # Run example simulation
    results = asyncio.run(run_example_simulation())
    
    # Convert to DataFrame for analysis
    df = analyze_results(results)
    print("\nMarket Trajectory:")
    print(df.to_string(index=False))