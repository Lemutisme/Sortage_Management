import json
from typing import Dict, Any

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig

class BuyerAgent(BaseAgent):
    """Healthcare buyer consortium agent."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("buyer_consortium", config)
        self.agent_type = "buyer"  # For prompt manager
        self.prompt_manager = PromptManager()

        self.demand_history = []
        self.supply_received_history = []
        self.total_cost = 0.0

    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for supply risk analysis."""

        prompt_context = {
            "period": context.get('period', 0),
            "n_periods": self.config.n_periods,
            "initial_demand": self.config.initial_demand,
            "fda_announcement": context.get('fda_announcement', 'None'),
            "last_supply": context.get('last_supply', 'Unknown'),
            "last_demand": context.get('last_demand', 'Unknown'),
            "disrupted_count": len(context.get('disrupted_manufacturers', [])),
            "n_manufacturers": self.config.n_manufacturers,
            "unit_profit": self.config.unit_profit,
            "stockout_penalty": self.config.stockout_penalty
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for demand decisions."""
        
        decision_context = {
            "unit_profit": self.config.unit_profit,
            "stockout_penalty": self.config.stockout_penalty,
            "initial_demand": self.config.initial_demand,
            "state_json": json.dumps(state_json, indent=2)
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            **decision_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    # Keep existing methods unchanged
    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: baseline demand."""
        return {
            "decision": {
                "demand_quantity": self.config.initial_demand,
                "demand_rationale": "baseline"
            },
            "reasoning": {
                "supply_risk_assessment": "Default baseline due to LLM failure",
                "cost_benefit_analysis": "Conservative approach",
                "patient_safety_considerations": "Maintain standard procurement"
            },
            "confidence": "low"
        }

    def record_outcome(self, demand: float, supply_received: float):
        """Record procurement outcomes."""
        self.demand_history.append(demand)
        self.supply_received_history.append(supply_received)
        
        # Calculate costs
        purchase_cost = supply_received * self.config.unit_profit
        stockout_cost = max(0, demand - supply_received) * self.config.stockout_penalty
        self.total_cost += purchase_cost + stockout_cost