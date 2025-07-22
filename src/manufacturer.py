import json
from typing import Dict, Any

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig, ManufacturerState

class ManufacturerAgent(BaseAgent):
    """Pharmaceutical manufacturer agent."""

    def __init__(self, manufacturer_id: int, config: SimulationConfig):
        super().__init__(f"manufacturer_{manufacturer_id}", config)
        self.manufacturer_id = manufacturer_id
        self.agent_type = "manufacturer"  # For prompt manager
        self.prompt_manager = PromptManager()

        self.state = ManufacturerState(
            id=manufacturer_id,
            capacity=config.initial_demand / config.n_manufacturers
        )

    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for market analysis."""

        # Format context for prompt templates
        prompt_context = {
            "manufacturer_id": self.manufacturer_id,
            "n_manufacturers": self.config.n_manufacturers,
            "period": context.get('period', 0),
            "n_periods": self.config.n_periods,
            "current_capacity": self.state.capacity,
            "disruption_status": 'disrupted' if self.state.disrupted else 'operational',
            "recovery_periods": self.state.disruption_recovery_periods,
            "fda_announcement": context.get('fda_announcement', 'None'),
            "last_demand": context.get('last_demand', 'Unknown'),
            "disrupted_count": len(context.get('disrupted_manufacturers', [])),
            "last_production": self.state.last_production
        }

        # Get formatted prompts from prompt manager
        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for capacity decisions."""

        # Format decision context for prompt templates
        decision_context = {
            "manufacturer_id": self.manufacturer_id,
            "current_capacity": self.state.capacity,
            "can_expand": not self.state.disrupted,
            "capacity_cost": self.config.capacity_cost,
            "unit_profit": self.config.unit_profit,
            "n_manufacturers": self.config.n_manufacturers,
            "state_json": json.dumps(state_json, indent=2)
        }

        # Get formatted prompts from prompt manager
        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            **decision_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: no capacity expansion."""
        return {
            "decision": {
                "capacity_investment": 0.0,
                "investment_percentage": 0.0
            },
            "reasoning": {
                "market_analysis": "Default conservative approach due to LLM failure",
                "competitive_strategy": "Maintain current position",
                "risk_assessment": "Avoid investment risk when uncertain"
            },
            "confidence": "low"
        }

    def apply_decision(self, decision: Dict[str, Any]) -> float:
        """Apply the investment decision and return investment amount."""
        if self.state.disrupted:
            return 0.0
        
        investment = decision.get("decision", {}).get("capacity_investment", 0.0)
        
        # Investment takes 1 period to become effective
        # This will be applied in the next period by the Environment
        self.state.investment_history.append(investment)
        
        return investment

    def update_capacity(self, new_capacity: float):
        """Update capacity (called by Environment)."""
        self.state.capacity = new_capacity

    def set_disruption(self, disrupted: bool, recovery_periods: int = 0):
        """Set disruption status."""
        self.state.disrupted = disrupted
        self.state.disruption_recovery_periods = recovery_periods

    def record_production(self, production: float, revenue: float):
        """Record production and financial results."""
        self.state.last_production = production
        cost = sum(self.state.investment_history[-1:]) * self.config.capacity_cost
        self.state.cumulative_profit += revenue - cost
