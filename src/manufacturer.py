import json
from typing import Dict, Any

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig, ManufacturerState

class ManufacturerAgent(BaseAgent):
    """Pharmaceutical manufacturer agent with comprehensive logging."""

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
        """Use modular prompt system for market analysis with detailed logging."""
        
        self.logger.info(f"Manufacturer {self.manufacturer_id} starting market analysis")
        
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
            "last_production": self.state.last_production,
            "baseline_production": self.config.initial_demand / self.config.n_manufacturers
        }

        self.logger.debug(f"Prompt context prepared: {prompt_context}")

        # Get formatted prompts from prompt manager
        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        self.logger.debug(f"Calling LLM for market analysis")
        
        # Call LLM with logging
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="market_analysis"
        )
        
        self.logger.info(f"Market analysis completed. Risk assessment: {result.get('market_conditions', {}).get('shortage_risk', 'unknown')}")
        
        return result

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for capacity decisions with detailed logging."""
        
        self.logger.info(f"Manufacturer {self.manufacturer_id} making capacity investment decision")
        
        # Log current state before decision
        self.logger.debug(f"Current state - Capacity: {self.state.capacity:.3f}, Disrupted: {self.state.disrupted}")
        self.logger.debug(f"Investment history: {self.state.investment_history}")

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

        self.logger.debug(f"Calling LLM for investment decision")
        
        # Call LLM with logging
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="capacity_investment_decision"
        )
        
        # Log decision outcome
        decision = result.get("decision", {})
        investment = float(decision.get("capacity_investment", 0.0))
        investment_pct = decision.get("investment_percentage", "0").replace("%","")
        confidence = result.get("confidence", "unknown")
        
        self.logger.info(
            f"Investment decision made - Amount: {investment:.3f}, "
            f"Percentage: {investment_pct}%, "
            f"Confidence: {confidence}"
        )
        
        # Log reasoning if available
        reasoning = result.get("reasoning", {})
        if reasoning:
            self.logger.debug(f"Decision reasoning: {json.dumps(reasoning, indent=2)}")
        return result

    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: no capacity expansion with logging."""
        self.logger.warning(f"Using default decision for manufacturer {self.manufacturer_id}")
        
        return {
            "decision": {
                "capacity_investment": 0.0,
                "investment_percentage": 0.0
            },
            "reasoning": {
                "market_analysis": "Default conservative approach due to LLM failure",
                "competitive_strategy": "Maintain current position",
                "risk_assessment": "Avoid investment risk when uncertain",
                "financial_justification": "No investment to minimize losses"
            },
            "confidence": "low",
            "contingency_plan": "Monitor market signals for next period"
        }

    def apply_decision(self, decision: Dict[str, Any]) -> float:
        """Apply the investment decision and return investment amount with logging."""
        
        if self.state.disrupted:
            self.logger.info(f"Manufacturer {self.manufacturer_id} cannot invest while disrupted")
            return 0.0
        
        investment = float(decision.get("decision", {}).get("capacity_investment", 0.0))
        
        # Investment takes 1 period to become effective
        self.state.investment_history.append(investment)
        
        self.logger.info(
            f"Investment applied - Amount: {investment:.3f}, "
            f"Will be effective next period. Total investments: {len(self.state.investment_history)}"
        )
        
        return investment

    def update_capacity(self, new_capacity: float):
        """Update capacity (called by Environment) with logging."""
        old_capacity = self.state.capacity
        self.state.capacity = new_capacity
        
        change = new_capacity - old_capacity
        self.logger.info(
            f"Capacity updated - From {old_capacity:.3f} to {new_capacity:.3f} "
            f"(change: {change:+.3f})"
        )

    def set_disruption(self, disrupted: bool, recovery_periods: int = 0):
        """Set disruption status with logging."""
        old_status = self.state.disrupted
        self.state.disrupted = disrupted
        self.state.disruption_recovery_periods = recovery_periods
        
        if disrupted and not old_status:
            self.logger.warning(
                f"Manufacturer {self.manufacturer_id} disrupted! "
                f"Recovery periods: {recovery_periods}"
            )
        elif not disrupted and old_status:
            self.logger.info(f"Manufacturer {self.manufacturer_id} recovered from disruption")

    def record_production(self, production: float, revenue: float):
        """Record production and financial results with logging."""
        old_production = self.state.last_production
        old_profit = self.state.cumulative_profit
        
        self.state.last_production = production
        
        # Calculate costs from recent investments
        recent_investment_cost = 0.0
        if self.state.investment_history:
            recent_investment_cost = self.state.investment_history[-1] * self.config.capacity_cost
        
        profit = revenue - recent_investment_cost
        self.state.cumulative_profit += profit
        
        self.logger.info(
            f"Production recorded - Quantity: {production:.3f} "
            f"(previous: {old_production:.3f}), Revenue: {revenue:.3f}, "
            f"Investment cost: {recent_investment_cost:.3f}, "
            f"Period profit: {profit:.3f}, "
            f"Cumulative profit: {self.state.cumulative_profit:.3f}"
        )
        
        # Log capacity utilization
        if self.state.capacity > 0:
            utilization = production / self.state.capacity
            self.logger.debug(f"Capacity utilization: {utilization:.1%}")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for logging and analysis."""
        return {
            "manufacturer_id": self.manufacturer_id,
            "capacity": self.state.capacity,
            "disrupted": self.state.disrupted,
            "recovery_periods": self.state.disruption_recovery_periods,
            "last_production": self.state.last_production,
            "cumulative_profit": self.state.cumulative_profit,
            "total_investments": len(self.state.investment_history),
            "total_invested_amount": sum(self.state.investment_history)
        }
