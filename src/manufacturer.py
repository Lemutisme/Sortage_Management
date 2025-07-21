import json
from typing import Dict, Any


from base import BaseAgent
from configs import SimulationConfig, ManufacturerState

class ManufacturerAgent(BaseAgent):
    """Pharmaceutical manufacturer agent."""
    
    def __init__(self, manufacturer_id: int, config: SimulationConfig):
        super().__init__(f"manufacturer_{manufacturer_id}", config)
        self.manufacturer_id = manufacturer_id
        self.state = ManufacturerState(
            id=manufacturer_id,
            capacity=config.initial_demand / config.n_manufacturers
        )
    
    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions from manufacturer perspective."""
        
        system_prompt = f"""
        You are a market intelligence analyst for pharmaceutical manufacturer {self.manufacturer_id} 
        in a market with {self.config.n_manufacturers} total manufacturers.
        
        Extract and analyze key decision factors from the provided context.
        """
        
        user_prompt = f"""
        Current Context:
        - Period: {context.get('period', 0)}/{self.config.n_periods}
        - Your current capacity: {self.state.capacity:.3f}
        - Disruption status: {'disrupted' if self.state.disrupted else 'operational'}
        - Recovery periods remaining: {self.state.disruption_recovery_periods}
        - FDA Alert: "{context.get('fda_announcement', 'None')}"
        - Last period demand: {context.get('last_demand', 'Unknown')}
        - Market disruptions: {len(context.get('disrupted_manufacturers', []))}/{self.config.n_manufacturers}
        - Your last production: {self.state.last_production:.3f}
        
        Analyze this information and respond with JSON containing:
        {{
            "role": "manufacturer",
            "goal": "maximize_profit_while_managing_risk",
            "market_conditions": {{
                "shortage_risk": "low/moderate/high",
                "demand_trend": "stable/increasing/volatile",
                "competitive_pressure": "low/moderate/high"
            }},
            "internal_state": {{
                "capacity_utilization": "percentage as float",
                "expansion_feasibility": "yes/no",
                "financial_health": "strong/moderate/constrained"
            }}
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt, 
            expected_json_keys=["role", "market_conditions", "internal_state"]
        )
    
    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Make capacity investment decision."""
        
        system_prompt = f"""
        You are the CEO of pharmaceutical manufacturer {self.manufacturer_id}.
        Make a capacity investment decision based on the analyzed market state.
        
        Decision Context:
        - Current capacity: {self.state.capacity:.3f}
        - Can only expand if not disrupted: {not self.state.disrupted}
        - Investment cost: {self.config.capacity_cost} per unit
        - Expected profit: {self.config.unit_profit} per unit sold
        - Expansion takes 1 period to become effective
        """
        
        user_prompt = f"""
        Market Analysis: {json.dumps(state_json, indent=2)}
        
        Choose your capacity investment level. Consider:
        1. Current market shortage risk
        2. Competitive dynamics
        3. Return on investment
        4. Timing of capacity coming online
        
        Respond with JSON:
        {{
            "decision": {{
                "capacity_investment": "numerical_value (0 for no investment)",
                "investment_percentage": "percentage_of_current_capacity"
            }},
            "reasoning": {{
                "market_analysis": "key market factors driving decision",
                "competitive_strategy": "positioning vs competitors", 
                "risk_assessment": "main risks and mitigation"
            }},
            "confidence": "low/moderate/high"
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt,
            expected_json_keys=["decision", "reasoning"]
        )
    
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