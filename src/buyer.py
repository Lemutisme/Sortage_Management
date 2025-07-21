import json
from typing import Dict, Any

from base import BaseAgent
from configs import SimulationConfig

class BuyerAgent(BaseAgent):
    """Healthcare buyer consortium agent."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("buyer_consortium", config)
        self.demand_history = []
        self.supply_received_history = []
        self.total_cost = 0.0
    
    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supply situation from buyer perspective."""
        
        system_prompt = """
        You are a procurement strategist for a healthcare consortium.
        Patient safety is your top priority while minimizing total costs.
        
        Analyze the market situation and supply risks.
        """
        
        user_prompt = f"""
        Current Context:
        - Period: {context.get('period', 0)}/{self.config.n_periods}
        - Baseline demand: {self.config.initial_demand}
        - FDA Alert: "{context.get('fda_announcement', 'None')}"
        - Last period supply received: {context.get('last_supply', 'Unknown')}
        - Last period demanded: {context.get('last_demand', 'Unknown')}
        - Market disruptions: {len(context.get('disrupted_manufacturers', []))}/{self.config.n_manufacturers}
        - Purchase price: {self.config.unit_profit}
        - Stockout penalty: {self.config.stockout_penalty}
        
        Analyze and respond with JSON:
        {{
            "role": "buyer_consortium",
            "goal": "minimize_total_cost_ensure_availability",
            "market_conditions": {{
                "supply_security": "secure/at_risk/critical",
                "shortage_probability": "low/moderate/high",
                "manufacturer_reliability": "high/moderate/concerning"
            }},
            "internal_state": {{
                "stockout_tolerance": "zero/minimal/limited",
                "budget_flexibility": "high/moderate/constrained",
                "patient_demand_pressure": "stable/increasing/urgent"
            }}
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt,
            expected_json_keys=["role", "market_conditions", "internal_state"]
        )
    
    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Make demand quantity decision."""
        
        system_prompt = f"""
        You are the Chief Procurement Officer for a healthcare consortium.
        Determine optimal purchase quantities balancing costs and patient safety.
        
        Cost Structure:
        - Purchase price: {self.config.unit_profit} per unit
        - Stockout penalty: {self.config.stockout_penalty} per unmet unit
        - Baseline demand: {self.config.initial_demand}
        """
        
        user_prompt = f"""
        Market Analysis: {json.dumps(state_json, indent=2)}
        
        Choose your demand quantity. Consider:
        1. Patient safety requirements (shortages impact care)
        2. Cost minimization (balance purchase vs stockout costs)
        3. Market dynamics (your demand affects prices/availability)
        4. FDA signals and shortage probability
        
        Decision options:
        - Baseline (1.0): Normal procurement
        - Conservative buffer (1.1-1.2): Modest safety stock
        - Moderate stockpiling (1.2-1.5): Significant buffer
        - Emergency stockpiling (1.5+): Crisis response
        
        Respond with JSON:
        {{
            "decision": {{
                "demand_quantity": "numerical_value",
                "demand_rationale": "baseline/buffer/stockpile/emergency"
            }},
            "reasoning": {{
                "supply_risk_assessment": "shortage probability and impact",
                "cost_benefit_analysis": "expected costs vs benefits",
                "patient_safety_considerations": "impact on patient care"
            }},
            "confidence": "low/moderate/high"
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt,
            expected_json_keys=["decision", "reasoning"]
        )
    
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