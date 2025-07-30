import json
from typing import Dict, Any

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig

class BuyerAgent(BaseAgent):
    """Healthcare buyer consortium agent with comprehensive logging."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("buyer_consortium", config)
        self.agent_type = "buyer"  # For prompt manager
        self.prompt_manager = PromptManager()

        self.demand_history = []
        self.supply_received_history = []
        self.total_cost = 0.0
        self.inventory = 0.5

    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for supply risk analysis with detailed logging."""
        
        self.logger.info("Buyer consortium starting supply risk analysis")

        # Extract key context information for logging
        fda_announcement = context.get('fda_announcement', 'None')
        disrupted_count = len(context.get('disrupted_manufacturers', []))
        
        self.logger.debug(f"Market context - FDA announcement: '{fda_announcement}', Disrupted manufacturers: {disrupted_count}")

        prompt_context = {
            "period": context.get('period', 0),
            "n_periods": self.config.n_periods,
            "initial_demand": self.config.initial_demand,
            "fda_announcement": fda_announcement,
            "last_supply": context.get('last_supply', 'Unknown'),
            "last_demand": context.get('last_demand', 'Unknown'),
            "inventory": self.inventory,
            "disrupted_count": disrupted_count,
            "n_manufacturers": self.config.n_manufacturers,
            "unit_profit": self.config.unit_profit,
            'holding_cost': self.config.holding_cost,
            "stockout_penalty": self.config.stockout_penalty
        }

        self.logger.debug(f"Prompt context prepared: {prompt_context}")

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        self.logger.debug("Calling LLM for supply risk analysis")
        
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="supply_risk_analysis"
        )
        
        # Log key risk assessments
        market_conditions = result.get('market_conditions', {})
        supply_security = market_conditions.get('supply_security', 'unknown')
        shortage_probability = market_conditions.get('shortage_probability', 'unknown')
        
        self.logger.info(
            f"Risk analysis completed - Supply security: {supply_security}, "
            f"Shortage probability: {shortage_probability}"
        )
        
        return result

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for demand decisions with detailed logging."""
        
        self.logger.info("Buyer consortium making procurement quantity decision")
        
        # Log current procurement status
        if self.demand_history:
            last_demand = self.demand_history[-1]
            last_supply = self.supply_received_history[-1] if self.supply_received_history else 0
            last_shortage = max(0, last_demand - last_supply)
            
            self.logger.debug(
                f"Previous period - Demand: {last_demand:.3f}, "
                f"Supply received: {last_supply:.3f}, "
                f"Shortage: {last_shortage:.3f}"
            )
        
        self.logger.debug(f"Total cost to date: {self.total_cost:.3f}")
        
        decision_context = {
            "unit_profit": self.config.unit_profit,
            "holding_cost": self.config.holding_cost,
            "stockout_penalty": self.config.stockout_penalty,
            "inventory": self.inventory,
            "initial_demand": self.config.initial_demand,
            "state_json": json.dumps(state_json, indent=2)
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            **decision_context
        )

        self.logger.debug("Calling LLM for procurement decision")
        
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="procurement_decision"
        )
        
        # Log decision outcome
        decision = result.get("decision", {})
        demand_quantity = float(decision.get("demand_quantity", self.config.initial_demand))
        demand_rationale = decision.get("demand_rationale", "unknown")
        confidence = result.get("confidence", "unknown")
        
        # Calculate demand multiplier
        demand_multiplier = demand_quantity / self.config.initial_demand
        
        self.logger.info(
            f"Procurement decision made - Quantity: {demand_quantity:.3f} "
            f"({demand_multiplier:.2f}x baseline), "
            f"Rationale: {demand_rationale}, "
            f"Confidence: {confidence}"
        )
        
        # Log cost implications
        expected_purchase_cost = demand_quantity * self.config.unit_profit
        self.logger.debug(f"Expected purchase cost: {expected_purchase_cost:.3f}")
        
        # Log reasoning if available
        reasoning = result.get("reasoning", {})
        if reasoning:
            self.logger.debug(f"Decision reasoning: {json.dumps(reasoning, indent=2)}")
        
        return result

    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: baseline demand with logging."""
        self.logger.warning("Using default procurement decision due to LLM failure")
        
        return {
            "decision": {
                "demand_quantity": self.config.initial_demand,
                "demand_rationale": "baseline"
            },
            "reasoning": {
                "supply_risk_assessment": "Default baseline due to LLM failure",
                "cost_benefit_analysis": "Conservative approach to avoid both stockout and excess costs",
                "patient_safety_considerations": "Maintain standard procurement to ensure basic supply",
                "market_impact_awareness": "Avoid disrupting market with abnormal demand"
            },
            "confidence": "low",
            "monitoring_plan": "Closely monitor supply levels and FDA announcements"
        }

    def record_outcome(self, demand: float, supply_received: float):
        """Record procurement outcomes with comprehensive logging."""
        
        # Record in history
        self.demand_history.append(demand)
        self.supply_received_history.append(supply_received)
        
        # Calculate costs
        purchase_cost = supply_received * self.config.unit_profit
        shortage = max(0, demand - supply_received)
        stockout_cost = shortage * self.config.stockout_penalty
        holding_cost = self.inventory * self.config.holding_cost
        total_period_cost = purchase_cost + stockout_cost + holding_cost

        old_total_cost = self.total_cost
        self.total_cost += total_period_cost
        
        # Comprehensive logging
        self.logger.info(
            f"Procurement outcome recorded - "
            f"Demand: {demand:.3f}, "
            f"Supply received: {supply_received:.3f}, "
            f"Shortage: {shortage:.3f}"
        )
        
        self.logger.info(
            f"Period costs - "
            f"Purchase: {purchase_cost:.3f}, "
            f"Stockout penalty: {stockout_cost:.3f}, "
            f"Holding cost: {holding_cost:.3f}, "
            f"Total: {total_period_cost:.3f}"
        )
        
        self.logger.info(
            f"Cumulative cost: {old_total_cost:.3f} → {self.total_cost:.3f} "
            f"(+{total_period_cost:.3f})"
        )
        
        # Log performance metrics
        if demand > 0:
            fill_rate = supply_received / demand
            self.logger.debug(f"Fill rate: {fill_rate:.1%}")
            
            if shortage > 0:
                shortage_percentage = shortage / demand
                self.logger.warning(f"Shortage occurred: {shortage_percentage:.1%} of demand unmet")
        
        # Log cost efficiency
        if supply_received > 0:
            cost_per_unit = total_period_cost / supply_received
            self.logger.debug(f"Effective cost per unit received: {cost_per_unit:.3f}")
    
    def update_inventory(self, inv_change: float):
        """Update capacity (called by Environment) with logging."""
        old_inventory = self.inventory
        self.inventory = max(0, self.inventory + inv_change)  # Ensure inventory doesn't go negative
    
        self.logger.info(
            f"Inventory updated - From {old_inventory:.3f} to {self.inventory:.3f} "
            f"(change: {inv_change:+.3f})"
        )


    def get_procurement_summary(self) -> Dict[str, Any]:
        """Get procurement performance summary for logging and analysis."""
        
        if not self.demand_history:
            return {"status": "no_procurement_history"}
        
        total_demand = sum(self.demand_history)
        total_supply = sum(self.supply_received_history)
        total_shortage = max(0, total_demand - total_supply)
        
        fill_rate = total_supply / total_demand if total_demand > 0 else 0
        average_demand = total_demand / len(self.demand_history)
        
        # Cost breakdown
        total_purchase_cost = total_supply * self.config.unit_profit
        total_stockout_cost = total_shortage * self.config.stockout_penalty
        total_holding_cost = self.total_cost - total_purchase_cost - total_stockout_cost
        
        summary = {
            "total_periods": len(self.demand_history),
            "total_demand": total_demand,
            "total_supply_received": total_supply,
            "total_shortage": total_shortage,
            "overall_fill_rate": fill_rate,
            "average_demand_per_period": average_demand,
            "total_cost": self.total_cost,
            "total_purchase_cost": total_purchase_cost,
            "total_holding_cost": total_holding_cost,
            "total_stockout_cost": total_stockout_cost,
            "periods_with_shortage": sum(1 for i, d in enumerate(self.demand_history) 
                                       if i < len(self.supply_received_history) and 
                                       d > self.supply_received_history[i])
        }
        
        self.logger.debug(f"Procurement summary: {json.dumps(summary, indent=2)}")
        return summary

    def get_demand_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze demand patterns for strategic insights."""
        
        if len(self.demand_history) < 2:
            return {"status": "insufficient_data"}
        
        baseline = self.config.initial_demand
        demand_multipliers = [d / baseline for d in self.demand_history]
        
        analysis = {
            "baseline_demand": baseline,
            "demand_history": self.demand_history,
            "demand_multipliers": demand_multipliers,
            "max_demand": max(self.demand_history),
            "min_demand": min(self.demand_history),
            "average_multiplier": sum(demand_multipliers) / len(demand_multipliers),
            "periods_above_baseline": sum(1 for m in demand_multipliers if m > 1.0),
            "periods_stockpiling": sum(1 for m in demand_multipliers if m > 1.2)  # 20%+ increase
        }
        
        return analysis