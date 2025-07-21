import logging
from typing import Dict, Optional, Any
from dataclasses import asdict
import numpy as np


from Environment import Environment
from configs import SimulationConfig, MarketState

class SimulationCoordinator:
    """Main simulation coordinator managing the multi-agent system."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = Environment(config)
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SimulationCoordinator")
    
    async def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        self.logger.info(f"Starting simulation with {self.config.n_manufacturers} manufacturers, {self.config.n_periods} periods")
        
        for period in range(self.config.n_periods):
            self.environment.current_period = period
            self.logger.info(f"=== Period {period + 1}/{self.config.n_periods} ===")
            
            # Step 1: Update disruptions and apply investments
            self.environment.update_disruptions()
            self.environment.apply_investments()
            
            # Step 2: FDA makes announcement decision
            fda_context = self.environment.create_context("fda")
            fda_decision = await self.environment.fda.make_decision(fda_context)
            fda_announcement = self.environment.fda.make_announcement(fda_decision)
            
            # Step 3: Manufacturers make capacity decisions
            manufacturer_decisions = []
            for manufacturer in self.environment.manufacturers:
                context = self.environment.create_context("manufacturer")
                context["fda_announcement"] = fda_announcement
                decision = await manufacturer.make_decision(context)
                investment = manufacturer.apply_decision(decision)
                manufacturer_decisions.append({
                    "manufacturer_id": manufacturer.manufacturer_id,
                    "investment": investment,
                    "decision": decision
                })
            
            # Step 4: Buyer makes demand decision
            buyer_context = self.environment.create_context("buyer")
            buyer_context["fda_announcement"] = fda_announcement
            buyer_decision = await self.environment.buyer.make_decision(buyer_context)
            demand = buyer_decision.get("decision", {}).get("demand_quantity", self.config.initial_demand)
            
            # Step 5: Market clearing
            productions, total_supply, shortage = self.environment.calculate_market_outcome(demand)
            
            # Step 6: Record outcomes
            market_state = MarketState(
                period=period,
                total_demand=demand,
                total_supply=total_supply,
                shortage_amount=shortage,
                shortage_percentage=shortage / demand if demand > 0 else 0,
                disrupted_manufacturers=[m.manufacturer_id for m in self.environment.manufacturers if m.state.disrupted],
                fda_announcement=fda_announcement
            )
            
            self.environment.market_history.append(market_state)
            
            # Update agent records
            for i, manufacturer in enumerate(self.environment.manufacturers):
                revenue = productions[i] * self.config.unit_profit
                manufacturer.record_production(productions[i], revenue)
            
            self.environment.buyer.record_outcome(demand, total_supply)
            
            # Log period results
            self.logger.info(f"Demand: {demand:.3f}, Supply: {total_supply:.3f}, Shortage: {shortage:.3f} ({shortage/demand:.1%})")
            if fda_announcement:
                self.logger.info(f"FDA Announcement: {fda_announcement}")
        
        return self.compile_results()
    
    def compile_results(self) -> Dict[str, Any]:
        """Compile simulation results for analysis."""
        results = {
            "config": asdict(self.config),
            "market_trajectory": [asdict(state) for state in self.environment.market_history],
            "manufacturer_states": [asdict(m.state) for m in self.environment.manufacturers],
            "buyer_total_cost": self.environment.buyer.total_cost,
            "fda_announcements": self.environment.fda.announcement_history,
            "decision_history": {
                "manufacturers": [m.decision_history for m in self.environment.manufacturers],
                "buyer": self.environment.buyer.decision_history,
                "fda": self.environment.fda.decision_history
            }
        }
        
        # Calculate summary metrics
        shortages = [state.shortage_percentage for state in self.environment.market_history]
        results["summary_metrics"] = {
            "total_shortage_periods": sum(1 for s in shortages if s > 0),
            "peak_shortage_percentage": max(shortages) if shortages else 0,
            "average_shortage_percentage": np.mean(shortages) if shortages else 0,
            "time_to_resolution": self.calculate_resolution_time(),
            "total_manufacturer_profit": sum(m.state.cumulative_profit for m in self.environment.manufacturers)
        }
        
        return results
    
    def calculate_resolution_time(self) -> Optional[int]:
        """Calculate time to shortage resolution."""
        shortage_periods = [i for i, state in enumerate(self.environment.market_history) 
                          if state.shortage_percentage > 0]
        
        if not shortage_periods:
            return None
        
        # Find the last period with shortage
        last_shortage_period = max(shortage_periods)
        
        # Check if resolved by end of simulation
        if last_shortage_period == len(self.environment.market_history) - 1:
            return None  # Not resolved within simulation period
        
        return last_shortage_period + 1