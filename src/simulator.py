import logging
from typing import Dict, Optional, Any
from dataclasses import asdict
import numpy as np

from Environment import Environment
from configs import SimulationConfig, MarketState
from logger import SimulationLogger

class SimulationCoordinator:
    """Main simulation coordinator managing the multi-agent system with comprehensive logging."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = Environment(config)
        self.results = []
        
        # Setup basic logging for coordinator FIRST
        self.logger = logging.getLogger("SimulationCoordinator")
        
        # Initialize comprehensive logging system
        self.simulation_logger = SimulationLogger(config)
        
        # Setup all agents with logging
        self._setup_agent_logging()
        
        self.logger.info(f"Simulation coordinator initialized with {config.n_manufacturers} manufacturers, {config.n_periods} periods")
    
    def _setup_agent_logging(self):
        """Setup logging for all agents in the environment."""
        
        try:
            # Setup manufacturer agents
            for manufacturer in self.environment.manufacturers:
                if hasattr(manufacturer, 'setup_simulation_logger'):
                    manufacturer.setup_simulation_logger(self.simulation_logger)
                    self.logger.info(f"Logging setup for manufacturer {manufacturer.manufacturer_id}")
                else:
                    # Fallback: set up logger manually
                    manufacturer.setup_logger(self.simulation_logger)
                    self.logger.info(f"Manual logging setup for manufacturer {manufacturer.manufacturer_id}")
            
            # Setup buyer agent
            if hasattr(self.environment.buyer, 'setup_simulation_logger'):
                self.environment.buyer.setup_simulation_logger(self.simulation_logger)
                self.logger.info("Logging setup for buyer consortium")
            else:
                self.environment.buyer.setup_logger(self.simulation_logger)
                self.logger.info("Manual logging setup for buyer consortium")
            
            # Setup FDA agent
            if hasattr(self.environment.fda, 'setup_simulation_logger'):
                self.environment.fda.setup_simulation_logger(self.simulation_logger)
                self.logger.info("Logging setup for FDA regulator")
            else:
                self.environment.fda.setup_logger(self.simulation_logger)
                self.logger.info("Manual logging setup for FDA regulator")
                
        except Exception as e:
            self.logger.error(f"Error setting up agent logging: {e}")
            # Continue without comprehensive logging if setup fails
            self.logger.warning("Continuing simulation with basic logging only")
    
    async def run_simulation(self, start_with_disruption=False) -> Dict[str, Any]:
        """Run the complete simulation with comprehensive logging."""
        self.logger.info(f"Starting simulation with {self.config.n_manufacturers} manufacturers, {self.config.n_periods} periods")
        
        try:
            for period in range(self.config.n_periods):
                self.environment.current_period = period
                
                # Log period start
                self.simulation_logger.log_period_start(period)
                self.logger.info(f"=== Period {period + 1}/{self.config.n_periods} ===")
                
                # Step 1: Update disruptions and apply investments
                self.logger.debug("Updating disruptions and applying investments")
                
                # Store disruption state before update
                old_disruptions = self.environment.disruptions.copy()

                force_disruption = start_with_disruption and period == 0
                self.environment.update_disruptions(force_disruption)
                self.environment.apply_investments()
                
                # Log disruption events
                new_disruptions = [d for d in self.environment.disruptions if d not in old_disruptions]
                self.simulation_logger.log_disruption_events(period, new_disruptions, self.environment.disruptions)
                # Step 2: FDA makes announcement decision
                self.logger.debug("FDA making announcement decision")
                
                fda_context = self.environment.create_context("fda")
                fda_decision = await self.environment.fda.make_decision(fda_context)

                fda_announcement = self.environment.fda.make_announcement(fda_decision)
                
                # Step 3: Manufacturers make capacity decisions
                self.logger.debug("Manufacturers making capacity decisions")
                
                manufacturer_decisions = []
                for manufacturer in self.environment.manufacturers:
                    context = self.environment.create_context("manufacturer")
                    context["fda_announcement"] = fda_announcement
                    
                    decision = await manufacturer.make_decision(context)
                    investment = manufacturer.apply_decision(decision)
                    
                    manufacturer_decisions.append({
                        "manufacturer_id": manufacturer.manufacturer_id,
                        "investment": investment,
                        "decision": decision,
                        "state_summary": manufacturer.get_state_summary()
                    })
                    
                    self.logger.debug(f"Manufacturer {manufacturer.manufacturer_id} invested {investment:.3f}")
                
                # Step 4: Buyer makes demand decision
                self.logger.debug("Buyer making demand decision")
                
                buyer_context = self.environment.create_context("buyer")
                buyer_context["fda_announcement"] = fda_announcement
                buyer_decision = await self.environment.buyer.make_decision(buyer_context)
                demand = buyer_decision.get("decision", {}).get("demand_quantity", self.config.initial_demand)
                
                # Step 5: Market clearing
                self.logger.debug("Calculating market clearing")
                
                productions, total_supply, shortage, inv_change = self.environment.calculate_market_outcome(demand)
                
                # Step 6: Record outcomes
                market_state = MarketState(
                    period=period,
                    total_demand=demand,
                    total_supply=total_supply,
                    shortage_amount=shortage,
                    unsold=max(inv_change, 0),
                    shortage_percentage=shortage / demand if demand > 0 else 0,
                    disrupted_manufacturers=[m.manufacturer_id for m in self.environment.manufacturers if m.state.disrupted],
                    fda_announcement=fda_announcement
                )
                
                self.environment.market_history.append(market_state)
                
                # Log market outcome
                manufacturer_states = [m.get_state_summary() for m in self.environment.manufacturers]
                self.simulation_logger.log_market_outcome(period, market_state, productions, manufacturer_states)
                
                # Update agent records
                for i, manufacturer in enumerate(self.environment.manufacturers):
                    revenue = productions[i] * self.config.unit_profit
                    manufacturer.record_production(productions[i], revenue)
                
                self.environment.buyer.record_outcome(demand, total_supply)
                
                # Log period summary
                self.logger.info(
                    f"Period {period + 1} completed - "
                    f"Demand: {demand:.3f}, "
                    f"Supply: {total_supply:.3f}, "
                    f"Shortage: {shortage:.3f} ({shortage/demand:.1%}), "
                    f"Unsold: {max(inv_change, 0):.3f}"
                )
                
                if fda_announcement:
                    self.logger.info(f"FDA Announcement: {fda_announcement}")
            
            # Compile final results
            results = self.compile_results()
            
            # Log simulation completion
            self.simulation_logger.log_simulation_end(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            
            # Log the error
            error_results = {"error": str(e), "partial_results": self.compile_results()}
            self.simulation_logger.log_simulation_end(error_results)
            
            raise
        
        finally:
            # Always close logger
            self.simulation_logger.close()
    
    def compile_results(self) -> Dict[str, Any]:
        """Compile simulation results for analysis with enhanced logging data."""
        
        self.logger.info("Compiling simulation results")
        
        # Get detailed agent summaries
        manufacturer_summaries = []
        for manufacturer in self.environment.manufacturers:
            summary = manufacturer.get_state_summary()
            manufacturer_summaries.append(summary)
        
        buyer_summary = self.environment.buyer.get_procurement_summary()
        fda_summary = self.environment.fda.get_announcement_history_summary()
        
        results = {
            "config": asdict(self.config),
            "market_trajectory": [asdict(state) for state in self.environment.market_history],
            "manufacturer_states": [asdict(m.state) for m in self.environment.manufacturers],
            "manufacturer_summaries": manufacturer_summaries,
            "buyer_total_cost": self.environment.buyer.total_cost,
            "buyer_summary": buyer_summary,
            "fda_announcements": self.environment.fda.announcement_history,
            "fda_summary": fda_summary,
            "decision_history": {
                "manufacturers": [m.decision_history for m in self.environment.manufacturers],
                "buyer": self.environment.buyer.decision_history,
                "fda": self.environment.fda.decision_history
            },
            "logging_session": {
                "simulation_id": self.simulation_logger.simulation_id,
                "session_directory": str(self.simulation_logger.session_dir),
                "total_events": len(self.simulation_logger.events),
                "decision_events": len(self.simulation_logger.decision_events),
                "market_events": len(self.simulation_logger.market_events)
            }
        }
        
        # Calculate summary metrics
        shortages = [state.shortage_percentage for state in self.environment.market_history]
        results["summary_metrics"] = {
            "total_shortage_periods": sum(1 for s in shortages if s > 0),
            "peak_shortage_percentage": max(shortages) if shortages else 0,
            "average_shortage_percentage": np.mean(shortages) if shortages else 0,
            "time_to_resolution": self.calculate_resolution_time(),
            "total_manufacturer_profit": sum(m.state.cumulative_profit for m in self.environment.manufacturers),
            "total_capacity_investments": sum(len(m.state.investment_history) for m in self.environment.manufacturers),
            "total_invested_amount": sum(sum(m.state.investment_history) for m in self.environment.manufacturers),
            "buyer_cost_efficiency": self._calculate_buyer_efficiency(),
            "fda_intervention_rate": len(self.environment.fda.announcement_history) / self.config.n_periods
        }
        
        self.logger.info(f"Results compiled - {len(self.simulation_logger.events)} events logged")
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
    
    def _calculate_buyer_efficiency(self) -> Dict[str, float]:
        """Calculate buyer cost efficiency metrics."""
        
        if not self.environment.buyer.demand_history:
            return {"status": "no_data"}
        
        total_demand = sum(self.environment.buyer.demand_history)
        total_supply = sum(self.environment.buyer.supply_received_history)
        
        # Calculate theoretical minimum cost (perfect procurement)
        theoretical_min_cost = total_demand * self.config.unit_profit
        actual_cost = self.environment.buyer.total_cost
        
        efficiency = theoretical_min_cost / actual_cost if actual_cost > 0 else 0
        
        return {
            "cost_efficiency": efficiency,
            "excess_cost_ratio": (actual_cost - theoretical_min_cost) / theoretical_min_cost if theoretical_min_cost > 0 else 0,
            "fill_rate": total_supply / total_demand if total_demand > 0 else 0
        }
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging summary."""
        
        return {
            "simulation_id": self.simulation_logger.simulation_id,
            "session_directory": str(self.simulation_logger.session_dir),
            "logging_stats": {
                "total_events": len(self.simulation_logger.events),
                "decision_events": len(self.simulation_logger.decision_events),
                "market_events": len(self.simulation_logger.market_events),
                "llm_calls": len([e for e in self.simulation_logger.events if e.event_type == "llm_call"])
            },
            "agent_logging": {
                "manufacturers": [f"manufacturer_{i}" for i in range(self.config.n_manufacturers)],
                "buyer": "buyer_consortium", 
                "fda": "fda_regulator"
            }
        }