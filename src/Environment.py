import random
import logging
from typing import Dict, List, Tuple, Any

from buyer import BuyerAgent
from fda import FDAAgent
from manufacturer import ManufacturerAgent
from configs import SimulationConfig, DisruptionEvent


# =============================================================================
# Environment and Coordination
# =============================================================================

class Environment:
    """Market environment managing supply-demand dynamics."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.current_period = 0
        self.disruptions = []
        self.market_history = []
        
        # Initialize manufacturers
        self.manufacturers = [
            ManufacturerAgent(i, config) 
            for i in range(config.n_manufacturers)
        ]
        
        # Initialize buyer and FDA
        self.buyer = BuyerAgent(config)
        self.fda = FDAAgent(config)
        
        self.logger = logging.getLogger("Environment")
    
    def generate_disruptions(self, force_disruption: bool = False) -> List[DisruptionEvent]:
        """Generate random disruptions for the current period."""
        new_disruptions = []

        for manufacturer_id in range(self.config.n_manufacturers):
            if random.random() < self.config.disruption_probability:
                duration = random.randint(1, 4)
                disruption = DisruptionEvent(
                    manufacturer_id=manufacturer_id,
                    start_period=self.current_period,
                    duration=duration,
                    magnitude=self.config.disruption_magnitude,
                    remaining_periods=duration
                )
                new_disruptions.append(disruption)
                self.logger.info(f"New disruption: Manufacturer {manufacturer_id}, duration {duration}")

        if self.config.n_disruptions_if_forced_disruption > self.config.n_manufacturers:
            raise ValueError(f"Cannot create {self.config.n_disruptions_if_forced_disruption} unique disruptions with only {self.config.n_manufacturers} manufacturers.")

        if force_disruption:
            # duration = random.randint(1, 4)
            duration = self.config.n_periods
            manufacturers_disrupted = random.sample(range(self.config.n_manufacturers), k = self.config.n_disruptions_if_forced_disruption)
            for mfr_disrupted in manufacturers_disrupted:
                disruption = DisruptionEvent(
                    # manufacturer_id=random.choice(range(self.config.n_manufacturers)),
                    manufacturer_id=mfr_disrupted,
                    start_period=self.current_period,
                    duration=duration,
                    magnitude=self.config.disruption_magnitude,
                    remaining_periods=duration
                )
                new_disruptions.append(disruption)
                self.logger.info(f"Forced disruption: Manufacturer {disruption.manufacturer_id}, duration {duration}")
            
        return new_disruptions
    
    def update_disruptions(self, force_disruption: bool = False):
        """Update ongoing disruptions and apply to manufacturers."""
        # Generate new disruptions
        new_disruptions = self.generate_disruptions(force_disruption)
        self.disruptions.extend(new_disruptions)
        
        # Update existing disruptions
        active_disruptions = []
        for disruption in self.disruptions:
            if disruption.remaining_periods > 0:
                disruption.remaining_periods -= 1
                active_disruptions.append(disruption)
        
        self.disruptions = active_disruptions
        
        # Apply disruptions to manufacturers
        disrupted_ids = set()
        for disruption in self.disruptions:
            if disruption.remaining_periods >= 0:
                manufacturer = self.manufacturers[disruption.manufacturer_id]
                
                # Apply capacity reduction if disruption just started
                if disruption.remaining_periods == disruption.duration - 1:
                    new_capacity = manufacturer.state.capacity * (1 - disruption.magnitude)
                    manufacturer.update_capacity(new_capacity)
                
                manufacturer.set_disruption(True, disruption.remaining_periods)
                disrupted_ids.add(disruption.manufacturer_id)
        
        # Clear disruption status for recovered manufacturers
        for manufacturer in self.manufacturers:
            if manufacturer.manufacturer_id not in disrupted_ids:
                manufacturer.set_disruption(False, 0)
    
    def calculate_market_outcome(self, demand: float) -> Tuple[List[float], float, float, float]:
        """Calculate supply allocation and market clearing."""
        n = self.config.n_manufacturers
        
        # Get current capacities
        capacities = [m.state.capacity for m in self.manufacturers]
        disrupted_ids = [m.manufacturer_id for m in self.manufacturers if m.state.disrupted]
        
        # Initial equal allocation
        initial_allocation = demand / n
        productions = []
        
        # Calculate production for disrupted manufacturers
        unfilled_demand = 0
        for i, manufacturer in enumerate(self.manufacturers):
            if manufacturer.state.disrupted:
                production = min(capacities[i], initial_allocation)
                productions.append(production)
                unfilled_demand += initial_allocation - production
            else:
                productions.append(0)  # Placeholder, will be updated
        
        # Redistribute unfilled demand to undisrupted manufacturers
        undisrupted_count = n - len(disrupted_ids)
        additional_allocation = unfilled_demand / undisrupted_count if undisrupted_count > 0 else 0
        
        for i, manufacturer in enumerate(self.manufacturers):
            if not manufacturer.state.disrupted:
                total_allocation = initial_allocation + additional_allocation
                production = min(capacities[i], total_allocation)
                productions[i] = production
        
        total_supply = sum(productions)
        shortage = max(0, demand - total_supply)
        
        inv_change = total_supply - self.config.initial_demand
        self.buyer.update_inventory(inv_change)
        
        return productions, total_supply, shortage, inv_change
    
    def apply_investments(self):
        """Apply capacity investments from previous period."""
        for manufacturer in self.manufacturers:
            if len(manufacturer.state.investment_history) > 0:
                # Investment from previous period becomes effective
                if len(manufacturer.state.investment_history) >= 1:
                    last_investment = manufacturer.state.investment_history[-1]
                    new_capacity = manufacturer.state.capacity + last_investment
                    manufacturer.update_capacity(new_capacity)
    
    def create_context(self, agent_type: str) -> Dict[str, Any]:
        """Create context information for agents."""
        # Get last period market state
        last_market = self.market_history[-1] if self.market_history else None
        
        base_context = {
            "period": self.current_period,
            "disrupted_manufacturers": [m.manufacturer_id for m in self.manufacturers if m.state.disrupted],
            "n_manufacturers": self.config.n_manufacturers,
            "n_periods": self.config.n_periods
        }
        
        if last_market:
            base_context.update({
                "last_demand": last_market.total_demand,
                "last_supply": last_market.total_supply,
                "last_shortage": last_market.shortage_amount,
                "fda_announcement": last_market.fda_announcement
            })
        
        return base_context