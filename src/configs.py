from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
# =============================================================================
# Configuration and Data Structures
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    n_manufacturers: int = 4
    n_periods: int = 4
    initial_demand: float = 1.0
    disruption_probability: float = 0.05
    disruption_magnitude: float = 0.2
    capacity_cost: float = 0.5
    unit_profit: float = 1.0
    stockout_penalty: float = 1.1

    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    max_retries: int = 3

    # API Configuration (set these based on your provider)
    api_key = open("./keys/openai.txt").read().strip()
    azure_endpoint = None  # For Azure OpenAI
    api_version = "2024-02-15-preview"  # For Azure OpenAI

@dataclass
class DisruptionEvent:
    """Represents a production disruption."""
    manufacturer_id: int
    start_period: int
    duration: int
    magnitude: float
    remaining_periods: int = None
    
    def __post_init__(self):
        if self.remaining_periods is None:
            self.remaining_periods = self.duration

@dataclass
class ManufacturerState:
    """State information for a manufacturer."""
    id: int
    capacity: float
    disrupted: bool = False
    disruption_recovery_periods: int = 0
    last_production: float = 0.0
    cumulative_profit: float = 0.0
    investment_history: List[float] = None
    
    def __post_init__(self):
        if self.investment_history is None:
            self.investment_history = []

@dataclass
class MarketState:
    """Overall market state information."""
    period: int
    total_demand: float
    total_supply: float
    shortage_amount: float
    shortage_percentage: float
    disrupted_manufacturers: List[int]
    fda_announcement: Optional[str] = None

class AnnouncementType(Enum):
    NONE = "none"
    MONITORING = "monitoring"
    ALERT = "alert"
    CRITICAL = "critical"