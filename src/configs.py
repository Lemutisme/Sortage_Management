from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import os

# =============================================================================
# Configuration and Data Structures with Safe API Key Loading
# =============================================================================

def load_api_key_safely() -> str:
    """Safely load API key from various sources."""
    
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and len(api_key.strip()) > 10:
        return api_key.strip()
    
    # Try reading from file
    key_file_paths = [
        "./keys/openai.txt",
        "../keys/openai.txt", 
        "keys/openai.txt",
        "openai.txt"
    ]
    
    for key_file in key_file_paths:
        try:
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    api_key = f.read().strip()
                if api_key and len(api_key) > 10:
                    return api_key
        except Exception as e:
            print(f"Warning: Could not read API key from {key_file}: {e}")
    
    # Return None if no key found - will use mock responses
    print("Warning: No OpenAI API key found. Using mock responses for testing.")
    print("To use real LLM calls:")
    print("1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
    print("2. Or create file: ./keys/openai.txt with your API key")
    return None

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

    # API Configuration with safe loading
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None  # For Azure OpenAI
    api_version: str = "2024-02-15-preview"  # For Azure OpenAI
    
    def __post_init__(self):
        """Load API key if not provided."""
        if self.api_key is None:
            self.api_key = load_api_key_safely()

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
    unsold: float
    shortage_percentage: float
    disrupted_manufacturers: List[int]
    fda_announcement: Optional[str] = None

class AnnouncementType(Enum):
    NONE = "none"
    MONITORING = "monitoring"
    ALERT = "alert"
    CRITICAL = "critical"