from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import os

# =============================================================================
# Configuration and Data Structures with Safe API Key Loading
# =============================================================================

def load_api_key_safely(env_name: str, file_candidates: List[str], provider_label: str) -> Optional[str]:
    """Safely load API key for a provider from env or files."""

    api_key = os.getenv(env_name)
    if api_key and len(api_key.strip()) > 10:
        return api_key.strip()

    for key_file in file_candidates:
        try:
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    api_key = f.read().strip()
                if api_key and len(api_key) > 10:
                    return api_key
        except Exception as e:
            print(f"Warning: Could not read {provider_label} API key from {key_file}: {e}")

    print(f"Warning: No {provider_label} API key found. Using mock responses for testing.")
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
    holding_cost: float = 0.1
    stockout_penalty: float = 1.1
    n_disruptions_if_forced_disruption: int = 1
    fda_mode: str = "reactive"  # proactive or reactive
    market_share: List[float] = None  # If None, equal shared by default

    # LLM Configuration
    llm_provider: str = "openai"  # one of: openai, anthropic, gemini, deepseek
    llm_model: str = "gpt-4o"
    # llm_model: str = "o3"
    llm_temperature: float = 0.3
    max_retries: int = 3

    # API Configuration with safe loading
    api_key: Optional[str] = None  # OpenAI default
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None  # For Azure OpenAI
    api_version: str = "2024-02-15-preview"  # For Azure OpenAI
    
    def __post_init__(self):
        """Load API keys if not provided."""
        if self.api_key is None:
            self.api_key = load_api_key_safely(
                env_name="OPENAI_API_KEY",
                file_candidates=[
                    "./keys/openai.txt",
                    "../keys/openai.txt",
                    "keys/openai.txt",
                    "openai.txt",
                ],
                provider_label="OpenAI",
            )

        if self.anthropic_api_key is None:
            self.anthropic_api_key = load_api_key_safely(
                env_name="ANTHROPIC_API_KEY",
                file_candidates=[
                    "./keys/anthropic.txt",
                    "../keys/anthropic.txt",
                    "keys/anthropic.txt",
                    "anthropic.txt",
                ],
                provider_label="Anthropic",
            )

        if self.gemini_api_key is None:
            self.gemini_api_key = load_api_key_safely(
                env_name="GEMINI_API_KEY",
                file_candidates=[
                    "./keys/gemini.txt",
                    "../keys/gemini.txt",
                    "keys/gemini.txt",
                    "gemini.txt",
                ],
                provider_label="Gemini",
            )

        if self.deepseek_api_key is None:
            self.deepseek_api_key = load_api_key_safely(
                env_name="DEEPSEEK_API_KEY",
                file_candidates=[
                    "./keys/deepseek.txt",
                    "../keys/deepseek.txt",
                    "keys/deepseek.txt",
                    "deepseek.txt",
                ],
                provider_label="DeepSeek",
            )

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
