import json
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
import openai

from configs import SimulationConfig

# Import LoggerMixin if available, otherwise create a simple fallback
try:
    from logger import LoggerMixin
except ImportError:
    # Simple fallback LoggerMixin if import fails
    class LoggerMixin:
        def setup_logger(self, simulation_logger):
            self.simulation_logger = simulation_logger
            if hasattr(simulation_logger, 'get_agent_logger'):
                self.agent_logger = simulation_logger.get_agent_logger(self.agent_id)

# =============================================================================
# Base Agent Framework with Logging Integration
# =============================================================================

class BaseAgent(ABC, LoggerMixin):
    """Base class for all simulation agents with integrated logging."""
    
    def __init__(self, agent_id: str, config: SimulationConfig):
        # Initialize LoggerMixin first
        super().__init__()
        
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        self.decision_history = []
        self.reasoning_history = []
        
        # Logger will be set by simulation coordinator
        self.simulation_logger = None
        self.current_period = 0
    
    def setup_simulation_logger(self, simulation_logger):
        """Setup the simulation logger (called by coordinator)."""
        try:
            # Try to use LoggerMixin method
            self.setup_logger(simulation_logger)
            self.logger.debug(f"Simulation logger setup completed for {self.agent_id}")
        except AttributeError:
            # Fallback: set up logging manually
            self.simulation_logger = simulation_logger
            self.agent_logger = simulation_logger.get_agent_logger(self.agent_id)
            self.logger.debug(f"Manual simulation logger setup for {self.agent_id}")
        except Exception as e:
            self.logger.warning(f"Failed to setup simulation logger for {self.agent_id}: {e}")
            # Set up minimal logging
            self.simulation_logger = simulation_logger

    def log_decision(self, period: int, decision_stage: str, context: dict, result: dict, llm_calls: list = None):
        """Log agent decision with fallback."""
        try:
            if hasattr(self, 'simulation_logger') and self.simulation_logger:
                self.simulation_logger.log_agent_decision(
                    self.agent_id, self.__class__.__name__, period, 
                    decision_stage, context, result, llm_calls or []
                )
        except Exception as e:
            self.logger.warning(f"Failed to log decision: {e}")

    def log_llm_call(self, period: int, stage: str, system_prompt: str, 
                    user_prompt: str, response: str, success: bool, 
                    attempt: int = 1, error: str = None):
        """Log LLM call with fallback."""
        try:
            if hasattr(self, 'simulation_logger') and self.simulation_logger:
                self.simulation_logger.log_llm_call(
                    self.agent_id, period, stage, system_prompt, 
                    user_prompt, response, success, attempt, error
                )
        except Exception as e:
            self.logger.warning(f"Failed to log LLM call: {e}")
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main decision-making pipeline with comprehensive logging."""
        self.current_period = context.get('period', 0)
        
        try:
            self.logger.info(f"Starting decision process for period {self.current_period}")
            
            # Step 1: Collect and analyze information
            self.logger.debug(f"Context received: {json.dumps(context, indent=2)}")
            
            state_json = await self.collect_and_analyze(context)
            
            # Log analysis stage
            self.log_decision(
                period=self.current_period,
                decision_stage="collect_and_analyze", 
                context=context,
                result=state_json
            )
            
            # Step 2: Make decision based on analyzed state
            decision_json = await self.decide(state_json)
            
            # Log decision stage
            self.log_decision(
                period=self.current_period,
                decision_stage="decide",
                context={"state_analysis": state_json},
                result=decision_json
            )
            
            # Store in agent history
            decision_record = {
                'period': self.current_period,
                'context': context,
                'state_analysis': state_json,
                'final_decision': decision_json,
                'timestamp': datetime.now().isoformat()
            }
            self.decision_history.append(decision_record)
            
            self.logger.info(f"Decision completed successfully for period {self.current_period}")
            return decision_json
            
        except Exception as e:
            self.logger.error(f"Decision making failed for period {self.current_period}: {e}")
            
            # Log the failure
            if hasattr(self, 'simulation_logger'):
                self.simulation_logger.log_agent_decision(
                    agent_id=self.agent_id,
                    agent_type=self.__class__.__name__,
                    period=self.current_period,
                    decision_stage="error_fallback",
                    context=context,
                    result={"error": str(e), "using_default": True},
                    llm_calls=[]
                )
            
            return self.get_default_decision(context)
    
    @abstractmethod
    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context and extract structured state variables."""
        pass
    
    @abstractmethod
    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on analyzed state."""
        pass
    
    @abstractmethod
    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision if LLM calls fail."""
        pass
    
    async def call_llm(self, system_prompt: str, user_prompt: str, 
                      expected_json_keys: List[str] = None,
                      stage: str = "unknown") -> Dict[str, Any]:
        """Call LLM with retry logic, JSON validation, and comprehensive logging."""
        
        llm_calls_log = []
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"LLM call attempt {attempt + 1} for stage '{stage}'")
                
                # Make the actual LLM call
                response = await self._make_llm_call(system_prompt, user_prompt)
                
                
                # Log successful call
                self.log_llm_call(
                    period=self.current_period,
                    stage=stage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    success=True,
                    attempt=attempt + 1
                )
                
                llm_calls_log.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "response_length": len(response) if response else 0,
                    "stage": stage
                })

                
                # Parse and validate JSON response
                if not response or response.strip() == "":
                    raise ValueError("Empty response from LLM")
                
                # Clean the response (remove any non-JSON content)
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                response = response.strip()
                
                # Find JSON object boundaries
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON object found in response")
                
                json_content = response[start_idx:end_idx+1]

                result = json.loads(json_content)

                if expected_json_keys:
                    missing_keys = [key for key in expected_json_keys if key not in result]
                    if missing_keys:
                        self.logger.warning(f"Missing expected keys: {missing_keys}, but continuing...")
                        # Don't raise error for missing keys, just warn
                
                self.logger.info(f"LLM call successful on attempt {attempt + 1}")
                return result
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed: {e}. Response: '{response[:200]}...'"
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {error_msg}")
                
                # Log failed call
                self.log_llm_call(
                    period=self.current_period,
                    stage=stage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response if 'response' in locals() else "",
                    success=False,
                    attempt=attempt + 1,
                    error=error_msg
                )
                
                llm_calls_log.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": error_msg,
                    "stage": stage
                })
                
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"All LLM call attempts failed for stage '{stage}' due to JSON parsing")
                    raise
                    
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {error_msg}")
                
                # Log failed call
                self.log_llm_call(
                    period=self.current_period,
                    stage=stage,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response="",
                    success=False,
                    attempt=attempt + 1,
                    error=error_msg
                )
                
                llm_calls_log.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": error_msg,
                    "stage": stage
                })
                
                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"All LLM call attempts failed for stage '{stage}'")
                    raise
                    
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _make_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Actual LLM API call with enhanced error handling and debugging."""
        
        # First check if we have an API key
        if not hasattr(self.config, 'api_key') or not self.config.api_key:
            self.logger.warning("No API key configured, using mock responses")
            return self._mock_response()
        
        # OPTION 1: OpenAI API
        try:
            import openai
            
            # Check if API key looks valid
            if len(self.config.api_key.strip()) < 10:
                self.logger.warning("API key appears invalid, using mock responses")
                return self._mock_response()

            client = openai.AsyncOpenAI(api_key=self.config.api_key)
            
            # Enhanced system prompt to ensure JSON compliance
            enhanced_system_prompt = system_prompt + "\n\nYou must respond with valid JSON format only. Do not include any text before or after the JSON object."

            self.logger.debug(f"Making OpenAI API call with model: {self.config.llm_model}")

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # temperature=self.config.llm_temperature,
                # max_tokens=2000
            )

            
            if not response or not response.choices:
                self.logger.error("OpenAI API returned empty response")
                return self._mock_response()
            
            content = response.choices[0].message.content
            
            if not content or content.strip() == "":
                self.logger.error("OpenAI API returned empty content")
                return self._mock_response()

            self.logger.debug(f"OpenAI API response received: {len(content)} characters")
            return content.strip()
            
        except ImportError:
            self.logger.warning("OpenAI library not installed, using mock responses")
            return self._mock_response()
        except openai.APIConnectionError as e:
            self.logger.error(f"OpenAI API connection failed: {e}")
            return self._mock_response()
        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI API authentication failed: {e}")
            return self._mock_response()
        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI API rate limit exceeded: {e}")
            return self._mock_response()
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return self._mock_response()
    
    def _mock_response(self) -> str:
        """Fallback mock responses for testing."""
        if "manufacturer" in self.agent_id:
            return json.dumps({
                "role": "manufacturer",
                "goal": "maximize_profit_while_managing_risk",
                "market_conditions": {"shortage_risk": "moderate"},
                "decision": {"capacity_investment": 0.1},
                "reasoning": {"market_analysis": "Mock response due to LLM unavailability"},
                "confidence": "low"
            })
        elif "buyer" in self.agent_id:
            return json.dumps({
                "role": "buyer_consortium", 
                "goal": "minimize_total_cost_ensure_availability",
                "decision": {"demand_quantity": 1.1},
                "reasoning": {"supply_risk_assessment": "Mock response due to LLM unavailability"},
                "confidence": "low"
            })
        else:
            return json.dumps({
                "role": "fda_regulator",
                "decision": {"announcement_type": "none"},
                "reasoning": {"shortage_assessment": "Mock response due to LLM unavailability"},
                "confidence": "low"
            })