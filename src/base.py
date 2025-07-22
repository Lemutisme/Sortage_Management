import json
import logging
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
import openai

from configs import SimulationConfig
# =============================================================================
# Base Agent Framework
# =============================================================================

class BaseAgent(ABC):
    """Base class for all simulation agents."""
    
    def __init__(self, agent_id: str, config: SimulationConfig):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        self.decision_history = []
        self.reasoning_history = []
    
    async def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main decision-making pipeline: collect_analyze -> decide."""
        try:
            # Step 1: Collect and analyze information
            state_json = await self.collect_and_analyze(context)
            
            # Step 2: Make decision based on analyzed state
            decision_json = await self.decide(state_json)
            
            # Log decision
            self.decision_history.append({
                'period': context.get('period', 0),
                'state': state_json,
                'decision': decision_json,
                'timestamp': datetime.now().isoformat()
            })
            
            return decision_json
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
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
                      expected_json_keys: List[str] = None) -> Dict[str, Any]:
        """Call LLM with retry logic and JSON validation."""
        for attempt in range(self.config.max_retries):
            try:
                # This is a placeholder - replace with your actual LLM API call
                response = await self._make_llm_call(system_prompt, user_prompt)
                
                # Debug: Log the raw response
                self.logger.info(f"Raw LLM response for {self.agent_id}: {response[:200]}...")
                
                # Parse and validate JSON response
                if not response or response.strip() == "":
                    raise ValueError("Empty response from LLM")
                
                result = json.loads(response)
                
                if expected_json_keys:
                    missing_keys = [key for key in expected_json_keys if key not in result]
                    if missing_keys:
                        raise ValueError(f"Missing required keys: {missing_keys}")
                
                return result
                
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _make_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Actual LLM API call - implement based on your chosen provider."""
        
        # OPTION 1: OpenAI API
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.config.api_key)
            
            # Enhanced system prompt to ensure JSON compliance
            enhanced_system_prompt = system_prompt + "\n\nYou must respond with valid JSON format only."

            response = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.llm_temperature,
                max_tokens=2000
                # Note: response_format removed for broader compatibility
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"LLM Raw Response: '{content}'")

            if not content or content.strip() == "":
                self.logger.error("LLM returned empty response")
                raise ValueError("Empty response from LLM")

            return content
            
        except ImportError:
            self.logger.warning("OpenAI library not installed, falling back to mock responses")
            return self._mock_response()
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
        
        # OPTION 2: Anthropic Claude API (uncomment to use)
        # try:
        #     import anthropic
        #     client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        #     
        #     response = await client.messages.create(
        #         model=self.config.llm_model,
        #         max_tokens=2000,
        #         temperature=self.config.llm_temperature,
        #         system=system_prompt,
        #         messages=[{"role": "user", "content": user_prompt}]
        #     )
        #     
        #     return response.content[0].text
        # 
        # except ImportError:
        #     self.logger.warning("Anthropic library not installed")
        #     return self._mock_response()
        # except Exception as e:
        #     self.logger.error(f"Anthropic API call failed: {e}")
        #     raise
        
        # OPTION 3: Azure OpenAI (uncomment to use)
        # try:
        #     from openai import AsyncAzureOpenAI
        #     client = AsyncAzureOpenAI(
        #         azure_endpoint=self.config.azure_endpoint,
        #         api_key=self.config.api_key,
        #         api_version=self.config.api_version
        #     )
        #     
        #     response = await client.chat.completions.create(
        #         model=self.config.llm_model,  # e.g., "gpt-4"
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_prompt}
        #         ],
        #         temperature=self.config.llm_temperature,
        #         max_tokens=2000
        #     )
        #     
        #     return response.choices[0].message.content
        # 
        # except ImportError:
        #     self.logger.warning("Azure OpenAI library not installed")
        #     return self._mock_response()
    
    def _mock_response(self) -> str:
        """Fallback mock responses for testing."""
        if "manufacturer" in self.agent_id:
            return json.dumps({
                "role": "manufacturer",
                "goal": "maximize_profit_while_managing_risk",
                "market_conditions": {"shortage_risk": "moderate"},
                "decision": {"capacity_investment": 0.1}
            })
        elif "buyer" in self.agent_id:
            return json.dumps({
                "role": "buyer_consortium", 
                "goal": "minimize_total_cost_ensure_availability",
                "decision": {"demand_quantity": 1.1}
            })
        else:
            return json.dumps({
                "role": "fda_regulator",
                "decision": {"announcement_type": "none"}
            })
