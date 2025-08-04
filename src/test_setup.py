#!/usr/bin/env python3
"""
Setup Test Script for Drug Shortage Simulation
==============================================

This script tests the basic setup and configuration before running the full simulation.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from configs import SimulationConfig
    print("✅ Configuration module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import configuration: {e}")
    sys.exit(1)

def test_api_key_setup():
    """Test API key configuration."""
    print("\n🔑 Testing API Key Setup...")
    
    config = SimulationConfig()
    
    if config.api_key and len(config.api_key) > 10:
        print(f"✅ API key found: {config.api_key[:8]}...{config.api_key[-4:]}")
        return True
    else:
        print("⚠️  No API key found - will use mock responses")
        print("   To use real LLM calls:")
        print("   1. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("   2. Or create file: ./keys/openai.txt with your API key")
        return False

def test_openai_library():
    """Test if OpenAI library is available."""
    print("\n📦 Testing OpenAI Library...")
    
    try:
        import openai
        print(f"✅ OpenAI library available (version: {openai.__version__})")
        return True
    except ImportError:
        print("❌ OpenAI library not installed")
        print("   Install with: pip install openai")
        return False

async def test_mock_llm_call():
    """Test that mock LLM calls work."""
    print("\n🤖 Testing Mock LLM Calls...")
    
    try:
        from base import BaseAgent
        from configs import SimulationConfig
        
        # Create a simple test agent
        class TestAgent(BaseAgent):
            async def collect_and_analyze(self, context):
                return {"test": "analysis"}
            
            async def decide(self, state_json):
                return {"test": "decision"}
            
            def get_default_decision(self, context):
                return {"test": "default"}
        
        config = SimulationConfig()
        agent = TestAgent("test_agent", config)
        
        # Test mock response
        mock_response = agent._mock_response()
        print(f"✅ Mock response generated: {len(mock_response)} characters")
        
        # Test JSON parsing
        import json
        parsed = json.loads(mock_response)
        print(f"✅ Mock response is valid JSON with keys: {list(parsed.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock LLM test failed: {e}")
        return False

async def test_simulation_setup():
    """Test simulation initialization."""
    print("\n🏭 Testing Simulation Setup...")
    
    try:
        from simulator import SimulationCoordinator
        from configs import SimulationConfig
        
        config = SimulationConfig(
            n_manufacturers=2,  # Small test
            n_periods=1,        # Single period
            llm_temperature=0.1
        )
        
        print(f"✅ Configuration created: {config.n_manufacturers} manufacturers, {config.n_periods} periods")
        
        # Test coordinator initialization
        coordinator = SimulationCoordinator(config)
        print("✅ Simulation coordinator created successfully")
        
        # Check that agents are initialized
        print(f"✅ {len(coordinator.environment.manufacturers)} manufacturers initialized")
        print(f"✅ Buyer agent: {coordinator.environment.buyer.agent_id}")
        print(f"✅ FDA agent: {coordinator.environment.fda.agent_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_mini_simulation():
    """Run a minimal simulation to test end-to-end functionality."""
    print("\n🚀 Running Mini Simulation (1 period, 2 manufacturers)...")
    
    try:
        from simulator import SimulationCoordinator
        from configs import SimulationConfig
        
        config = SimulationConfig(
            n_manufacturers=2,
            n_periods=1,
            disruption_probability=0.0,  # No disruptions for test
            llm_temperature=0.1
        )
        
        coordinator = SimulationCoordinator(config)
        results = await coordinator.run_simulation()
        
        print("✅ Mini simulation completed successfully!")
        
        # Print key results
        metrics = results['summary_metrics']
        print(f"   Peak shortage: {metrics['peak_shortage_percentage']:.1%}")
        print(f"   Total manufacturer profit: {metrics['total_manufacturer_profit']:.3f}")
        print(f"   Buyer cost: {results['buyer_total_cost']:.3f}")
        
        # Print logging info
        logging_info = results.get('logging_session', {})
        if logging_info:
            print(f"   Simulation ID: {logging_info.get('simulation_id', 'unknown')}")
            print(f"   Log events: {logging_info.get('total_events', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mini simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧬 Drug Shortage Simulation - Setup Test")
    print("=" * 50)
    
    tests = [
        ("API Key Setup", test_api_key_setup),
        ("OpenAI Library", test_openai_library),
        ("Mock LLM Calls", test_mock_llm_call),
        ("Simulation Setup", test_simulation_setup),
        ("Mini Simulation", run_mini_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your simulation is ready to run.")
        print("   Use: python main.py")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix the issues above.")
        if passed >= 3:  # If basic functionality works
            print("   Basic functionality appears to work - you can try running with mock responses.")

if __name__ == "__main__":
    asyncio.run(main())