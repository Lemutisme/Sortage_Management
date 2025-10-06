"""
Drug Shortage Multi-Agent Simulation System with Comprehensive Logging
====================================================================

A comprehensive simulation framework for modeling pharmaceutical supply chain
dynamics with LLM-based agents and detailed logging for analysis.
"""

from typing import Dict, Any
import pandas as pd
import asyncio

from pathlib import Path
from datetime import datetime

from simulator import SimulationCoordinator
from configs import SimulationConfig
from tqdm import tqdm

# =============================================================================
# Enhanced Example Usage with Logging
# =============================================================================

async def run_logged_simulation(config: SimulationConfig, 
                                start_with_disruption: bool = False,
                                simulation_name: str = "default") -> Dict[str, Any]:
    """
    Run a simulation with comprehensive logging and analysis.
    
    Args:
        config: Simulation configuration
        simulation_name: Descriptive name for this simulation run
        
    Returns:
        Complete simulation results with logging metadata
    """
    
    print(f"\n🚀 Starting simulation: {simulation_name}")
    print(f"Configuration: {config.n_manufacturers} manufacturers, {config.n_periods} periods")
    
    # Create and run simulation with logging
    coordinator = SimulationCoordinator(config)
    
    try:
        results = await coordinator.run_simulation(start_with_disruption)
        
        # Print logging summary
        logging_summary = coordinator.get_logging_summary()
        print(f"\n📊 Logging Summary:")
        print(f"  Simulation ID: {logging_summary['simulation_id']}")
        print(f"  Log Directory: {logging_summary['session_directory']}")
        print(f"  Total Events: {logging_summary['logging_stats']['total_events']}")
        print(f"  Decision Events: {logging_summary['logging_stats']['decision_events']}")
        print(f"  LLM Calls: {logging_summary['logging_stats']['llm_calls']}")
        
        # Print simulation results summary
        print(f"\n📈 Simulation Results:")
        metrics = results['summary_metrics']
        print(f"  Peak shortage: {metrics['peak_shortage_percentage']:.1%}")
        print(f"  Shortage periods: {metrics['total_shortage_periods']}/{config.n_periods}")
        print(f"  Total buyer cost: {results['buyer_total_cost']:.3f}")
        print(f"  Total manufacturer profit: {metrics['total_manufacturer_profit']:.3f}")
        print(f"  FDA interventions: {metrics['fda_intervention_rate']:.1%} of periods")
        # print(f"  Peak shortage: {metrics['peak_shortage_percentage']}")
        # print(f"  Shortage periods: {metrics['total_shortage_periods']}/{config.n_periods}")
        # print(f"  Total buyer cost: {results['buyer_total_cost']}")
        # print(f"  Total manufacturer profit: {metrics['total_manufacturer_profit']}")
        # print(f"  FDA interventions: {metrics['fda_intervention_rate']} of periods")
        
        return results
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        raise

async def run_comparative_study(model_override: str = None, provider_override: str = None):
    """Run multiple simulations to compare different scenarios."""
    
    print("🔬 Running Comparative Study with Different Market Conditions")
    
    scenarios = [
        {
            "name": "Low Disruption Baseline",
            "config": SimulationConfig(
                n_manufacturers=4,
                n_periods=4,
                disruption_probability=0.02,
                disruption_magnitude=0.15
            )
        },
        {
            "name": "High Disruption Stress Test", 
            "config": SimulationConfig(
                n_manufacturers=4,
                n_periods=4,
                disruption_probability=0.10,
                disruption_magnitude=0.30
            )
        },
        {
            "name": "Concentrated Market (3 manufacturers)",
            "config": SimulationConfig(
                n_manufacturers=3,
                n_periods=4,
                disruption_probability=0.05,
                disruption_magnitude=0.20
            )
        },
        {
            "name": "Competitive Market (6 manufacturers)",
            "config": SimulationConfig(
                n_manufacturers=6,
                n_periods=4,
                disruption_probability=0.05,
                disruption_magnitude=0.20
            )
        }
    ]
    
    comparative_results = []
    
    for scenario in scenarios:
        print(f"\n--- Running Scenario: {scenario['name']} ---")
        
        try:
            # Apply overrides if provided
            if model_override:
                scenario['config'].llm_model = model_override
            if provider_override:
                scenario['config'].llm_provider = provider_override

            results = await run_logged_simulation(
                scenario['config'], 
                scenario['name']
            )
            
            # Extract key metrics for comparison
            metrics = results['summary_metrics']
            comparative_results.append({
                "scenario": scenario['name'],
                "n_manufacturers": scenario['config'].n_manufacturers,
                "disruption_prob": scenario['config'].disruption_probability,
                "disruption_magnitude": scenario['config'].disruption_magnitude,
                "peak_shortage": metrics['peak_shortage_percentage'],
                "avg_shortage": metrics['average_shortage_percentage'],
                "shortage_periods": metrics['total_shortage_periods'],
                "resolution_time": metrics['time_to_resolution'],
                "total_profit": metrics['total_manufacturer_profit'],
                "buyer_cost": results['buyer_total_cost'],
                "fda_interventions": len(results['fda_announcements']),
                "simulation_id": results['logging_session']['simulation_id']
            })
            
        except Exception as e:
            print(f"❌ Scenario '{scenario['name']}' failed: {e}")
            comparative_results.append({
                "scenario": scenario['name'],
                "error": str(e)
            })
    
    # Create comparison summary
    if comparative_results:
        comparison_df = pd.DataFrame([r for r in comparative_results if 'error' not in r])
        
        if not comparison_df.empty:
            print(f"\n📊 Comparative Analysis Summary:")
            print("=" * 80)
            
            for _, row in comparison_df.iterrows():
                print(f"{row['scenario']:<30} | "
                      f"Peak: {row['peak_shortage']:.1%} | "
                      f"Periods: {row['shortage_periods']}/4 | "
                      f"Resolution: {row['resolution_time'] or 'N/A'} | "
                      f"FDA: {row['fda_interventions']}")
            
            # Save comparison results
            comparison_df.to_csv("comparative_study_results.csv", index=False)
            print(f"\n💾 Comparative results saved to: comparative_study_results.csv")
        
        return comparative_results

async def analyze_agent_decisions(simulation_results: Dict[str, Any]):
    """Analyze agent decision patterns from simulation results."""
    
    print(f"\n🧠 Analyzing Agent Decision Patterns")
    
    # Manufacturer decision analysis
    manufacturer_decisions = simulation_results['decision_history']['manufacturers']
    
    print(f"\n📊 Manufacturer Decision Analysis:")
    for i, decisions in enumerate(manufacturer_decisions):
        if decisions:
            investments = []
            confidences = []
            
            for decision in decisions:
                final_decision = decision.get('final_decision', {})
                investment = float(final_decision.get('decision', {}).get('capacity_investment', 0))
                confidence = final_decision.get('confidence', 'unknown')
                
                investments.append(investment)
                confidences.append(confidence)
            
            total_investment = sum(investments)
            avg_confidence = len([c for c in confidences if c in ['moderate', 'high']]) / len(confidences) if confidences else 0
            
            print(f"  Manufacturer {i}: Total Investment: {total_investment:.3f}, Confidence Rate: {avg_confidence:.1%}")
    
    # Buyer decision analysis
    buyer_decisions = simulation_results['decision_history']['buyer']
    
    if buyer_decisions:
        print(f"\n🏥 Buyer Decision Analysis:")
        
        demand_quantities = []
        stockpiling_periods = 0
        
        for decision in buyer_decisions:
            final_decision = decision.get('final_decision', {})
            demand = final_decision.get('decision', {}).get('demand_quantity', 1.0)
            rationale = final_decision.get('decision', {}).get('demand_rationale', 'baseline')
            
            demand_quantities.append(demand)
            
            if 'stockpile' in rationale.lower() or demand > 1.2:
                stockpiling_periods += 1
        
        avg_demand = sum(demand_quantities) / len(demand_quantities) if demand_quantities else 1.0
        max_demand = max(demand_quantities) if demand_quantities else 1.0
        
        print(f"  Average Demand: {avg_demand:.3f} ({avg_demand:.1%} of baseline)")
        print(f"  Peak Demand: {max_demand:.3f}")
        print(f"  Stockpiling Periods: {stockpiling_periods}/{len(buyer_decisions)}")
    
    # FDA decision analysis
    fda_decisions = simulation_results['decision_history']['fda']
    
    if fda_decisions:
        print(f"\n🏛️ FDA Decision Analysis:")
        
        announcement_types = {}
        for decision in fda_decisions:
            final_decision = decision.get('final_decision', {})
            announcement_type = final_decision.get('decision', {}).get('announcement_type', 'none')
            announcement_types[announcement_type] = announcement_types.get(announcement_type, 0) + 1
        
        for ann_type, count in announcement_types.items():
            percentage = count / len(fda_decisions) * 100
            print(f"  {ann_type.title()}: {count} times ({percentage:.1f}%)")

def export_detailed_analysis(results: Dict[str, Any], export_dir: str = "analysis_exports"):
    """Export detailed analysis files for further research."""
    
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    
    simulation_id = results['logging_session']['simulation_id']
    
    print(f"\n💾 Exporting detailed analysis to {export_dir}/")
    
    # Export market trajectory
    market_df = pd.DataFrame(results['market_trajectory'])
    market_df.to_csv(export_path / f"market_trajectory_{simulation_id}.csv", index=False)
    
    # Export manufacturer states
    manufacturer_df = pd.DataFrame(results['manufacturer_summaries'])
    manufacturer_df.to_csv(export_path / f"manufacturer_states_{simulation_id}.csv", index=False)
    
    # Export buyer summary
    buyer_summary = results['buyer_summary']
    if isinstance(buyer_summary, dict) and 'status' not in buyer_summary:
        buyer_df = pd.DataFrame([buyer_summary])
        buyer_df.to_csv(export_path / f"buyer_summary_{simulation_id}.csv", index=False)
    
    # Export FDA announcements
    if results['fda_announcements']:
        fda_df = pd.DataFrame(results['fda_announcements'])
        fda_df.to_csv(export_path / f"fda_announcements_{simulation_id}.csv", index=False)
    
    print(f"  ✅ Analysis files exported with ID: {simulation_id}")

# =============================================================================
# Main Execution Functions
# =============================================================================

async def run_single_example(start_with_disruption: bool = False, model_override: str = None, provider_override: str = None, **config_overrides):
    """Run a single example simulation with detailed logging."""
    
    default_params = {
        "n_manufacturers": 2,
        "n_periods": 4,
        "disruption_probability": 0.05,
        "disruption_magnitude": 0.3,
        "llm_temperature": 0.3,
        "n_disruptions_if_forced_disruption": 1,
    }

    # keys you want to exclude
    exclude_keys = {'model', 'provider'}
    # remove them before merging
    filtered_overrides = {k: v for k, v in config_overrides.items() if k not in exclude_keys}
    final_params = {**default_params, **filtered_overrides}

    config = SimulationConfig(**final_params)

    # Apply overrides
    if model_override:
        config.llm_model = model_override
    if provider_override:
        config.llm_provider = provider_override

    results = await run_logged_simulation(config, start_with_disruption, "Single Example Run")

    # Analyze agent decisions
    await analyze_agent_decisions(results)
    
    # Export detailed analysis
    export_detailed_analysis(results)
    
    return results

async def run_gt_experiments(
        df: pd.DataFrame,
        show_progress: bool = True,
        export_dir: str = "gt_evaluation",
        n_simulations: int = 1,
        model_override: str = None,
        provider_override: str = None,
        **config_overrides
):
    if model_override:
        export_dir += f"/model_{model_override}"
    if config_overrides.get('llm_temperature') is not None:
        export_dir += f"/temp_{config_overrides['llm_temperature']}"
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = export_path  / f"gt_experiments_{ts}.csv"
    config_path = export_path / f"gt_experiments_config_{ts}.txt"
    exists_header = False

    save_config = False
    iterator = tqdm(df.itertuples(index=False), total=len(df)) if show_progress else df.itertuples(index=False)
    comparative_results = []
    comparison_df = pd.DataFrame()
    for row in iterator:
        row_dict = row._asdict()
        temp_results = []
        for sim in range(n_simulations):
            print(f"\n📊 Running Ground Truth Experiment: gt_id_{row_dict['gt_id']}_simulation_{sim}")
            print("=" * 80)
            try:
                default_params = {
                    "n_manufacturers": int(row_dict['n_manufacturers']),
                    "n_periods": int(row_dict['periods']),
                    "disruption_probability": 0.05,
                    "disruption_magnitude": row_dict['disruption_magnitude'],
                    "llm_temperature": 0.3,
                    "n_disruptions_if_forced_disruption": int(row_dict['disruption_number'])
                }
                print("In ground truth experiment, n_manufacturers, n_periods, disruption_magnitude are set by the GT data and cannot be overridden.")
                # Apply overrides
                for key, value in config_overrides.items():
                    if key in ["llm_temperature", "fda_mode"]:
                        default_params[key] = value
                        print("overriding", key, value)
                if model_override:
                    default_params['llm_model'] = model_override
                if provider_override:
                    default_params['llm_provider'] = provider_override

                config = SimulationConfig(**default_params)
                if not save_config:
                    with open(config_path, 'w') as f:
                        f.write(str(config))
                    print(f"\n💾 Configuration saved to: {config_path}")
                    save_config = True

                start_with_disruption = True if row_dict['disruption_number'] > 0 else False

                results = await run_logged_simulation(config, start_with_disruption, "gt_id_" + str(row_dict['gt_id']))
                # Analyze agent decisions
                await analyze_agent_decisions(results)
                # Export detailed analysis
                export_detailed_analysis(results)

                # Extract key metrics for comparison
                metrics = results['summary_metrics']
                temp_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "simulation_id": results['logging_session']['simulation_id'],
                    "#simulation": sim,
                    "gt_type":row_dict['gt_type'],
                    "n_manufacturers": config.n_manufacturers,
                    "periods":row_dict['periods'],
                    "disruption_prob": config.disruption_probability,
                    "disruption_magnitude": config.disruption_magnitude,
                    "trajectory": results['market_trajectory'],
                    "peak_shortage": metrics['peak_shortage_percentage'],
                    "avg_shortage": metrics['average_shortage_percentage'],
                    "shortage_periods": metrics['total_shortage_periods'],
                    "resolution_time": metrics['time_to_resolution'],
                    "total_profit": metrics['total_manufacturer_profit'],
                    "buyer_cost": results['buyer_total_cost'],
                    "fda_interventions": len(results['fda_announcements']),
                })
            except Exception as e:
                print(f"❌ Scenario gt_id_{str(row_dict['gt_id'])}_simulation_{sim} failed: {e}")
                comparative_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "error": str(e)
                })
    
        if temp_results:
            temp_df = pd.DataFrame([r for r in temp_results if 'error' not in r])
            comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True) 
            comparative_results+=temp_results
        
            if not temp_df.empty:
                if not exists_header:
                    temp_df.to_csv(csv_path, index=False)
                    exists_header = True
                else:
                    temp_df.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"\n💾 Comparative results saved to: gt_experiments_{ts}.csv")
    return comparison_df


async def run_quick_policy_test(model_override: str = None, provider_override: str = None, **config_overrides):
    """Quick test of different policy scenarios."""
    
    print("🎯 FDA Policy Effectiveness Test: Reactive vs. Proactive")
    
   # --- 1. Define Common Test Parameters for High Disruption ---
    common_params = {
        "n_manufacturers": 4,
        "n_periods": 10, # Good period length for policy comparison
        "disruption_probability": 0.08,
        "disruption_magnitude": 0.25,
    }

    final_params = {**common_params, **config_overrides}
    if "fda_mode" in final_params:
        del final_params["fda_mode"] # Remove to avoid conflict in SimulationConfig
        print("⚠️ Warning: Since we are comparing policy interventions, 'fda_mode' in config_overrides will be ignored and set explicitly for each policy.")

    # Apply overrides
    if model_override:
        final_params['llm_model'] = model_override
    if provider_override:
        final_params['llm_provider'] = provider_override

    # --- 2. Create the two Policy Configurations (Distinct Instances) ---

    # a) Reactive Policy Configuration
    reactive_config = SimulationConfig(
        **final_params,
        fda_mode='reactive' # Explicitly set the mode
    )
    
    # b) Proactive Policy Configuration
    proactive_config = SimulationConfig(
        **final_params,
        fda_mode='proactive' # Explicitly set the mode
    )
    
    # --- 3. Run Simulations and Collect Results ---
    
    print("\n--- Running REACTIVE Policy Test (High Disruption) ---")
    reactive_results = await run_logged_simulation(
        reactive_config, 
        "Reactive FDA Policy Test"
    )
    
    print("\n--- Running PROACTIVE Policy Test (High Disruption) ---")
    proactive_results = await run_logged_simulation(
        proactive_config, 
        "Proactive FDA Policy Test"
    )
    
    # --- 4. Compare Outcomes ---
    
    print(f"\n📊 Policy Test Comparison (Scenario: High Disruption):")
    
    # Reactive Metrics
    reactive_peak = reactive_results['summary_metrics']['peak_shortage_percentage']
    reactive_interventions = len(reactive_results['fda_announcements'])
    
    # Proactive Metrics
    proactive_peak = proactive_results['summary_metrics']['peak_shortage_percentage']
    proactive_interventions = len(proactive_results['fda_announcements'])
    
    # Output Table
    print("\n| Policy | Peak Shortage | Total Interventions |")
    print("|:---|:---|:---|")
    print(f"| **Reactive** | {reactive_peak:.1%} | {reactive_interventions} |")
    print(f"| **Proactive** | {proactive_peak:.1%} | {proactive_interventions} |")
    
    # Analysis
    if proactive_peak < reactive_peak:
        print("\n✅ **Conclusion:** The Proactive policy resulted in a lower peak shortage, indicating better performance in this scenario.")
    else:
        print("\n❌ **Conclusion:** The Reactive policy performed better or equally, suggesting the Proactive mode may be over-intervening or its prompts need tuning.")

if __name__ == "__main__":
    import argparse
    import asyncio
    from pathlib import Path
    import pandas as pd

    print("🧬 Drug Shortage Multi-Agent Simulation with Comprehensive Logging")
    print("=" * 70)

    # --- New Argument Parsing Logic using argparse ---
    parser = argparse.ArgumentParser(description="Run the Drug Shortage Multi-Agent Simulation.")

    parser.add_argument(
        "mode",
        type=str,
        nargs='?', # Makes the mode optional, defaults to None if not provided
        default="single", # Default mode if none is specified
        choices=["single", "comparative", "policy", "gt_experiment_disc", "gt_experiment_nondisc"],
        help="The simulation mode to run."
    )

    # Keeping --model and --provider for legacy compatibility if needed, but llm_model is preferred
    parser.add_argument("--model", type=str, help="Legacy: Specify a model name directly.")
    parser.add_argument("--provider", type=str, help="Legacy: Specify a provider directly.")

    parser.add_argument("--n_manufacturers", type=int, help="Number of manufacturers in the simulation.")
    parser.add_argument("--n_periods", type=int, help="Number of periods in the simulation.")
    parser.add_argument("--disruption_probability", type=float, help="Probability of disruption per manufacturer per period.")
    
    def parse_list_of_floats(arg_value):
        """
        Splits a comma-separated string into a list of float numbers.
        """
        try:
            # Split the string by comma, strip whitespace, and convert to float
            items = [float(item.strip()) for item in arg_value.split(',') if item.strip()]
            return items
        except ValueError:
            # Raise an error if any item can't be converted to a float
            raise argparse.ArgumentTypeError(f"Invalid list value: '{arg_value}'. Items must be valid floating-point numbers.")
    parser.add_argument("--market_share", type=parse_list_of_floats, help="Comma-separated market share percentages for manufacturers (e.g., '0.5,0.3,0.2').")
    parser.add_argument("--fda_mode", type=str, choices=["reactive", "proactive"], help="FDA policy mode to use in the simulation.")

    parser.add_argument("--llm_temperature", type=float, help="LLM sampling temperature for decision-making.")

    args = parser.parse_args()

    if args.market_share is not None:
        if args.n_manufacturers is None:
            print("\n❌ Error: The '--market_share' argument requires '--n_manufacturers' to be set.")
            print("Please provide both arguments when customizing market share.")
            exit(1)
        elif len(args.market_share) != args.n_manufacturers:
            print(f"\n❌ Error: The length of '--market_share' ({len(args.market_share)}) does not match '--n_manufacturers' ({args.n_manufacturers}).")
            print("Please ensure the market share list matches the number of manufacturers.")
            exit(1)
        elif sum(args.market_share) == 0.0:
            print(f"\n❌ Error: The sum of '--market_share' cannot be zero.")
            print("Please provide valid market share percentages that sum to a positive value.")
            exit(1)
        else:
            args.market_share = [share / sum(args.market_share) for share in args.market_share] # Normalize to sum to 1.0
            print(f"\n✅ Normalized market share: {args.market_share}")

    # Use the new llm_model argument, but allow legacy override
    model_override = args.model #if args.model else args.llm_model
    provider_override = args.provider

    # Create a dictionary of SimulationConfig overrides
    config_overrides = {
        key: value for key, value in args.__dict__.items() if value is not None and key not in ['mode', 'model', 'provider']
    }

    # --- Main Simulation Logic ---
    if args.mode == "comparative":
        print(f"Running comparative study with model: {model_override}...")
        asyncio.run(run_comparative_study(model_override, provider_override))
        
    elif args.mode == "policy":
        print(f"Running policy test with model: {model_override}...")
        asyncio.run(run_quick_policy_test(model_override, provider_override, **config_overrides))
        
    elif args.mode == "gt_experiment_disc":
        print(f"Running ground truth experiment (Discontinued) with model: {model_override}...")
        HERE = Path(__file__).resolve().parent
        csv_path = HERE / "../data" / "GT_Disc.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {df.shape[0]} trajectories from {csv_path.name}")
        asyncio.run(run_gt_experiments(df, n_simulations=1, model_override=model_override, provider_override=provider_override, **config_overrides))
        
    elif args.mode == "gt_experiment_nondisc":
        print(f"Running ground truth experiment (No Discontinued) with model: {model_override}...")
        HERE = Path(__file__).resolve().parent
        csv_path = HERE / "../data" / "GT_NoDisc.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {df.shape[0]} trajectories from {csv_path.name}")
        # Example of running a subset, adjust as needed
        asyncio.run(run_gt_experiments(df.iloc[:], model_override=model_override, provider_override=provider_override, **config_overrides))
        
    else: # This handles the "single" mode (default)
        print(f"Running single example simulation with model: {model_override}...")
        asyncio.run(run_single_example(start_with_disruption=True, model_override=model_override, provider_override=provider_override, **config_overrides))

    # --- Footer ---
    print("\n✅ Simulation completed! Check the generated log files for detailed analysis.")
    print("💡 Log files include:")
    print("   - Comprehensive event logs (JSONL format)")
    print("   - Agent-specific decision logs")
    print("   - Market outcome tracking")
    print("   - LLM call logs for debugging")
    print("   - CSV exports for analysis")
