# File: web_server.py
"""
Flask-SocketIO Web Server for ShortageSim
Integrates the existing simulation framework with the web interface
"""

import asyncio

import sys
import os
import threading
import uuid

from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simulator import SimulationCoordinator
from src.configs import SimulationConfig
from src.prompts import PromptManager

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Global variables for managing simulations
active_simulations = {}
simulation_threads = {}

class WebSimulationCoordinator:
    """
    Wrapper around SimulationCoordinator to handle web-based communication
    """
    
    def __init__(self, config: SimulationConfig, session_id: str):
        self.config = config
        self.session_id = session_id
        self.coordinator = SimulationCoordinator(config)
        self.is_running = False
        self.current_period = 0
        
    async def run_with_updates(self, start_with_disruption: bool = False):
        """Run simulation with real-time web updates"""
        try:
            self.is_running = True
            
            # Send initialization status
            socketio.emit('simulation_status', {
                'status': 'running',
                'detail': 'Initializing simulation...',
                'progress': 0,
                'session_id': self.session_id
            })
            
            # Setup logging callback for real-time updates  
            self._setup_logging_callbacks()
            
            # Run the simulation
            results = await self.coordinator.run_simulation(start_with_disruption)
            
            # Send completion status
            socketio.emit('simulation_status', {
                'status': 'completed',
                'detail': 'Simulation completed successfully!',
                'progress': 100,
                'session_id': self.session_id
            })
            
            # Send final results
            socketio.emit('simulation_results', {
                'results': self._serialize_results(results),
                'session_id': self.session_id
            })
            
            return results
            
        except Exception as e:
            self.is_running = False
            socketio.emit('simulation_status', {
                'status': 'error',
                'detail': f'Simulation failed: {str(e)}',
                'progress': 0,
                'session_id': self.session_id
            })
            raise
        finally:
            self.is_running = False
            
    def _setup_logging_callbacks(self):
        """Setup callbacks to stream simulation events to the web interface"""
        
        # Hook into the existing logging system
        original_log_period_start = self.coordinator.simulation_logger.log_period_start
        original_log_market_outcome = self.coordinator.simulation_logger.log_market_outcome
        
        def enhanced_log_period_start(period):
            self.current_period = period
            progress = ((period + 1) / self.config.n_periods) * 100
            
            socketio.emit('simulation_status', {
                'status': 'running',
                'detail': f'Running Period {period + 1}/{self.config.n_periods}...',
                'progress': progress,
                'session_id': self.session_id
            })
            
            return original_log_period_start(period)
            
        def enhanced_log_market_outcome(period, market_state, allocations, manufacturer_states):
            # Send period data to web interface
            period_data = self._create_period_data(period, market_state, allocations, manufacturer_states)
            
            socketio.emit('period_completed', {
                'period': period,
                'data': period_data,
                'session_id': self.session_id
            })
            
            return original_log_market_outcome(period, market_state, allocations, manufacturer_states)
        
        # Replace methods with enhanced versions
        self.coordinator.simulation_logger.log_period_start = enhanced_log_period_start
        self.coordinator.simulation_logger.log_market_outcome = enhanced_log_market_outcome
    
    def _create_period_data(self, period, market_state, allocations, manufacturer_states):
        """Format period data for web interface"""
        
        # Get agent decisions from history
        manufacturer_decisions = []
        for i, mfr in enumerate(self.coordinator.environment.manufacturers):
            if len(mfr.decision_history) > period:
                decision = mfr.decision_history[period]
                manufacturer_decisions.append({
                    'id': i,
                    'investment': decision.get('final_decision', {}).get('decision', {}).get('capacity_investment', 0),
                    'reasoning': decision.get('final_decision', {}).get('reasoning', {}).get('market_analysis', 'No reasoning available')[:100] + '...'
                })
        
        buyer_decision = {}
        if len(self.coordinator.environment.buyer.decision_history) > period:
            buyer_history = self.coordinator.environment.buyer.decision_history[period]
            buyer_decision = {
                'demand_quantity': buyer_history.get('final_decision', {}).get('decision', {}).get('demand_quantity', 1.0),
                'reasoning': buyer_history.get('final_decision', {}).get('reasoning', {}).get('supply_risk_assessment', 'No reasoning available')[:100] + '...'
            }
        
        fda_decision = {}
        if len(self.coordinator.environment.fda.decision_history) > period:
            fda_history = self.coordinator.environment.fda.decision_history[period]
            fda_decision = {
                'announcement_type': fda_history.get('final_decision', {}).get('decision', {}).get('announcement_type', 'none'),
                'reasoning': fda_history.get('final_decision', {}).get('reasoning', {}).get('shortage_assessment', 'No reasoning available')[:100] + '...'
            }
        
        return {
            'period': period,
            'total_demand': market_state.total_demand,
            'total_supply': market_state.total_supply,
            'shortage_amount': market_state.shortage_amount,
            'shortage_percentage': market_state.shortage_percentage,
            'disrupted_manufacturers': market_state.disrupted_manufacturers,
            'fda_announcement': market_state.fda_announcement or "",
            'agent_decisions': {
                'manufacturers': manufacturer_decisions,
                'buyer': buyer_decision,
                'fda': fda_decision
            }
        }
    
    def _serialize_results(self, results):
        """Convert simulation results to JSON-serializable format"""
        
        # Convert any non-serializable objects
        serialized = {}
        
        for key, value in results.items():
            if key == 'config':
                # Convert config dataclass to dict
                serialized[key] = value
            elif key == 'market_trajectory':
                # Convert market states to dicts
                serialized[key] = value
            else:
                serialized[key] = value
                
        return serialized
    
    def stop(self):
        """Stop the running simulation"""
        self.is_running = False
        # You might need to add more cleanup logic here

# Route handlers
@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('shortagesim.html')

@app.route('/api/config/defaults')
def get_default_config():
    """Get default configuration values"""
    config = SimulationConfig()
    prompt_manager = PromptManager()
    
    # Get default prompts
    mfr_sys, _, _ = prompt_manager.get_prompt('manufacturer', 'collector_analyst', 
                                             manufacturer_id=0, n_manufacturers=4, 
                                             period=0, n_periods=4, current_capacity=0.25,
                                             disruption_status='operational', recovery_periods=0,
                                             fda_announcement='None', last_demand=1.0, 
                                             disrupted_count=0, last_production=0.25,
                                             baseline_production=0.25)
    
    buyer_sys, _, _ = prompt_manager.get_prompt('buyer', 'collector_analyst',
                                               period=0, n_periods=4, initial_demand=1.0,
                                               fda_announcement='None', last_supply=1.0,
                                               last_demand=1.0, inventory=0.5, 
                                               disrupted_count=0, n_manufacturers=4,
                                               unit_profit=1.0, holding_cost=0.1,
                                               stockout_penalty=1.1)
    
    fda_sys, _, _ = prompt_manager.get_prompt('fda', 'collector_analyst',
                                             period=0, n_periods=4, last_supply=1.0,
                                             last_demand=1.0, shortage_amount=0,
                                             shortage_percentage=0, disrupted_count=0,
                                             n_manufacturers=4)
    
    return jsonify({
        'config': {
            'n_manufacturers': config.n_manufacturers,
            'n_periods': config.n_periods,
            'disruption_probability': config.disruption_probability,
            'disruption_magnitude': config.disruption_magnitude,
            'capacity_cost': config.capacity_cost,
            'unit_profit': config.unit_profit,
            'holding_cost': config.holding_cost,
            'stockout_penalty': config.stockout_penalty,
            'llm_model': config.llm_model,
            'temperature': config.llm_temperature,
            'max_retries': config.max_retries
        },
        'prompts': {
            'manufacturer_system': mfr_sys,
            'buyer_system': buyer_sys,
            'fda_system': fda_sys
        }
    })

@app.route('/api/simulations', methods=['GET'])
def list_simulations():
    """List active simulations"""
    return jsonify({
        'active_simulations': list(active_simulations.keys()),
        'count': len(active_simulations)
    })

@app.route('/api/simulations/<session_id>/stop', methods=['POST'])
def stop_simulation(session_id):
    """Stop a running simulation"""
    if session_id in active_simulations:
        active_simulations[session_id].stop()
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Simulation not found'}), 404

@app.route('/api/results/<session_id>/export/<format>')
def export_results(session_id, format):
    """Export simulation results"""
    if session_id not in active_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    # Implementation would depend on your existing export functionality
    # For now, return a placeholder
    return jsonify({'message': f'Export in {format} format not yet implemented'})

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = str(uuid.uuid4())
    emit('session_created', {'session_id': session_id})
    print(f"Client connected with session ID: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('run_simulation')
def handle_run_simulation(data):
    """Handle simulation run request"""
    session_id = data.get('session_id')
    config_data = data.get('config', {})
    
    try:
        # Create SimulationConfig from web data
        config = create_config_from_web_data(config_data)
        
        # Create coordinator
        coordinator = WebSimulationCoordinator(config, session_id)
        active_simulations[session_id] = coordinator
        
        # Run simulation in separate thread
        def run_simulation_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    coordinator.run_with_updates(
                        start_with_disruption=config_data.get('force_disruption', False)
                    )
                )
            except Exception as e:
                print(f"Simulation error: {e}")
            finally:
                loop.close()
                if session_id in active_simulations:
                    del active_simulations[session_id]
        
        thread = threading.Thread(target=run_simulation_thread)
        simulation_threads[session_id] = thread
        thread.start()
        
        emit('simulation_started', {'session_id': session_id})
        
    except Exception as e:
        emit('simulation_error', {
            'error': str(e),
            'session_id': session_id
        })

@socketio.on('stop_simulation')
def handle_stop_simulation(data):
    """Handle simulation stop request"""
    session_id = data.get('session_id')
    
    if session_id in active_simulations:
        active_simulations[session_id].stop()
        emit('simulation_stopped', {'session_id': session_id})
    else:
        emit('simulation_error', {
            'error': 'Simulation not found',
            'session_id': session_id
        })

def create_config_from_web_data(config_data):
    """Convert web form data to SimulationConfig"""
    
    config = SimulationConfig(
        n_manufacturers=config_data.get('n_manufacturers', 4),
        n_periods=config_data.get('n_periods', 4),
        disruption_probability=config_data.get('disruption_probability', 0.05),
        disruption_magnitude=config_data.get('disruption_magnitude', 0.2),
        capacity_cost=config_data.get('capacity_cost', 0.5),
        unit_profit=config_data.get('unit_profit', 1.0),
        holding_cost=config_data.get('holding_cost', 0.1),
        stockout_penalty=config_data.get('stockout_penalty', 1.1),
        llm_model=config_data.get('llm_model', 'gpt-4o'),
        llm_temperature=config_data.get('temperature', 0.3),
        max_retries=config_data.get('max_retries', 3),
        api_key=config_data.get('api_key') or None,
        n_disruptions_if_forced_disruption=config_data.get('n_disruptions_if_forced_disruption', 1)
    )
    
    return config

# Development server
if __name__ == '__main__':
    # Create templates directory and save the HTML file
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    print("Starting ShortageSim Web Server...")
    print("Access the interface at: http://localhost:5000")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
