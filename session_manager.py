# File: session_manager.py
"""
Session and simulation lifecycle management
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from config_web import WebConfig
from web_simulation_coordinator import WebSimulationCoordinator

@dataclass
class SimulationSession:
    session_id: str
    coordinator: Optional['WebSimulationCoordinator']
    created_at: datetime
    last_activity: datetime
    status: str  # 'idle', 'running', 'completed', 'error'
    results: Optional[Dict] = None

class SessionManager:
    """Manages simulation sessions and cleanup"""
    
    def __init__(self, config: 'WebConfig'):
        self.config = config
        self.sessions: Dict[str, SimulationSession] = {}
        self.cleanup_thread = None
        self.start_cleanup_thread()
    
    def create_session(self, session_id: str) -> SimulationSession:
        """Create a new simulation session"""
        session = SimulationSession(
            session_id=session_id,
            coordinator=None,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            status='idle'
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SimulationSession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session
    
    def update_session_status(self, session_id: str, status: str, results: Optional[Dict] = None):
        """Update session status"""
        if session_id in self.sessions:
            self.sessions[session_id].status = status
            self.sessions[session_id].last_activity = datetime.now()
            if results:
                self.sessions[session_id].results = results
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        timeout = timedelta(seconds=self.config.SESSION_TIMEOUT)
        
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_activity > timeout
        ]
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
    
    def remove_session(self, session_id: str):
        """Remove and cleanup session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Stop running simulation if any
            if session.coordinator and session.coordinator.is_running:
                session.coordinator.stop()
            
            del self.sessions[session_id]
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Clean up every 5 minutes
                self.cleanup_expired_sessions()
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.status == 'running')
        
        return {
            'total_sessions': total_sessions,
            'active_simulations': active_sessions,
            'max_concurrent': self.config.MAX_CONCURRENT_SIMULATIONS
        }