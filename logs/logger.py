"""
Advanced Logging System with Rotation and Analytics
"""
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional
import threading
from queue import Queue
import hashlib
import gzip
import re

class AdvancedLogger:
    """Advanced logging system with multiple handlers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_queue = Queue()
        self.log_analytics = LogAnalytics()
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Initialize logging system
        self._setup_logging()
        
        # Start log processor thread
        self.processor_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.processor_thread.start()
        
        print("📝 অ্যাডভান্সড লগিং সিস্টেম প্রস্তুত")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger('MayaAssistant')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (rotating)
        file_handler = RotatingFileHandler(
            'logs/maya_assistant.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Error handler (separate file for errors)
        error_handler = RotatingFileHandler(
            'logs/errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
        
        # JSON handler for structured logging
        self.json_logger = JSONLogger()
    
    def log(self, level: str, message: str, extra: Dict[str, Any] = None, 
           module: str = None, function: str = None):
        """Log message with additional context"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level.upper(),
                "message": message,
                "module": module or self._get_caller_module(),
                "function": function or self._get_caller_function(),
                "extra": extra or {},
                "thread": threading.current_thread().name,
                "process_id": os.getpid()
            }
            
            # Add to queue for processing
            self.log_queue.put(log_entry)
            
            # Also log using standard logger
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message, extra=extra)
            
        except Exception as e:
            print(f"❌ লগিং ত্রুটি: {e}")
    
    def debug(self, message: str, extra: Dict[str, Any] = None, 
             module: str = None, function: str = None):
        """Log debug message"""
        self.log("debug", message, extra, module, function)
    
    def info(self, message: str, extra: Dict[str, Any] = None, 
            module: str = None, function: str = None):
        """Log info message"""
        self.log("info", message, extra, module, function)
    
    def warning(self, message: str, extra: Dict[str, Any] = None, 
               module: str = None, function: str = None):
        """Log warning message"""
        self.log("warning", message, extra, module, function)
    
    def error(self, message: str, extra: Dict[str, Any] = None, 
             module: str = None, function: str = None):
        """Log error message"""
        self.log("error", message, extra, module, function)
    
    def critical(self, message: str, extra: Dict[str, Any] = None, 
                module: str = None, function: str = None):
        """Log critical message"""
        self.log("critical", message, extra, module, function)
    
    def log_interaction(self, user_input: str, ai_response: str, 
                       confidence: float, emotion: str):
        """Log AI interaction"""
        interaction_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "user_input": user_input,
            "ai_response": ai_response,
            "confidence": confidence,
            "emotion": emotion,
            "session_id": self._get_session_id()
        }
        
        # Save to interactions log
        self._save_interaction_log(interaction_log)
        
        # Also log normally
        self.info(f"ইন্টারঅ্যাকশন: {user_input[:50]}...", extra=interaction_log)
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system event"""
        event_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "details": details,
            "system_info": self._get_system_info()
        }
        
        # Save to system log
        self._save_system_log(event_log)
        
        self.info(f"সিস্টেম ইভেন্ট: {event_type}", extra=event_log)
    
    def log_performance(self, operation: str, duration: float, 
                       details: Dict[str, Any] = None):
        """Log performance metrics"""
        perf_log = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "operation": operation,
            "duration_ms": duration * 1000,
            "details": details or {}
        }
        
        # Save to performance log
        self._save_performance_log(perf_log)
        
        self.debug(f"পারফরম্যান্স: {operation} - {duration:.3f}s", extra=perf_log)
    
    def _process_log_queue(self):
        """Process log entries from queue"""
        while True:
            log_entry = self.log_queue.get()
            if log_entry is None:
                break
            
            try:
                # Process log entry
                self.json_logger.log(log_entry)
                
                # Update analytics
                self.log_analytics.update(log_entry)
                
                # Check for alerts
                self._check_alerts(log_entry)
                
            except Exception as e:
                print(f"❌ লগ প্রসেসিং ত্রুটি: {e}")
            
            self.log_queue.task_done()
    
    def _check_alerts(self, log_entry: Dict[str, Any]):
        """Check log entries for alert conditions"""
        # Check for error spikes
        if log_entry["level"] == "ERROR":
            recent_errors = self.log_analytics.get_recent_errors(minutes=5)
            if len(recent_errors) > 10:
                self._send_alert(f"ত্রুটি স্পাইক: ৫ মিনিটে {len(recent_errors)}টি ত্রুটি")
        
        # Check for critical errors
        if log_entry["level"] == "CRITICAL":
            self._send_alert(f"সমালোচনামূলক ত্রুটি: {log_entry['message'][:100]}")
    
    def _send_alert(self, message: str):
        """Send alert notification"""
        # In production, send via email, Slack, etc.
        print(f"🚨 অ্যালার্ট: {message}")
        
        # Log the alert
        self.warning(f"অ্যালার্ট: {message}")
    
    def _save_interaction_log(self, log_entry: Dict[str, Any]):
        """Save interaction log to file"""
        try:
            log_file = "logs/interactions.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ ইন্টারঅ্যাকশন লগ সংরক্ষণ করা যায়নি: {e}")
    
    def _save_system_log(self, log_entry: Dict[str, Any]):
        """Save system log to file"""
        try:
            log_file = "logs/system_events.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ সিস্টেম লগ সংরক্ষণ করা যায়নি: {e}")
    
    def _save_performance_log(self, log_entry: Dict[str, Any]):
        """Save performance log to file"""
        try:
            log_file = "logs/performance.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ পারফরম্যান্স লগ সংরক্ষণ করা যায়নি: {e}")
    
    def _get_caller_module(self) -> str:
        """Get calling module name"""
        try:
            import inspect
            frame = inspect.currentframe()
            # Go back 3 frames: _get_caller_module -> log -> actual caller
            for _ in range(3):
                frame = frame.f_back
            
            module = inspect.getmodule(frame)
            if module:
                return module.__name__
        except:
            pass
        
        return "unknown"
    
    def _get_caller_function(self) -> str:
        """Get calling function name"""
        try:
            import inspect
            frame = inspect.currentframe()
            # Go back 3 frames
            for _ in range(3):
                frame = frame.f_back
            
            return frame.f_code.co_name
        except:
            return "unknown"
    
    def _get_session_id(self) -> str:
        """Generate session ID"""
        session_hash = hashlib.md5()
        session_hash.update(str(datetime.now().timestamp()).encode())
        session_hash.update(str(os.getpid()).encode())
        return session_hash.hexdigest()[:8]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "boot_time": psutil.boot_time(),
            "python_version": os.sys.version
        }
    
    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get log statistics for given period"""
        return self.log_analytics.get_statistics(hours)
    
    def search_logs(self, query: str, level: str = None, 
                   hours: int = 24) -> List[Dict[str, Any]]:
        """Search logs"""
        return self.log_analytics.search(query, level, hours)
    
    def compress_old_logs(self, days: int = 7):
        """Compress logs older than specified days"""
        self.log_analytics.compress_old_logs(days)

class JSONLogger:
    """JSON structured logger"""
    
    def __init__(self):
        self.log_dir = "logs/json"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Current log file
        self.current_date = datetime.now().date()
        self.current_file = self._get_log_file_path()
    
    def log(self, log_entry: Dict[str, Any]):
        """Log structured JSON entry"""
        try:
            # Check if we need to rotate to new day
            today = datetime.now().date()
            if today != self.current_date:
                self.current_date = today
                self.current_file = self._get_log_file_path()
            
            # Write log entry
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"❌ JSON লগিং ত্রুটি: {e}")
    
    def _get_log_file_path(self) -> str:
        """Get log file path for current date"""
        date_str = self.current_date.strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"log_{date_str}.jsonl")

class LogAnalytics:
    """Log analytics and statistics"""
    
    def __init__(self):
        self.log_stats = {
            "total_logs": 0,
            "by_level": {},
            "by_module": {},
            "by_hour": {},
            "error_rate": 0
        }
        self.last_update = datetime.now()
    
    def update(self, log_entry: Dict[str, Any]):
        """Update analytics with new log entry"""
        # Update counters
        self.log_stats["total_logs"] += 1
        
        # Update by level
        level = log_entry["level"]
        self.log_stats["by_level"][level] = self.log_stats["by_level"].get(level, 0) + 1
        
        # Update by module
        module = log_entry["module"]
        self.log_stats["by_module"][module] = self.log_stats["by_module"].get(module, 0) + 1
        
        # Update by hour
        hour = datetime.fromisoformat(log_entry["timestamp"]).hour
        self.log_stats["by_hour"][hour] = self.log_stats["by_hour"].get(hour, 0) + 1
        
        # Update error rate
        error_count = self.log_stats["by_level"].get("ERROR", 0)
        warning_count = self.log_stats["by_level"].get("WARNING", 0)
        total = self.log_stats["total_logs"]
        
        if total > 0:
            self.log_stats["error_rate"] = (error_count + warning_count * 0.5) / total
        
        self.last_update = datetime.now()
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for given period"""
        stats = self.log_stats.copy()
        
        # Add calculated fields
        stats["average_logs_per_hour"] = stats["total_logs"] / max(1, hours)
        stats["most_active_module"] = max(
            stats["by_module"].items(), 
            key=lambda x: x[1], 
            default=("none", 0)
        )[0]
        stats["most_active_hour"] = max(
            stats["by_hour"].items(), 
            key=lambda x: x[1], 
            default=(0, 0)
        )[0]
        
        return stats
    
    def get_recent_errors(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent error logs"""
        errors = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        try:
            # Read from JSON logs
            log_files = self._get_recent_log_files()
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry["timestamp"])
                            
                            if (log_entry["level"] in ["ERROR", "CRITICAL"] and 
                                log_time > cutoff_time):
                                errors.append(log_entry)
                        except:
                            continue
        except:
            pass
        
        return errors
    
    def search(self, query: str, level: str = None, 
              hours: int = 24) -> List[Dict[str, Any]]:
        """Search logs"""
        results = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            log_files = self._get_recent_log_files(hours)
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry["timestamp"])
                            
                            if log_time < cutoff_time:
                                continue
                            
                            # Check level filter
                            if level and log_entry["level"] != level.upper():
                                continue
                            
                            # Check query
                            query_lower = query.lower()
                            message = log_entry.get("message", "").lower()
                            module = log_entry.get("module", "").lower()
                            
                            if (query_lower in message or 
                                query_lower in module or
                                query in json.dumps(log_entry).lower()):
                                results.append(log_entry)
                        except:
                            continue
        except Exception as e:
            print(f"❌ লগ সার্চ ত্রুটি: {e}")
        
        return results
    
    def compress_old_logs(self, days: int = 7):
        """Compress logs older than specified days"""
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        try:
            log_dir = "logs/json"
            for filename in os.listdir(log_dir):
                if filename.startswith("log_") and filename.endswith(".jsonl"):
                    # Extract date from filename
                    date_str = filename[4:14]  # log_YYYY-MM-DD.jsonl
                    try:
                        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        
                        if file_date < cutoff_date:
                            # Compress file
                            file_path = os.path.join(log_dir, filename)
                            compressed_path = file_path + ".gz"
                            
                            with open(file_path, 'rb') as f_in:
                                with gzip.open(compressed_path, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            # Remove original
                            os.remove(file_path)
                            print(f"✅ লগ কম্প্রেস করা হয়েছে: {filename}")
                    except:
                        continue
        except Exception as e:
            print(f"❌ লগ কম্প্রেশন ত্রুটি: {e}")
    
    def _get_recent_log_files(self, hours: int = 24) -> List[str]:
        """Get list of recent log files"""
        log_files = []
        log_dir = "logs/json"
        
        if not os.path.exists(log_dir):
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for filename in os.listdir(log_dir):
            if filename.startswith("log_") and filename.endswith(".jsonl"):
                file_path = os.path.join(log_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time > cutoff_time:
                    log_files.append(file_path)
        
        return sorted(log_files)

class LogMonitor:
    """Real-time log monitoring"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.monitoring = False
        self.callbacks = []
    
    def start_monitoring(self):
        """Start real-time log monitoring"""
        self.monitoring = True
        print("👁️ লগ মনিটরিং শুরু করা হয়েছে")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        print("🛑 লগ মনিটরিং বন্ধ করা হয়েছে")
    
    def register_callback(self, callback):
        """Register callback for log events"""
        self.callbacks.append(callback)
    
    def watch_for_pattern(self, pattern: str, callback):
        """Watch for specific pattern in logs"""
        import re
        regex = re.compile(pattern)
        
        def pattern_callback(log_entry):
            if regex.search(json.dumps(log_entry)):
                callback(log_entry)
        
        self.register_callback(pattern_callback)