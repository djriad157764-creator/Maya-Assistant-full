"""
Configuration Management for Maya Assistant
"""
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml

class RunMode(Enum):
    """Application run modes"""
    CLI = "cli"
    GUI = "gui"
    BACKGROUND = "background"
    API = "api"

class VoiceGender(Enum):
    """Voice gender options"""
    FEMALE = "female"
    MALE = "male"
    NEUTRAL = "neutral"

@dataclass
class BrainConfig:
    """Brain configuration"""
    model_name: str = "banglabert"
    model_path: str = "models/banglabert"
    reasoning_enabled: bool = True
    learning_enabled: bool = True
    emotion_detection: bool = True
    max_thought_history: int = 1000
    max_decision_history: int = 500
    confidence_threshold: float = 0.6

@dataclass
class MemoryConfig:
    """Memory configuration"""
    short_term_capacity: int = 100
    long_term_capacity: int = 10000
    consolidation_threshold: int = 5
    memory_types: List[str] = field(default_factory=lambda: [
        "factual", "procedural", "episodic", "semantic", "emotional"
    ])
    auto_consolidate: bool = True
    consolidation_interval: int = 3600  # seconds

@dataclass
class VoiceConfig:
    """Voice configuration"""
    language: str = "bn"
    gender: VoiceGender = VoiceGender.FEMALE
    speech_rate: int = 150
    volume: float = 0.9
    emotion_enabled: bool = True
    voice_cache_enabled: bool = True
    voice_cache_size: int = 100
    listen_timeout: int = 5
    phrase_time_limit: int = 10
    energy_threshold: int = 300
    pause_threshold: float = 0.8

@dataclass
class VisionConfig:
    """Vision configuration"""
    camera_enabled: bool = True
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    screenshot_enabled: bool = True
    face_recognition_enabled: bool = True
    object_detection_enabled: bool = True
    ocr_enabled: bool = True
    gesture_recognition_enabled: bool = True
    save_captures: bool = True

@dataclass
class SystemConfig:
    """System configuration"""
    auto_update: bool = True
    update_check_interval: int = 86400  # seconds
    backup_enabled: bool = True
    backup_interval: int = 604800  # seconds (7 days)
    performance_monitoring: bool = True
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "cpu_limit": 80.0,
        "memory_limit": 85.0,
        "disk_limit": 90.0
    })
    allowed_applications: List[str] = field(default_factory=lambda: [
        "notepad", "calculator", "chrome", "firefox", "explorer"
    ])

@dataclass
class InternetConfig:
    """Internet configuration"""
    enabled: bool = True
    search_engine: str = "google"
    default_browser: str = "chrome"
    weather_api_key: str = ""
    news_api_key: str = ""
    translate_api_key: str = ""
    enable_proxy: bool = False
    proxy_url: str = ""

@dataclass
class StorageConfig:
    """Storage configuration"""
    encryption_enabled: bool = True
    encryption_password: str = "default_password_change_me"
    compression_enabled: bool = True
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    backup_count: int = 5
    auto_cleanup: bool = True
    cleanup_days: int = 30

@dataclass
class LoggingConfig:
    """Logging configuration"""
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    console_output: bool = True
    file_output: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    json_logging: bool = True
    log_analytics: bool = True
    alert_on_errors: bool = True
    error_threshold: int = 10

@dataclass
class AssistantConfig:
    """Main assistant configuration"""
    # Basic settings
    name: str = "মায়া"
    version: str = "2.0.0"
    author: str = "Maya Assistant Team"
    mode: RunMode = RunMode.CLI
    language: str = "bn"
    
    # Component configurations
    brain: BrainConfig = field(default_factory=BrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    internet: InternetConfig = field(default_factory=InternetConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Behavior settings
    auto_start: bool = True
    speak_responses: bool = True
    listen_on_startup: bool = True
    personality: str = "friendly"  # friendly, professional, humorous, assistant
    response_style: str = "detailed"  # brief, detailed, creative
    
    # Paths
    data_dir: str = "data"
    logs_dir: str = "logs"
    models_dir: str = "models"
    temp_dir: str = "temp"
    
    # API Keys (for production, these should be in .env)
    openweather_api_key: str = ""
    newsapi_key: str = ""
    google_api_key: str = ""
    google_translate_key: str = ""
    
    # Security
    require_auth: bool = False
    auth_password: str = ""
    session_timeout: int = 3600
    
    # Performance
    max_threads: int = 10
    max_queue_size: int = 100
    enable_caching: bool = True
    
    # Updates
    check_for_updates: bool = True
    auto_install_updates: bool = False

class Config:
    """Configuration manager"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or create default"""
        config_paths = [
            self.config_file,
            "config.yaml",
            "config.yml",
            os.path.join(os.path.dirname(__file__), "config.json")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        if config_path.endswith(('.yaml', '.yml')):
                            config_data = yaml.safe_load(f)
                        else:
                            config_data = json.load(f)
                    
                    # Convert to AssistantConfig
                    self.config = self._dict_to_config(config_data)
                    print(f"✅ কনফিগারেশন লোড করা হয়েছে: {config_path}")
                    return
                    
                except Exception as e:
                    print(f"⚠️ কনফিগারেশন লোড করতে সমস্যা ({config_path}): {e}")
        
        # Create default config
        print("⚠️ কোন কনফিগারেশন ফাইল পাওয়া যায়নি, ডিফল্ট ব্যবহার করা হচ্ছে")
        self.config = AssistantConfig()
        self._save_config()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AssistantConfig:
        """Convert dictionary to AssistantConfig"""
        # Handle nested dataclasses
        brain_config = BrainConfig(**config_dict.get('brain', {}))
        memory_config = MemoryConfig(**config_dict.get('memory', {}))
        voice_config_dict = config_dict.get('voice', {})
        if 'gender' in voice_config_dict:
            voice_config_dict['gender'] = VoiceGender(voice_config_dict['gender'])
        voice_config = VoiceConfig(**voice_config_dict)
        vision_config = VisionConfig(**config_dict.get('vision', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        internet_config = InternetConfig(**config_dict.get('internet', {}))
        storage_config = StorageConfig(**config_dict.get('storage', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Handle mode
        mode_str = config_dict.get('mode', 'cli')
        try:
            mode = RunMode(mode_str)
        except:
            mode = RunMode.CLI
        
        # Create main config
        config = AssistantConfig(
            name=config_dict.get('name', 'মায়া'),
            version=config_dict.get('version', '2.0.0'),
            author=config_dict.get('author', 'Maya Assistant Team'),
            mode=mode,
            language=config_dict.get('language', 'bn'),
            brain=brain_config,
            memory=memory_config,
            voice=voice_config,
            vision=vision_config,
            system=system_config,
            internet=internet_config,
            storage=storage_config,
            logging=logging_config,
            auto_start=config_dict.get('auto_start', True),
            speak_responses=config_dict.get('speak_responses', True),
            listen_on_startup=config_dict.get('listen_on_startup', True),
            personality=config_dict.get('personality', 'friendly'),
            response_style=config_dict.get('response_style', 'detailed'),
            data_dir=config_dict.get('data_dir', 'data'),
            logs_dir=config_dict.get('logs_dir', 'logs'),
            models_dir=config_dict.get('models_dir', 'models'),
            temp_dir=config_dict.get('temp_dir', 'temp'),
            openweather_api_key=config_dict.get('openweather_api_key', ''),
            newsapi_key=config_dict.get('newsapi_key', ''),
            google_api_key=config_dict.get('google_api_key', ''),
            google_translate_key=config_dict.get('google_translate_key', ''),
            require_auth=config_dict.get('require_auth', False),
            auth_password=config_dict.get('auth_password', ''),
            session_timeout=config_dict.get('session_timeout', 3600),
            max_threads=config_dict.get('max_threads', 10),
            max_queue_size=config_dict.get('max_queue_size', 100),
            enable_caching=config_dict.get('enable_caching', True),
            check_for_updates=config_dict.get('check_for_updates', True),
            auto_install_updates=config_dict.get('auto_install_updates', False)
        )
        
        return config
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            # Convert enums to strings
            config_dict['mode'] = self.config.mode.value
            config_dict['voice']['gender'] = self.config.voice.gender.value
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            print(f"✅ কনফিগারেশন সংরক্ষণ করা হয়েছে: {self.config_file}")
            
        except Exception as e:
            print(f"❌ কনফিগারেশন সংরক্ষণ করা যায়নি: {e}")
    
    def load(self) -> Dict[str, Any]:
        """Load configuration as dictionary"""
        if self.config is None:
            self._load_config()
        
        # Convert to dictionary
        config_dict = asdict(self.config)
        
        # Convert enums
        config_dict['mode'] = self.config.mode.value
        config_dict['voice']['gender'] = self.config.voice.gender.value
        
        return config_dict
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration"""
        try:
            # Convert updates to config structure
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    # Check nested configs
                    for nested_attr in ['brain', 'memory', 'voice', 'vision', 
                                       'system', 'internet', 'storage', 'logging']:
                        if hasattr(getattr(self.config, nested_attr), key):
                            setattr(getattr(self.config, nested_attr), key, value)
                            break
            
            # Save updated config
            self._save_config()
            
            print("✅ কনফিগারেশন আপডেট করা হয়েছে")
            
        except Exception as e:
            print(f"❌ কনফিগারেশন আপডেট করা যায়নি: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            # Check main config
            if hasattr(self.config, key):
                return getattr(self.config, key)
            
            # Check nested configs
            for nested_attr in ['brain', 'memory', 'voice', 'vision', 
                               'system', 'internet', 'storage', 'logging']:
                nested_config = getattr(self.config, nested_attr)
                if hasattr(nested_config, key):
                    return getattr(nested_config, key)
            
            return default
            
        except:
            return default
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
    
    def validate(self) -> List[str]:
        """Validate configuration and return errors"""
        errors = []
        
        # Check required API keys if features are enabled
        if self.config.internet.enabled:
            if self.config.openweather_api_key and not self.config.openweather_api_key.startswith(('http', 'valid_')):
                errors.append("অবৈধ OpenWeatherMap API কী")
        
        # Check directories
        directories = [self.config.data_dir, self.config.logs_dir, 
                      self.config.models_dir, self.config.temp_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except:
                    errors.append(f"ডিরেক্টরি তৈরি করা যায়নি: {directory}")
        
        # Check resource limits
        limits = self.config.system.resource_limits
        if limits['cpu_limit'] > 100 or limits['cpu_limit'] < 1:
            errors.append("CPU লিমিট ১-১০০% এর মধ্যে হতে হবে")
        
        return errors

# Singleton instance
_config_instance = None

def get_config() -> Config:
    """Get configuration singleton"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance