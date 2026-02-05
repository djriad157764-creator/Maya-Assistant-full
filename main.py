"""
Maya Assistant - Main Application Entry Point
Advanced AI Assistant with Bengali Voice Interface
"""
import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import signal
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from core.brain import AICoreBrain
from core.memory import AdvancedMemorySystem
from core.teacher import AITeacherSystem
from system.control import SystemController, InternetController
from voice.speech import VoiceRecognitionEngine, BanglaTTS, VoiceCommandProcessor
from vision.camera import AdvancedVision
from data.storage import AdvancedStorage
from logs.logger import AdvancedLogger

class MayaAssistant:
    """Main AI Assistant Class"""
    
    def __init__(self):
        print("""
        ╔══════════════════════════════════════════════════════════════╗
        ║                   🌟 মায়া সহকারী 🌟                        ║
        ║           Advanced AI Assistant with Bengali Voice           ║
        ║                                                              ║
        ║                Version: 2.0.0 | Ultra Pro Max                ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Load configuration
        print("⚙️ কনফিগারেশন লোড করা হচ্ছে...")
        self.config = Config().load()
        
        # Initialize logging
        print("📝 লগিং সিস্টেম শুরু করা হচ্ছে...")
        self.logger = AdvancedLogger(self.config)
        self.logger.info("মায়া সহকারী শুরু হচ্ছে", module="main")
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Status tracking
        self.running = True
        self.current_mode = "normal"  # normal, quiet, active
        
        print("✅ মায়া সহকারী প্রস্তুত! কথা বলা শুরু করতে 'হ্যালো মায়া' বলুন")
        self.logger.info("মায়া সহকারী সফলভাবে শুরু হয়েছে", module="main")
    
    def _initialize_components(self):
        """Initialize all AI components"""
        try:
            # Initialize core components
            print("🧠 মস্তিষ্ক প্রস্তুত করা হচ্ছে...")
            self.brain = AICoreBrain(self.config)
            
            print("💾 মেমরি সিস্টেম শুরু করা হচ্ছে...")
            self.memory = AdvancedMemorySystem(self.config)
            
            print("👨‍🏫 শিক্ষক সিস্টেম শুরু করা হচ্ছে...")
            self.teacher = AITeacherSystem(self.brain, self.config)
            
            # Initialize system components
            print("🖥️ সিস্টেম কন্ট্রোলার শুরু করা হচ্ছে...")
            self.system = SystemController(self.config)
            
            print("🌐 ইন্টারনেট কন্ট্রোলার শুরু করা হচ্ছে...")
            self.internet = InternetController(self.config)
            
            # Initialize voice components
            print("🎤 ভয়েস সিস্টেম শুরু করা হচ্ছে...")
            self.voice_recognition = VoiceRecognitionEngine(self.config)
            self.tts = BanglaTTS(self.config)
            self.voice_processor = VoiceCommandProcessor(self.config)
            
            # Register voice callbacks
            self.voice_recognition.register_callback(self._on_voice_command)
            
            # Initialize vision components
            print("👁️ ভিশন সিস্টেম শুরু করা হচ্ছে...")
            self.vision = AdvancedVision(self.config)
            
            # Initialize storage
            print("💾 স্টোরেজ সিস্টেম শুরু করা হচ্ছে...")
            self.storage = AdvancedStorage(self.config)
            
            # Start voice listening
            self.voice_recognition.start_listening(language="bn")
            
            # Greet user
            self._greet_user()
            
            self.logger.info("সব কম্পোনেন্ট সফলভাবে শুরু হয়েছে", module="main")
            
        except Exception as e:
            self.logger.error(f"কম্পোনেন্ট শুরু করতে ব্যর্থ: {e}", module="main")
            print(f"❌ ত্রুটি: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⚠️ শাটডাউন সংকেত পাওয়া গেছে ({signum})...")
        self.shutdown()
    
    def _greet_user(self):
        """Greet the user on startup"""
        try:
            # Get current time for appropriate greeting
            current_hour = datetime.now().hour
            
            if 5 <= current_hour < 12:
                greeting = "শুভ সকাল"
            elif 12 <= current_hour < 16:
                greeting = "শুভ অপরাহ্ন"
            elif 16 <= current_hour < 20:
                greeting = "শুভ সন্ধ্যা"
            else:
                greeting = "শুভ রাত্রি"
            
            greeting_message = f"{greeting}! আমি মায়া, আপনার ব্যক্তিগত সহকারী। কীভাবে সাহায্য করতে পারি?"
            
            # Speak greeting
            self.tts.speak(greeting_message, emotion="happy")
            
            self.logger.info(f"ব্যবহারকারীকে অভিবাদন জানানো হয়েছে: {greeting}", module="main")
            
        except Exception as e:
            self.logger.error(f"অভিবাদন জানাতে ব্যর্থ: {e}", module="main")
    
    def _on_voice_command(self, voice_command):
        """Handle incoming voice commands"""
        try:
            self.logger.log_interaction(
                user_input=voice_command.text,
                ai_response="",
                confidence=voice_command.confidence,
                emotion=voice_command.emotion or "neutral"
            )
            
            # Process command
            processed = self.voice_processor.process_command(voice_command)
            
            # Send to brain for processing
            brain_response = self.brain.process_input(
                voice_command.text,
                context={
                    "emotion": voice_command.emotion,
                    "confidence": voice_command.confidence,
                    "language": voice_command.language
                }
            )
            
            # Generate response
            response = self._generate_response(brain_response, processed)
            
            # Speak response
            self.tts.speak(response, emotion=voice_command.emotion)
            
            # Log interaction
            self.logger.log_interaction(
                user_input=voice_command.text,
                ai_response=response,
                confidence=brain_response.get("confidence", 0.0),
                emotion=voice_command.emotion or "neutral"
            )
            
            # Perform actions if needed
            self._perform_actions(brain_response, processed)
            
        except Exception as e:
            error_msg = f"কমান্ড প্রসেস করতে ব্যর্থ: {e}"
            self.logger.error(error_msg, module="main")
            self.tts.speak("দুঃখিত, কিছু সমস্যা হয়েছে। আবার চেষ্টা করুন।")
    
    def _generate_response(self, brain_response: Dict[str, Any], 
                          processed_command: Dict[str, Any]) -> str:
        """Generate appropriate response"""
        try:
            # Get decision from brain
            decision = brain_response.get("decision")
            if not decision:
                return "দুঃখিত, বুঝতে পারিনি। আবার বলবেন?"
            
            # Generate response based on decision
            action = decision.action
            
            response_templates = {
                "প্রশ্নের উত্তর দিন": "আমি উত্তর দিচ্ছি...",
                "স্পষ্ট করে জিজ্ঞাসা করুন": "আপনি কি স্পষ্ট করে বলবেন?",
                "কাজটি সম্পাদন করুন": "ঠিক আছে, কাজটি করছি...",
                "তথ্য অনুসন্ধান করুন": "এক মুহূর্ত, তথ্য খুঁজছি...",
                "ভাবপ্রবণ উত্তর দিন": "আপনার অনুভূতি আমি বুঝতে পেরেছি...",
                "সৃজনশীল উত্তর দিন": "একটা মজার উত্তর দিচ্ছি...",
                "বিষয় পরিবর্তন করুন": "চলুন অন্য কিছু নিয়ে কথা বলি...",
                "মজাদার উত্তর দিন": "হাসির জন্য তৈরি? শুনুন..."
            }
            
            response = response_templates.get(
                action, 
                "আপনি কি অন্য কিছু বলতে চান?"
            )
            
            # Add specific information if available
            if "entities" in processed_command:
                entities = processed_command["entities"]
                
                if "app_name" in entities:
                    response = f"{entities['app_name']} {response}"
                elif "query" in entities:
                    response = f"{response} '{entities['query']}'"
            
            return response
            
        except Exception as e:
            self.logger.error(f"রেসপন্স জেনারেট করতে ব্যর্থ: {e}", module="main")
            return "দুঃখিত, এখন উত্তর দিতে পারছি না।"
    
    def _perform_actions(self, brain_response: Dict[str, Any], 
                        processed_command: Dict[str, Any]):
        """Perform actions based on decision"""
        try:
            decision = brain_response.get("decision")
            if not decision:
                return
            
            action = decision.action
            entities = processed_command.get("entities", {})
            
            if action == "কাজটি সম্পাদন করুন":
                if "app_name" in entities:
                    app_name = entities["app_name"]
                    self.system.open_application(app_name)
                    
                    # Log action
                    self.logger.info(
                        f"অ্যাপ্লিকেশন খোলা হয়েছে: {app_name}",
                        module="system",
                        extra={"app_name": app_name}
                    )
            
            elif action == "তথ্য অনুসন্ধান করুন":
                if "query" in entities:
                    query = entities["query"]
                    self.internet.search_web(query, engine="google")
                    
                    # Log action
                    self.logger.info(
                        f"অনুসন্ধান করা হয়েছে: {query}",
                        module="internet",
                        extra={"query": query}
                    )
            
            elif action == "সময় বল":
                current_time = datetime.now().strftime("%I:%M %p")
                self.tts.speak(f"এখন সময় {current_time}")
            
            elif action == "তারিখ বল":
                current_date = datetime.now().strftime("%d %B, %Y")
                self.tts.speak(f"আজকের তারিখ {current_date}")
            
        except Exception as e:
            self.logger.error(f"একশন পারফর্ম করতে ব্যর্থ: {e}", module="main")
    
    def run_command_line(self):
        """Run in command line mode"""
        print("\n" + "="*60)
        print("কমান্ড লাইন মোড - টাইপ 'exit' বা 'quit' দিয়ে বের হন")
        print("="*60 + "\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("আপনি: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'বের হন']:
                    print("বিদায়!")
                    self.shutdown()
                    break
                
                if not user_input:
                    continue
                
                # Process input
                brain_response = self.brain.process_input(user_input)
                
                # Generate and display response
                if brain_response and "decision" in brain_response:
                    decision = brain_response["decision"]
                    print(f"মায়া: {decision.reasoning}")
                    
                    # Speak if requested
                    if self.config.get("speak_responses", True):
                        self.tts.speak(decision.reasoning)
                else:
                    print("মায়া: দুঃখিত, বুঝতে পারিনি।")
                
            except KeyboardInterrupt:
                print("\nবিদায়!")
                self.shutdown()
                break
            except Exception as e:
                print(f"ত্রুটি: {e}")
                self.logger.error(f"কমান্ড লাইন ত্রুটি: {e}", module="main")
    
    def run_interactive(self):
        """Run in interactive mode with GUI (placeholder)"""
        print("ইন্টারঅ্যাকটিভ মোড - শীঘ্রই আসছে...")
        # TODO: Implement GUI interface
        self.run_command_line()
    
    def run_background(self):
        """Run in background mode"""
        print("ব্যাকগ্রাউন্ড মোড - মায়া শুনছে...")
        
        try:
            while self.running:
                # Check for voice commands in queue
                voice_command = self.voice_recognition.get_next_command(timeout=1)
                
                if voice_command:
                    # Process immediately in main thread
                    self._on_voice_command(voice_command)
                
                # Sleep to prevent CPU overuse
                import time
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nব্যাকগ্রাউন্ড মোড বন্ধ করা হচ্ছে...")
        except Exception as e:
            self.logger.error(f"ব্যাকগ্রাউন্ড মোড ত্রুটি: {e}", module="main")
    
    def shutdown(self):
        """Graceful shutdown"""
        print("\n🛑 মায়া সহকারী বন্ধ করা হচ্ছে...")
        self.running = False
        
        try:
            # Stop voice recognition
            self.voice_recognition.stop_listening()
            
            # Save all data
            self._save_all_data()
            
            # Close components
            if hasattr(self, 'vision') and self.vision.camera:
                self.vision.camera.release()
            
            # Log shutdown
            self.logger.info("মায়া সহকারী সফলভাবে বন্ধ হয়েছে", module="main")
            
            print("✅ সবকিছু সেভ করা হয়েছে। বিদায়!")
            
        except Exception as e:
            self.logger.error(f"শাটডাউন ত্রুটি: {e}", module="main")
            print(f"❌ শাটডাউন ত্রুটি: {e}")
        
        sys.exit(0)
    
    def _save_all_data(self):
        """Save all data before shutdown"""
        try:
            print("💾 সব ডেটা সংরক্ষণ করা হচ্ছে...")
            
            # Save brain knowledge
            if hasattr(self, 'brain'):
                self.brain._save_knowledge_base()
            
            # Save memory
            if hasattr(self, 'memory'):
                self.memory.consolidate()
            
            # Backup data
            if hasattr(self, 'storage'):
                self.storage.backup_data()
            
            print("✅ সব ডেটা সংরক্ষণ করা হয়েছে")
            
        except Exception as e:
            print(f"❌ ডেটা সংরক্ষণ করা যায়নি: {e}")
            self.logger.error(f"ডেটা সংরক্ষণ ত্রুটি: {e}", module="main")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            "running": self.running,
            "mode": self.current_mode,
            "components": {
                "brain": hasattr(self, 'brain'),
                "memory": hasattr(self, 'memory'),
                "voice": hasattr(self, 'voice_recognition') and self.voice_recognition.is_listening,
                "vision": hasattr(self, 'vision') and self.vision.camera is not None,
                "system": hasattr(self, 'system'),
                "internet": hasattr(self, 'internet') and self.internet._check_internet()
            },
            "timestamp": datetime.now().isoformat(),
            "uptime": self._get_uptime()
        }
        
        return status
    
    def _get_uptime(self) -> str:
        """Get uptime as string"""
        if not hasattr(self, '_start_time'):
            self._start_time = datetime.now()
        
        uptime = datetime.now() - self._start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{hours}ঘণ্টা {minutes}মিনিট {seconds}সেকেন্ড"

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='মায়া সহকারী - Advanced AI Assistant')
    parser.add_argument('--mode', '-m', choices=['cli', 'gui', 'background'], 
                       default='cli', help='রান মোড (cli, gui, background)')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='শান্ত মোড (কথা বলবে না)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='লগ লেভেল')
    
    args = parser.parse_args()
    
    try:
        # Create and run assistant
        assistant = MayaAssistant()
        
        # Set mode based on arguments
        if args.quiet:
            assistant.current_mode = "quiet"
            print("শান্ত মোড: শুধুমাত্র টেক্সট আউটপুট")
        
        # Set log level
        if args.log_level:
            # This would be implemented in logger configuration
            print(f"লগ লেভেল: {args.log_level}")
        
        # Run in selected mode
        if args.mode == 'cli':
            assistant.run_command_line()
        elif args.mode == 'gui':
            assistant.run_interactive()
        elif args.mode == 'background':
            assistant.run_background()
        
    except KeyboardInterrupt:
        print("\n\nবিদায়!")
        if 'assistant' in locals():
            assistant.shutdown()
    except Exception as e:
        print(f"\n❌ গুরুতর ত্রুটি: {e}")
        traceback.print_exc()
        
        # Try to log error if logger exists
        if 'assistant' in locals() and hasattr(assistant, 'logger'):
            assistant.logger.critical(f"গুরুতর ত্রুটি: {e}", module="main")
        
        sys.exit(1)

if __name__ == "__main__":
    # Ensure proper encoding for Bengali
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'bn_BD.UTF-8')
    except:
        pass
    
    # Run the assistant
    main()