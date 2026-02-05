#!/usr/bin/env python3
"""
Maya Assistant - Automated Setup Script
5-minute installation and configuration
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import json
import urllib.request
import zipfile
import tarfile
from typing import Dict, List, Optional, Tuple
import argparse

class SetupAssistant:
    """Automated setup assistant for Maya AI"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.config = {}
        
        # Colors for terminal output
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.BLUE = '\033[94m'
        self.END = '\033[0m'
        self.BOLD = '\033[1m'
    
    def print_header(self):
        """Print setup header"""
        print(f"""
{self.BOLD}{self.BLUE}
╔══════════════════════════════════════════════════════════════╗
║                   🌟 মায়া সহকারী সেটআপ 🌟                  ║
║           Advanced AI Assistant Installation Wizard          ║
║                                                              ║
║                Version: 2.0.0 | Ultra Pro Max                ║
╚══════════════════════════════════════════════════════════════╝
{self.END}""")
    
    def check_requirements(self) -> bool:
        """Check system requirements"""
        print(f"\n{self.BOLD}📋 প্রয়োজনীয়তা পরীক্ষা করা হচ্ছে...{self.END}")
        
        requirements = [
            ("Python 3.8+", self.python_version >= (3, 8), 
             f"পাওয়া গেছে: Python {self.python_version.major}.{self.python_version.minor}"),
            ("OS", self.system in ["Windows", "Linux", "Darwin"], 
             f"অপারেটিং সিস্টেম: {self.system}"),
            ("Disk Space", self.check_disk_space(), 
             "ডিস্ক স্পেস: পর্যাপ্ত"),
            ("RAM", self.check_ram(), 
             "RAM: কমপক্ষে 4GB পাওয়া গেছে")
        ]
        
        all_passed = True
        for name, check, message in requirements:
            if check:
                print(f"   {self.GREEN}✓{self.END} {name}: {message}")
            else:
                print(f"   {self.RED}✗{self.END} {name}: প্রয়োজনীয়তা পূরণ হয়নি")
                all_passed = False
        
        return all_passed
    
    def check_disk_space(self, min_gb: int = 2) -> bool:
        """Check available disk space"""
        try:
            if self.system == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(str(self.project_root)), 
                    None, None, ctypes.pointer(free_bytes)
                )
                free_gb = free_bytes.value / (1024**3)
            else:
                stat = os.statvfs(str(self.project_root))
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            return free_gb >= min_gb
        except:
            return True  # If we can't check, assume it's okay
    
    def check_ram(self, min_gb: int = 4) -> bool:
        """Check available RAM"""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            return ram_gb >= min_gb
        except:
            return True  # If we can't check, assume it's okay
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        print(f"\n{self.BOLD}🐍 ভার্চুয়াল এনভায়রনমেন্ট তৈরি করা হচ্ছে...{self.END}")
        
        if self.venv_path.exists():
            response = input(f"{self.YELLOW}⚠️ ভার্চুয়াল এনভায়রনমেন্ট ইতিমধ্যে আছে। পুনঃনির্মাণ করবেন? (y/N): {self.END}")
            if response.lower() != 'y':
                return True
            
            try:
                shutil.rmtree(self.venv_path)
                print(f"{self.GREEN}   পুরাতন এনভায়রনমেন্ট মুছে ফেলা হয়েছে{self.END}")
            except Exception as e:
                print(f"{self.RED}   পুরাতন এনভায়রনমেন্ট মুছতে সমস্যা: {e}{self.END}")
                return False
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], 
                          check=True, capture_output=True)
            print(f"{self.GREEN}   ✓ ভার্চুয়াল এনভায়রনমেন্ট তৈরি করা হয়েছে{self.END}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{self.RED}   ✗ ভার্চুয়াল এনভায়রনমেন্ট তৈরি করা যায়নি: {e}{self.END}")
            print(f"{self.YELLOW}   বিকল্প: python -m venv venv কমান্ডটি ম্যানুয়ালি চালান{self.END}")
            return False
    
    def get_pip_path(self) -> Path:
        """Get pip executable path"""
        if self.system == "Windows":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def get_python_path(self) -> Path:
        """Get python executable path"""
        if self.system == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def install_dependencies(self, minimal: bool = False) -> bool:
        """Install Python dependencies"""
        print(f"\n{self.BOLD}📦 প্যাকেজ ইনস্টল করা হচ্ছে...{self.END}")
        
        pip_path = self.get_pip_path()
        
        # Upgrade pip first
        try:
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            print(f"{self.GREEN}   ✓ pip আপগ্রেড করা হয়েছে{self.END}")
        except:
            print(f"{self.YELLOW}   ⚠️ pip আপগ্রেড করা যায়নি, চালিয়ে যাচ্ছি...{self.END}")
        
        # Install requirements
        requirements_file = "requirements-minimal.txt" if minimal else "requirements.txt"
        
        if not Path(requirements_file).exists():
            # Create minimal requirements if file doesn't exist
            self.create_minimal_requirements()
        
        try:
            cmd = [str(pip_path), "install", "-r", requirements_file]
            if minimal:
                cmd.append("--no-deps")
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                print(f"{self.GREEN}   ✓ সব প্যাকেজ ইনস্টল করা হয়েছে{self.END}")
                return True
            else:
                print(f"{self.RED}   ✗ প্যাকেজ ইনস্টলেশন ব্যর্থ{self.END}")
                print(f"{self.YELLOW}   Error: {process.stderr[:200]}...{self.END}")
                
                # Try alternative installation method
                return self.install_dependencies_alternative(minimal)
                
        except Exception as e:
            print(f"{self.RED}   ✗ প্যাকেজ ইনস্টলেশন ব্যর্থ: {e}{self.END}")
            return False
    
    def install_dependencies_alternative(self, minimal: bool) -> bool:
        """Alternative dependency installation method"""
        print(f"{self.YELLOW}   বিকল্প ইনস্টলেশন পদ্ধতি চেষ্টা করা হচ্ছে...{self.END}")
        
        pip_path = self.get_pip_path()
        core_packages = [
            "torch", "transformers", "numpy", "opencv-python",
            "SpeechRecognition", "pyttsx3", "gTTS", "requests"
        ]
        
        if minimal:
            core_packages = core_packages[:4]  # Just core packages
        
        success_count = 0
        for package in core_packages:
            try:
                subprocess.run([str(pip_path), "install", package], 
                              check=True, capture_output=True)
                print(f"{self.GREEN}   ✓ {package} ইনস্টল করা হয়েছে{self.END}")
                success_count += 1
            except:
                print(f"{self.YELLOW}   ⚠️ {package} ইনস্টল করা যায়নি{self.END}")
        
        return success_count >= len(core_packages) // 2  # At least half should succeed
    
    def create_minimal_requirements(self):
        """Create minimal requirements file"""
        minimal_req = """# Minimal requirements for Maya Assistant
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
opencv-python>=4.8.0
SpeechRecognition>=3.10.0
pyttsx3>=2.90
gTTS>=2.3.0
requests>=2.31.0
python-dotenv>=1.0.0
"""
        
        with open("requirements-minimal.txt", "w", encoding="utf-8") as f:
            f.write(minimal_req)
    
    def setup_configuration(self) -> bool:
        """Setup configuration files"""
        print(f"\n{self.BOLD}⚙️ কনফিগারেশন ফাইল সেটআপ করা হচ্ছে...{self.END}")
        
        # Create directories
        directories = ["data", "logs", "models", "temp", "voice/samples", 
                      "voice/profiles", "vision/known_faces", "downloads"]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"{self.GREEN}   ✓ ডিরেক্টরি তৈরি করা হয়েছে: {directory}{self.END}")
        
        # Create config.json if doesn't exist
        config_path = self.project_root / "config.json"
        if not config_path.exists():
            default_config = {
                "name": "মায়া",
                "version": "2.0.0",
                "mode": "cli",
                "language": "bn",
                "voice": {
                    "language": "bn",
                    "gender": "female",
                    "speech_rate": 150,
                    "volume": 0.9
                },
                "brain": {
                    "model_name": "banglabert",
                    "reasoning_enabled": True,
                    "learning_enabled": True
                },
                "auto_start": True,
                "speak_responses": True
            }
            
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            
            print(f"{self.GREEN}   ✓ কনফিগারেশন ফাইল তৈরি করা হয়েছে{self.END}")
        
        # Create .env file if doesn't exist
        env_path = self.project_root / ".env"
        if not env_path.exists():
            env_template = """# Maya Assistant Environment Configuration

# Application Settings
APP_NAME="Maya Assistant"
APP_VERSION="2.0.0"
APP_ENV="development"

# API Keys (Get these from respective websites)
OPENWEATHER_API_KEY=""
NEWSAPI_KEY=""
GOOGLE_API_KEY=""
GOOGLE_TRANSLATE_KEY=""

# Voice Configuration
VOICE_LANGUAGE="bn"
VOICE_GENDER="female"
VOICE_RATE="150"

# Security Settings
ENCRYPTION_KEY="change-this-in-production"

# Logging Settings
LOG_LEVEL="INFO"
"""
            
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_template)
            
            print(f"{self.GREEN}   ✓ .env ফাইল তৈরি করা হয়েছে{self.END}")
            print(f"{self.YELLOW}   ⚠️ দয়া করে .env ফাইল এডিট করে API কী যোগ করুন{self.END}")
        
        return True
    
    def download_models(self) -> bool:
        """Download necessary AI models"""
        print(f"\n{self.BOLD}🤖 AI মডেল ডাউনলোড করা হচ্ছে...{self.END}")
        
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # List of models to download
        models = [
            {
                "name": "BanglaBERT",
                "url": "https://huggingface.co/csebuetnlp/banglabert/resolve/main/pytorch_model.bin",
                "path": models_dir / "banglabert" / "pytorch_model.bin",
                "optional": False
            }
        ]
        
        success_count = 0
        for model in models:
            model_path = model["path"]
            
            if model_path.exists():
                print(f"{self.GREEN}   ✓ {model['name']} ইতিমধ্যে আছে{self.END}")
                success_count += 1
                continue
            
            try:
                print(f"{self.BLUE}   📥 {model['name']} ডাউনলোড হচ্ছে...{self.END}")
                
                # Create directory
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                urllib.request.urlretrieve(model["url"], model_path)
                
                print(f"{self.GREEN}   ✓ {model['name']} ডাউনলোড সম্পন্ন{self.END}")
                success_count += 1
                
            except Exception as e:
                if model["optional"]:
                    print(f"{self.YELLOW}   ⚠️ {model['name']} ডাউনলোড করা যায়নি (ঐচ্ছিক){self.END}")
                else:
                    print(f"{self.RED}   ✗ {model['name']} ডাউনলোড করা যায়নি: {e}{self.END}")
        
        # Create model config file
        config_file = models_dir / "model_config.json"
        if not config_file.exists():
            model_config = {
                "banglabert": {
                    "path": "models/banglabert",
                    "type": "language_model",
                    "language": "bn"
                }
            }
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(model_config, f, indent=2)
        
        return success_count > 0
    
    def test_installation(self) -> bool:
        """Test if installation was successful"""
        print(f"\n{self.BOLD}🧪 ইনস্টলেশন টেস্ট করা হচ্ছে...{self.END}")
        
        python_path = self.get_python_path()
        
        test_script = """
import sys
print("Python version:", sys.version)

try:
    import torch
    print("✓ PyTorch installed:", torch.__version__)
except ImportError:
    print("✗ PyTorch not installed")

try:
    import transformers
    print("✓ Transformers installed:", transformers.__version__)
except ImportError:
    print("✗ Transformers not installed")

try:
    import speech_recognition as sr
    print("✓ SpeechRecognition installed")
except ImportError:
    print("✗ SpeechRecognition not installed")

print("\\nইনস্টলেশন টেস্ট সম্পন্ন!")
"""
        
        try:
            result = subprocess.run(
                [str(python_path), "-c", test_script],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            if result.returncode == 0:
                print(f"{self.GREEN}   ✓ ইনস্টলেশন টেস্ট পাস{self.END}")
                return True
            else:
                print(f"{self.RED}   ✗ ইনস্টলেশন টেস্ট ব্যর্থ{self.END}")
                return False
                
        except Exception as e:
            print(f"{self.RED}   ✗ টেস্ট চলতে সমস্যা: {e}{self.END}")
            return False
    
    def create_startup_scripts(self):
        """Create startup scripts for easy launch"""
        print(f"\n{self.BOLD}🚀 স্টার্টআপ স্ক্রিপ্ট তৈরি করা হচ্ছে...{self.END}")
        
        python_path = self.get_python_path()
        project_dir = str(self.project_root)
        
        # Windows batch file
        if self.system == "Windows":
            bat_content = f"""@echo off
echo মায়া সহকারী শুরু হচ্ছে...
cd /d "{project_dir}"
"{python_path}" main.py %*
pause
"""
            
            bat_path = self.project_root / "start_maya.bat"
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(bat_content)
            
            print(f"{self.GREEN}   ✓ Windows batch file তৈরি করা হয়েছে{self.END}")
        
        # Linux/macOS shell script
        else:
            sh_content = f"""#!/bin/bash
echo "মায়া সহকারী শুরু হচ্ছে..."
cd "{project_dir}"
"{python_path}" main.py "$@"
"""
            
            sh_path = self.project_root / "start_maya.sh"
            with open(sh_path, "w", encoding="utf-8") as f:
                f.write(sh_content)
            
            # Make executable
            os.chmod(sh_path, 0o755)
            
            print(f"{self.GREEN}   ✓ Shell script তৈরি করা হয়েছে{self.END}")
    
    def show_completion_message(self):
        """Show completion message"""
        print(f"""
{self.BOLD}{self.GREEN}
╔══════════════════════════════════════════════════════════════╗
║                    🎉 ইনস্টলেশন সম্পূর্ণ! 🎉                ║
║                                                              ║
║           মায়া সহকারী সফলভাবে ইনস্টল করা হয়েছে            ║
╚══════════════════════════════════════════════════════════════╝
{self.END}

{self.BOLD}পরবর্তী ধাপসমূহ:{self.END}

1. {self.BLUE}সহকারী শুরু করুন:{self.END}
   {self.YELLOW}python main.py{self.END}

2. {self.BLUE}API কী সেট আপ করুন:{self.END}
   {self.YELLOW}ফাইল: .env এডিট করুন{self.END}

3. {self.BLUE}ভয়েস টেস্ট করুন:{self.END}
   {self.YELLOW}বলুন: "হ্যালো মায়া"{self.END}

{self.BOLD}দ্রুত রেফারেন্স:{self.END}

• {self.GREEN}start_maya.bat{self.END} (Windows) বা {self.GREEN}start_maya.sh{self.END} (Linux/macOS)
• {self.YELLOW}--help{self.END} ফ্লাগ ব্যবহার করে সব অপশন দেখুন
• {self.YELLOW}--mode background{self.END} ব্যাকগ্রাউন্ড মোডে চালান

{self.BOLD}সাহায্যের জন্য:{self.END}

• GitHub: https://github.com/yourusername/maya-assistant
• ডকুমেন্টেশন: README.md এবং docs/ ফোল্ডার
• ইস্যু রিপোর্ট: GitHub Issues পেজে

{self.YELLOW}⭐ Star দিয়ে আমাদের সাপোর্ট করুন যদি প্রকল্পটি পছন্দ করেন!{self.END}
""")
    
    def cleanup(self):
        """Cleanup temporary files"""
        print(f"\n{self.BOLD}🧹 টেম্পোরারি ফাইল ক্লিনআপ...{self.END}")
        
        # Remove temporary files if they exist
        temp_files = ["setup_cache", "downloads/temp"]
        
        for temp_file in temp_files:
            temp_path = self.project_root / temp_file
            if temp_path.exists():
                try:
                    if temp_path.is_file():
                        temp_path.unlink()
                    else:
                        shutil.rmtree(temp_path)
                    print(f"{self.GREEN}   ✓ {temp_file} মুছে ফেলা হয়েছে{self.END}")
                except:
                    pass
    
    def run(self, minimal: bool = False, reset: bool = False):
        """Run the complete setup process"""
        self.print_header()
        
        # Check requirements
        if not self.check_requirements():
            response = input(f"{self.YELLOW}⚠️ কিছু প্রয়োজনীয়তা পূরণ হয়নি। তবুও চালিয়ে যাবেন? (y/N): {self.END}")
            if response.lower() != 'y':
                print(f"{self.RED}ইনস্টলেশন বাতিল করা হয়েছে।{self.END}")
                return False
        
        # Reset if requested
        if reset:
            self.cleanup()
        
        # Setup steps
        steps = [
            ("ভার্চুয়াল এনভায়রনমেন্ট তৈরি", self.create_virtual_environment),
            ("প্যাকেজ ইনস্টল", lambda: self.install_dependencies(minimal)),
            ("কনফিগারেশন সেটআপ", self.setup_configuration),
            ("AI মডেল ডাউনলোড", self.download_models),
            ("ইনস্টলেশন টেস্ট", self.test_installation),
            ("স্টার্টআপ স্ক্রিপ্ট তৈরি", self.create_startup_scripts)
        ]
        
        failed_steps = []
        for step_name, step_func in steps:
            print(f"\n{self.BOLD}➡️ {step_name}...{self.END}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"{self.RED}   ✗ {step_name} ত্রুটি: {e}{self.END}")
                failed_steps.append(step_name)
        
        # Show results
        if failed_steps:
            print(f"\n{self.RED}{self.BOLD}⚠️ কিছু ধাপ ব্যর্থ হয়েছে:{self.END}")
            for step in failed_steps:
                print(f"   • {step}")
            
            print(f"\n{self.YELLOW}বিকল্প সমাধান:{self.END}")
            print("1. ম্যানুয়ালি প্রয়োজনীয়তা ইনস্টল করুন")
            print("2. requirements-minimal.txt ব্যবহার করুন")
            print("3. GitHub Issues-এ সমস্যা রিপোর্ট করুন")
        else:
            self.show_completion_message()
            self.cleanup()
        
        return len(failed_steps) == 0

def main():
    """Main entry point for setup script"""
    parser = argparse.ArgumentParser(
        description="মায়া সহকারী - অটোমেটেড সেটআপ স্ক্রিপ্ট",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
উদাহরণ:
  python setup.py                     # সম্পূর্ণ ইনস্টলেশন
  python setup.py --minimal          # মিনিমাল ইনস্টলেশন
  python setup.py --reset           # রিসেট এবং পুনঃইনস্টল
  python setup.py --no-models       # মডেল ছাড়া ইনস্টল
  python setup.py --help            # সাহায্য দেখুন
        """
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="মিনিমাল ইনস্টলেশন (শুধুমাত্র প্রয়োজনীয় প্যাকেজ)"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="পুরাতন ইনস্টলেশন রিসেট করুন"
    )
    
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="AI মডেল ডাউনলোড করবেন না"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="শুধুমাত্র ইনস্টলেশন টেস্ট করুন"
    )
    
    args = parser.parse_args()
    
    # Create setup assistant
    assistant = SetupAssistant()
    
    # Run test only if requested
    if args.test_only:
        return assistant.test_installation()
    
    # Run full setup
    success = assistant.run(minimal=args.minimal, reset=args.reset)
    
    # Return exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()