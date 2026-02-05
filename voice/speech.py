"""
Advanced Speech Recognition and Synthesis with Bangla Support
"""
import speech_recognition as sr
import pyttsx3
import torch
import numpy as np
from gtts import gTTS
import pygame
from pygame import mixer
import os
from typing import Dict, List, Any, Optional, Callable
import queue
import threading
import time
from datetime import datetime
import json
import sounddevice as sd
import soundfile as sf
import wave
import librosa
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
from dataclasses import dataclass
from enum import Enum

@dataclass
class VoiceCommand:
    """Voice command with context"""
    text: str
    confidence: float
    timestamp: datetime
    language: str
    speaker_id: Optional[str]
    emotion: Optional[str]

class VoiceRecognitionEngine:
    """Advanced voice recognition engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.is_listening = False
        self.callbacks = []
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        self.pause_threshold = 0.8
        self.phrase_threshold = 0.3
        self.non_speaking_duration = 0.5
        
        # Language models
        self.supported_languages = {
            "bn": "bn-BD",  # Bengali
            "en": "en-US",  # English
            "hi": "hi-IN",  # Hindi
            "ar": "ar-SA",  # Arabic
        }
        
        # Initialize audio devices
        self._initialize_audio_devices()
        
        # Load voice profiles
        self.voice_profiles = self._load_voice_profiles()
        
        # Start background listener
        self.listener_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self.listener_thread.start()
        
        print("🎤 ভয়েস রিকগনিশন ইঞ্জিন প্রস্তুত")
    
    def _initialize_audio_devices(self):
        """Initialize audio devices"""
        print("🔊 অডিও ডিভাইসগুলি সনাক্ত করা হচ্ছে...")
        
        # List available microphones
        mic_list = sr.Microphone.list_microphone_names()
        print(f"🎙️ পাওয়া মাইক্রোফোন: {len(mic_list)}")
        
        for i, mic_name in enumerate(mic_list):
            print(f"  {i}: {mic_name}")
        
        # Configure microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.phrase_threshold = self.phrase_threshold
            self.recognizer.non_speaking_duration = self.non_speaking_duration
        
        print("✅ অডিও ডিভাইস কনফিগার করা হয়েছে")
    
    def start_listening(self, language: str = "bn"):
        """Start continuous listening"""
        if not self.is_listening:
            self.is_listening = True
            self.current_language = language
            print(f"👂 শোনা শুরু করা হয়েছে ({language})...")
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        print("🔇 শোনা বন্ধ করা হয়েছে")
    
    def listen_once(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[VoiceCommand]:
        """Listen for a single command"""
        try:
            with self.microphone as source:
                print("🎤 বলুন...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            # Process audio
            command = self._process_audio(audio, self.current_language)
            return command
            
        except sr.WaitTimeoutError:
            print("⏰ সময় শেষ: কিছু বলা হয়নি")
            return None
        except Exception as e:
            print(f"❌ শোনা যায়নি: {e}")
            return None
    
    def _continuous_listen(self):
        """Background continuous listening"""
        while True:
            if self.is_listening:
                try:
                    command = self.listen_once(timeout=1, phrase_time_limit=5)
                    if command and command.confidence > 0.6:  # Confidence threshold
                        self.command_queue.put(command)
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                callback(command)
                            except Exception as e:
                                print(f"❌ কলব্যাক ত্রুটি: {e}")
                except Exception as e:
                    print(f"❌ ক্রমাগত শোনার ত্রুটি: {e}")
            
            time.sleep(0.1)
    
    def _process_audio(self, audio, language: str = "bn") -> Optional[VoiceCommand]:
        """Process audio and convert to text"""
        try:
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio)
            
            # Convert to text
            text = ""
            confidence = 0.0
            
            if language == "bn":
                # Try Google Speech Recognition for Bangla
                try:
                    text = self.recognizer.recognize_google(
                        processed_audio, 
                        language=self.supported_languages[language],
                        show_all=False
                    )
                    confidence = 0.8  # Google doesn't provide confidence for Bangla
                except sr.UnknownValueError:
                    # Fallback to English
                    try:
                        text = self.recognizer.recognize_google(
                            processed_audio, 
                            language=self.supported_languages["en"]
                        )
                        confidence = 0.6
                    except sr.UnknownValueError:
                        return None
                except sr.RequestError as e:
                    print(f"❌ গুগল স্পিচ সার্ভিস ত্রুটি: {e}")
                    return None
            
            else:
                # For other languages
                try:
                    result = self.recognizer.recognize_google(
                        processed_audio, 
                        language=self.supported_languages.get(language, "en-US"),
                        show_all=True
                    )
                    
                    if isinstance(result, dict) and 'alternative' in result:
                        best_result = result['alternative'][0]
                        text = best_result.get('transcript', '')
                        confidence = best_result.get('confidence', 0.0)
                    else:
                        text = result
                        confidence = 0.7
                except sr.UnknownValueError:
                    return None
                except sr.RequestError as e:
                    print(f"❌ স্পিচ রিকগনিশন সার্ভিস ত্রুটি: {e}")
                    return None
            
            if text:
                # Detect emotion from voice
                emotion = self._detect_emotion(audio)
                
                # Identify speaker if possible
                speaker_id = self._identify_speaker(audio)
                
                command = VoiceCommand(
                    text=text,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    language=language,
                    speaker_id=speaker_id,
                    emotion=emotion
                )
                
                print(f"📝 শোনা হয়েছে: '{text}' (আত্মবিশ্বাস: {confidence:.2%})")
                return command
            
            return None
            
        except Exception as e:
            print(f"❌ অডিও প্রসেসিং ত্রুটি: {e}")
            return None
    
    def _preprocess_audio(self, audio) -> sr.AudioData:
        """Preprocess audio for better recognition"""
        try:
            # Convert to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            sample_rate = audio.sample_rate
            
            # Noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_data.astype(np.float32),
                sr=sample_rate,
                stationary=True
            )
            
            # Normalize
            normalized = self._normalize_audio(reduced_noise)
            
            # Convert back to AudioData
            processed_audio = sr.AudioData(
                normalized.astype(np.int16).tobytes(),
                sample_rate,
                audio.sample_width
            )
            
            return processed_audio
            
        except Exception as e:
            print(f"❌ অডিও প্রিপ্রসেসিং ত্রুটি: {e}")
            return audio
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio volume"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val * 32767
        return audio_data
    
    def _detect_emotion(self, audio) -> Optional[str]:
        """Detect emotion from voice (simplified)"""
        try:
            # Extract features
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Calculate pitch and energy
            pitch = self._extract_pitch(audio_data, audio.sample_rate)
            energy = np.sqrt(np.mean(audio_data**2))
            
            # Simple emotion detection based on features
            if energy > 10000:
                if pitch > 200:
                    return "excited"
                else:
                    return "angry"
            elif energy < 3000:
                return "calm"
            elif pitch > 180:
                return "happy"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"❌ আবেগ সনাক্তকরণ ত্রুটি: {e}")
            return None
    
    def _extract_pitch(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Extract pitch from audio"""
        try:
            # Use librosa for pitch detection
            pitches, magnitudes = librosa.piptrack(
                y=audio_data.astype(np.float32),
                sr=sample_rate
            )
            
            # Get pitch with highest magnitude
            pitch_idx = magnitudes.argmax()
            pitch = pitches.flatten()[pitch_idx]
            
            return float(pitch) if pitch > 0 else 100.0
        except:
            return 100.0  # Default pitch
    
    def _identify_speaker(self, audio) -> Optional[str]:
        """Identify speaker (simplified version)"""
        # In production, use speaker recognition models
        # This is a placeholder
        return None
    
    def _load_voice_profiles(self) -> Dict[str, Any]:
        """Load voice profiles"""
        try:
            with open("voice/profiles.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"❌ ভয়েস প্রোফাইল লোড করা যায়নি: {e}")
            return {}
    
    def register_callback(self, callback: Callable[[VoiceCommand], None]):
        """Register callback for voice commands"""
        self.callbacks.append(callback)
        print(f"✅ কলব্যাক নিবন্ধিত: {callback.__name__}")
    
    def get_next_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """Get next voice command from queue"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def save_audio_sample(self, audio, filename: str = None) -> Optional[str]:
        """Save audio sample to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice/samples/audio_{timestamp}.wav"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(audio.sample_rate)
                wf.writeframes(audio.frame_data)
            
            print(f"💾 অডিও স্যাম্পল সংরক্ষণ করা হয়েছে: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ অডিও সংরক্ষণ করা যায়নি: {e}")
            return None

class BanglaTTS:
    """Advanced Bangla Text-to-Speech with female voice"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = pyttsx3.init()
        self.voice_cache = {}
        self.current_voice = None
        
        # Initialize voices
        self._initialize_voices()
        
        # Initialize pygame mixer for audio playback
        mixer.init()
        
        print("🗣️ বাংলা টেক্সট-টু-স্পিচ প্রস্তুত")
    
    def _initialize_voices(self):
        """Initialize available voices"""
        voices = self.engine.getProperty('voices')
        
        print(f"🔊 পাওয়া ভয়েস: {len(voices)}")
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} ({voice.languages})")
        
        # Try to find Bangla or female voice
        bangla_voice = None
        female_voice = None
        
        for voice in voices:
            if 'bn' in str(voice.languages).lower() or 'bengali' in voice.name.lower():
                bangla_voice = voice
                break
            elif 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                female_voice = voice
        
        # Set voice
        if bangla_voice:
            self.engine.setProperty('voice', bangla_voice.id)
            self.current_voice = bangla_voice
            print("✅ বাংলা ভয়েস সেট করা হয়েছে")
        elif female_voice:
            self.engine.setProperty('voice', female_voice.id)
            self.current_voice = female_voice
            print("✅ মহিলা ভয়েস সেট করা হয়েছে")
        else:
            print("⚠️ বাংলা বা মহিলা ভয়েস পাওয়া যায়নি, ডিফল্ট ব্যবহার করা হচ্ছে")
        
        # Configure voice properties
        self.engine.setProperty('rate', 150)  # Speech rate
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    def speak(self, text: str, use_cache: bool = True, 
             emotion: str = None) -> Optional[str]:
        """Speak text with emotion"""
        try:
            print(f"🗣️ বলা হচ্ছে: '{text[:50]}...'")
            
            # Check cache first
            cache_key = f"{text}_{emotion}"
            if use_cache and cache_key in self.voice_cache:
                audio_file = self.voice_cache[cache_key]
                self._play_audio(audio_file)
                return audio_file
            
            # Generate speech
            if emotion:
                text = self._add_emotional_prosody(text, emotion)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                audio_file = tmp_file.name
            
            # Use gTTS for better Bangla support
            try:
                tts = gTTS(text=text, lang='bn', slow=False)
                tts.save(audio_file)
                
                # Cache the audio
                if use_cache and len(self.voice_cache) < 100:  # Limit cache size
                    self.voice_cache[cache_key] = audio_file
                
                # Play audio
                self._play_audio(audio_file)
                
                return audio_file
                
            except Exception as gtts_error:
                print(f"❌ gTTS ত্রুটি, পিইটিটিএসএক্স৩ ব্যবহার করা হচ্ছে: {gtts_error}")
                
                # Fallback to pyttsx3
                self.engine.save_to_file(text, audio_file)
                self.engine.runAndWait()
                
                # Cache the audio
                if use_cache:
                    self.voice_cache[cache_key] = audio_file
                
                # Play audio
                self._play_audio(audio_file)
                
                return audio_file
            
        except Exception as e:
            print(f"❌ কথা বলা যায়নি: {e}")
            return None
    
    def _play_audio(self, audio_file: str):
        """Play audio file"""
        try:
            mixer.music.load(audio_file)
            mixer.music.play()
            
            # Wait for playback to finish
            while mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ অডিও প্লে করা যায়নি: {e}")
    
    def _add_emotional_prosody(self, text: str, emotion: str) -> str:
        """Add emotional prosody to text"""
        emotional_prefixes = {
            "happy": "আনন্দের সাথে বলছি, ",
            "sad": "দুঃখের সাথে বলছি, ",
            "excited": "উত্তেজিত হয়ে বলছি, ",
            "angry": "রাগান্বিত হয়ে বলছি, ",
            "calm": "শান্তভাবে বলছি, ",
            "surprised": "বিস্ময়ের সাথে বলছি, "
        }
        
        if emotion in emotional_prefixes:
            return emotional_prefixes[emotion] + text
        
        return text
    
    def save_speech(self, text: str, filename: str = None, 
                   emotion: str = None) -> Optional[str]:
        """Save speech to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice/speech/speech_{timestamp}.mp3"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Generate speech
            if emotion:
                text = self._add_emotional_prosody(text, emotion)
            
            tts = gTTS(text=text, lang='bn', slow=False)
            tts.save(filename)
            
            print(f"💾 স্পিচ সংরক্ষণ করা হয়েছে: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ স্পিচ সংরক্ষণ করা যায়নি: {e}")
            return None
    
    def set_voice_properties(self, rate: int = None, volume: float = None, 
                            voice_id: str = None):
        """Set voice properties"""
        try:
            if rate is not None:
                self.engine.setProperty('rate', rate)
                print(f"✅ স্পিচ রেট সেট করা হয়েছে: {rate}")
            
            if volume is not None:
                self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
                print(f"✅ ভলিউম সেট করা হয়েছে: {volume}")
            
            if voice_id is not None:
                self.engine.setProperty('voice', voice_id)
                print(f"✅ ভয়েস সেট করা হয়েছে")
                
        except Exception as e:
            print(f"❌ ভয়েস প্রপার্টি সেট করা যায়নি: {e}")
    
    def list_available_voices(self) -> List[Dict[str, Any]]:
        """List all available voices"""
        voices = []
        all_voices = self.engine.getProperty('voices')
        
        for voice in all_voices:
            voices.append({
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages,
                "gender": voice.gender if hasattr(voice, 'gender') else "unknown"
            })
        
        return voices

class VoiceCommandProcessor:
    """Process voice commands with natural language understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.command_patterns = self._load_command_patterns()
        self.context = {}
        
    def process_command(self, voice_command: VoiceCommand) -> Dict[str, Any]:
        """Process voice command and extract intent"""
        try:
            text = voice_command.text.lower()
            language = voice_command.language
            
            # Clean text
            text = self._clean_text(text)
            
            # Extract intent
            intent = self._extract_intent(text, language)
            
            # Extract entities
            entities = self._extract_entities(text, intent, language)
            
            # Generate response
            response = self._generate_response(intent, entities, voice_command.emotion)
            
            result = {
                "original_text": voice_command.text,
                "cleaned_text": text,
                "intent": intent,
                "entities": entities,
                "response": response,
                "confidence": voice_command.confidence,
                "emotion": voice_command.emotion,
                "language": language,
                "timestamp": voice_command.timestamp.isoformat()
            }
            
            print(f"🎯 কমান্ড প্রসেস করা হয়েছে: {intent}")
            return result
            
        except Exception as e:
            print(f"❌ কমান্ড প্রসেস করা যায়নি: {e}")
            return {
                "error": str(e),
                "original_text": voice_command.text if voice_command else ""
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove punctuation (keep for Bangla)
        # punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        # text = text.translate(str.maketrans('', '', punctuation))
        
        return text
    
    def _extract_intent(self, text: str, language: str) -> str:
        """Extract intent from text"""
        # Bangla command patterns
        bangla_patterns = {
            "greeting": ["হ্যালো", "হাই", "কেমন আছ", "কি অবস্থা", "নমস্কার"],
            "time": ["সময় কত", "কয়টা বাজে", "সময় বল", "ঘড়িতে কটা"],
            "date": ["আজকের তারিখ", "তারিখ বল", "কত তারিখ"],
            "weather": ["আবহাওয়া", "আকাশ", "বৃষ্টি", "তাপমাত্রা", "আদৌ"],
            "open_app": ["খোল", "চালু কর", "ওপেন", "স্টার্ট"],
            "close_app": ["বন্ধ কর", "ক্লোজ", "স্টপ", "বাতিল"],
            "search": ["খুঁজ", "সার্চ", "জানতে চাই", "তথ্য"],
            "music": ["গান", "সঙ্গীত", "মিউজিক", "প্লে", "বাজাও"],
            "joke": ["রসিকতা", "মজা", "কৌতুক", "হাসি"],
            "news": ["খবর", "নিউজ", "তাজা খবর", "সংবাদ"],
            "calculation": ["গণনা", "ক্যালকুলেট", "যোগ", "বিয়োগ", "গুণ", "ভাগ"],
            "reminder": ["মনে করিয়ে", "রিমাইন্ডার", "মনে রাখ", "ইঙ্গিত"],
            "translation": ["অনুবাদ", "ট্রান্সলেট", "ভাষান্তর"],
            "system": ["সিস্টেম", "কম্পিউটার", "শাটডাউন", "রিস্টার্ট"]
        }
        
        # English command patterns
        english_patterns = {
            "greeting": ["hello", "hi", "hey", "how are you"],
            "time": ["time", "what time", "clock"],
            "date": ["date", "today's date", "what date"],
            "weather": ["weather", "temperature", "forecast"],
            "open_app": ["open", "start", "launch"],
            "close_app": ["close", "stop", "exit", "quit"],
            "search": ["search", "find", "look for"],
            "music": ["music", "song", "play music"],
            "joke": ["joke", "funny", "make me laugh"],
            "news": ["news", "headlines", "latest news"],
            "calculation": ["calculate", "math", "add", "subtract"],
            "reminder": ["reminder", "remind me", "remember"],
            "translation": ["translate", "translation"],
            "system": ["system", "computer", "shutdown", "restart"]
        }
        
        # Select patterns based on language
        patterns = bangla_patterns if language == "bn" else english_patterns
        
        # Match patterns
        for intent, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        
        # Default intent
        return "unknown"
    
    def _extract_entities(self, text: str, intent: str, language: str) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        if intent == "open_app" or intent == "close_app":
            # Extract app name
            apps = ["ক্রোম", "ফায়ারফক্স", "নোটপ্যাড", "ক্যালকুলেটর", 
                   "মিউজিক", "ভিডিও", "ওয়ার্ড", "এক্সেল"]
            
            for app in apps:
                if app in text:
                    entities["app_name"] = app
                    break
        
        elif intent == "search":
            # Extract search query
            search_keywords = ["খুঁজ", "সার্চ", "জানতে চাই"]
            for keyword in search_keywords:
                if keyword in text:
                    query_start = text.find(keyword) + len(keyword)
                    entities["query"] = text[query_start:].strip()
                    break
        
        elif intent == "weather":
            # Extract location
            locations = ["ঢাকা", "চট্টগ্রাম", "রাজশাহী", "খুলনা", "বরিশাল", "সিলেট"]
            for location in locations:
                if location in text:
                    entities["location"] = location
                    break
        
        elif intent == "time" or intent == "date":
            # Extract time/date modifiers
            modifiers = ["এখন", "আজ", "কাল", "পরশু", "গতকাল"]
            for modifier in modifiers:
                if modifier in text:
                    entities["modifier"] = modifier
                    break
        
        return entities
    
    def _generate_response(self, intent: str, entities: Dict[str, Any], 
                          emotion: str = None) -> str:
        """Generate appropriate response"""
        responses = {
            "greeting": {
                "bn": ["হ্যালো! কেমন আছেন?", "নমস্কার! কি সাহায্য করতে পারি?", "হাই! আপনার জন্য কি করতে পারি?"],
                "en": ["Hello! How can I help you?", "Hi there! What can I do for you?", "Greetings! How may I assist you?"]
            },
            "time": {
                "bn": [f"এখন সময় {datetime.now().strftime('%I:%M %p')}", f"ঘড়িতে {datetime.now().strftime('%I:%M %p')} বাজে"],
                "en": [f"The time is {datetime.now().strftime('%I:%M %p')}", f"It's {datetime.now().strftime('%I:%M %p')} now"]
            },
            "date": {
                "bn": [f"আজকের তারিখ {datetime.now().strftime('%d %B, %Y')}", f"আজ {datetime.now().strftime('%d %B, %Y')}"],
                "en": [f"Today's date is {datetime.now().strftime('%B %d, %Y')}", f"It's {datetime.now().strftime('%B %d, %Y')} today"]
            },
            "weather": {
                "bn": ["আবহাওয়া তথ্য সংগ্রহ করা হচ্ছে...", "এক মুহূর্ত, আবহাওয়া চেক করছি..."],
                "en": ["Getting weather information...", "Checking the weather for you..."]
            },
            "open_app": {
                "bn": ["অ্যাপ্লিকেশন খোলা হচ্ছে...", "ঠিক আছে, খুলছি..."],
                "en": ["Opening the application...", "Alright, starting it up..."]
            },
            "close_app": {
                "bn": ["অ্যাপ্লিকেশন বন্ধ করা হচ্ছে...", "ঠিক আছে, বন্ধ করছি..."],
                "en": ["Closing the application...", "Okay, shutting it down..."]
            },
            "search": {
                "bn": ["অনুসন্ধান করা হচ্ছে...", "তথ্য খুঁজছি..."],
                "en": ["Searching for information...", "Looking that up for you..."]
            },
            "music": {
                "bn": ["গান বাজানো হচ্ছে...", "সঙ্গীত শোনা যাক..."],
                "en": ["Playing music...", "Let's listen to some music..."]
            },
            "joke": {
                "bn": ["একটা মজার কথা শুনুন...", "হাসির জন্য তৈরি?"],
                "en": ["Here's a funny one...", "Ready for a laugh?"]
            },
            "news": {
                "bn": ["তাজা খবর সংগ্রহ করা হচ্ছে...", "সাম্প্রতিক সংবাদ দেখছি..."],
                "en": ["Getting the latest news...", "Checking recent headlines..."]
            },
            "unknown": {
                "bn": ["দুঃখিত, বুঝতে পারিনি। আবার বলবেন?", "মাফ করবেন, বুঝতে সমস্যা হচ্ছে।"],
                "en": ["Sorry, I didn't understand that. Could you repeat?", "I apologize, I'm having trouble understanding."]
            }
        }
        
        # Select language (default to Bangla)
        lang = "bn"
        if intent in responses:
            response_list = responses[intent].get(lang, ["দুঃখিত, এখন উত্তর দিতে পারছি না।"])
            response = response_list[0]
            
            # Add emotion if specified
            if emotion:
                emotional_suffixes = {
                    "happy": " 😊",
                    "sad": " 😔",
                    "excited": " 🎉",
                    "angry": " 😠"
                }
                if emotion in emotional_suffixes:
                    response += emotional_suffixes[emotion]
            
            return response
        
        return "দুঃখিত, এখন উত্তর দিতে পারছি না।"
    
    def _load_command_patterns(self) -> Dict[str, Any]:
        """Load command patterns from file"""
        try:
            with open("voice/command_patterns.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"❌ কমান্ড প্যাটার্ন লোড করা যায়নি: {e}")
            return {}

class VoiceActivityDetector:
    """Voice Activity Detection"""
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.01):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.is_speaking = False
        self.silence_duration = 0
    
    def detect(self, audio_data: np.ndarray) -> bool:
        """Detect if voice is present"""
        # Calculate energy
        energy = np.sqrt(np.mean(audio_data**2))
        
        # Update speaking state
        if energy > self.threshold:
            self.is_speaking = True
            self.silence_duration = 0
            return True
        else:
            if self.is_speaking:
                self.silence_duration += len(audio_data) / self.sample_rate
                
                # If silence continues for more than 0.5 seconds, stop speaking
                if self.silence_duration > 0.5:
                    self.is_speaking = False
            
            return False
    
    def get_speech_segments(self, audio_data: np.ndarray) -> List[tuple]:
        """Get speech segments from audio"""
        window_size = int(self.sample_rate * 0.1)  # 100ms windows
        segments = []
        current_segment = None
        
        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i + window_size]
            has_speech = self.detect(window)
            
            if has_speech and current_segment is None:
                # Start of speech
                current_segment = [i / self.sample_rate]
            elif not has_speech and current_segment is not None:
                # End of speech
                current_segment.append(i / self.sample_rate)
                segments.append(tuple(current_segment))
                current_segment = None
        
        # Add last segment if still speaking
        if current_segment is not None:
            current_segment.append(len(audio_data) / self.sample_rate)
            segments.append(tuple(current_segment))
        
        return segments

class VoiceTrainer:
    """Train voice recognition for specific speakers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.speaker_profiles = {}
        
    def train_speaker(self, speaker_id: str, audio_samples: List[np.ndarray]):
        """Train voice model for specific speaker"""
        try:
            # Extract features
            features = []
            for sample in audio_samples:
                mfcc = self._extract_mfcc(sample)
                pitch = self._extract_pitch(sample)
                features.append({
                    "mfcc": mfcc,
                    "pitch": pitch
                })
            
            # Create speaker profile
            self.speaker_profiles[speaker_id] = {
                "features": features,
                "trained_at": datetime.now().isoformat(),
                "sample_count": len(audio_samples)
            }
            
            # Save profile
            self._save_speaker_profile(speaker_id)
            
            print(f"✅ স্পিকার প্রশিক্ষণ সম্পূর্ণ: {speaker_id}")
            return True
            
        except Exception as e:
            print(f"❌ স্পিকার প্রশিক্ষণ ব্যর্থ: {e}")
            return False
    
    def identify_speaker(self, audio_sample: np.ndarray) -> Optional[str]:
        """Identify speaker from audio sample"""
        try:
            if not self.speaker_profiles:
                return None
            
            # Extract features from sample
            sample_mfcc = self._extract_mfcc(audio_sample)
            sample_pitch = self._extract_pitch(audio_sample)
            
            best_match = None
            best_score = 0
            
            for speaker_id, profile in self.speaker_profiles.items():
                score = self._calculate_similarity(sample_mfcc, sample_pitch, profile["features"])
                
                if score > best_score:
                    best_score = score
                    best_match = speaker_id
            
            # Threshold for identification
            if best_score > 0.7:
                return best_match
            
            return None
            
        except Exception as e:
            print(f"❌ স্পিকার শনাক্তকরণ ব্যর্থ: {e}")
            return None
    
    def _extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data.astype(np.float32),
                sr=16000,
                n_mfcc=13
            )
            return np.mean(mfcc, axis=1)
        except:
            return np.zeros(13)
    
    def _extract_pitch(self, audio_data: np.ndarray) -> float:
        """Extract pitch"""
        try:
            pitches, magnitudes = librosa.piptrack(
                y=audio_data.astype(np.float32),
                sr=16000
            )
            pitch_idx = magnitudes.argmax()
            pitch = pitches.flatten()[pitch_idx]
            return float(pitch) if pitch > 0 else 100.0
        except:
            return 100.0
    
    def _calculate_similarity(self, mfcc1: np.ndarray, pitch1: float, 
                            features_list: List[Dict[str, Any]]) -> float:
        """Calculate similarity score"""
        scores = []
        
        for features in features_list:
            # MFCC similarity (cosine similarity)
            mfcc2 = features["mfcc"]
            if mfcc1.shape == mfcc2.shape:
                mfcc_sim = np.dot(mfcc1, mfcc2) / (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2))
            else:
                mfcc_sim = 0
            
            # Pitch similarity
            pitch2 = features["pitch"]
            pitch_sim = 1.0 - min(abs(pitch1 - pitch2) / 100.0, 1.0)
            
            # Combined score
            score = (mfcc_sim + pitch_sim) / 2
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _save_speaker_profile(self, speaker_id: str):
        """Save speaker profile to file"""
        try:
            os.makedirs("voice/profiles", exist_ok=True)
            filename = f"voice/profiles/{speaker_id}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.speaker_profiles[speaker_id], f, indent=2)
        except Exception as e:
            print(f"❌ স্পিকার প্রোফাইল সংরক্ষণ করা যায়নি: {e}")