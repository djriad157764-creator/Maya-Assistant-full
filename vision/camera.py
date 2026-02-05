"""
Advanced Computer Vision with Camera and Screen Capture
"""
import cv2
import numpy as np
import pyautogui
import mss
import mss.tools
from PIL import Image, ImageGrab
import torch
import torchvision
from torchvision import models, transforms
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import os
import json
import threading
from queue import Queue
import face_recognition
import mediapipe as mp
from deepface import DeepFace
import pytesseract
from screeninfo import get_monitors

class AdvancedVision:
    """Advanced computer vision system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera = None
        self.screen_capture = ScreenCapture()
        self.object_detector = ObjectDetector()
        self.face_recognizer = FaceRecognizer()
        self.ocr_engine = OCREngine()
        self.gesture_recognizer = GestureRecognizer()
        
        # Initialize camera
        self._initialize_camera()
        
        # Vision processing queue
        self.processing_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_vision_queue, daemon=True)
        self.processing_thread.start()
        
        print("👁️ ভিশন সিস্টেম প্রস্তুত")
    
    def _initialize_camera(self):
        """Initialize camera"""
        try:
            # Try to open camera
            self.camera = cv2.VideoCapture(0)
            
            if self.camera.isOpened():
                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                print("📷 ক্যামেরা প্রস্তুত")
            else:
                print("⚠️ ক্যামেরা খোলা যায়নি")
                self.camera = None
                
        except Exception as e:
            print(f"❌ ক্যামেরা ইনিশিয়ালাইজেশন ত্রুটি: {e}")
            self.camera = None
    
    def capture_camera(self, save_path: str = None) -> Optional[np.ndarray]:
        """Capture image from camera"""
        if self.camera is None:
            print("❌ ক্যামেরা উপলব্ধ নেই")
            return None
        
        try:
            ret, frame = self.camera.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save if path provided
                if save_path:
                    cv2.imwrite(save_path, frame)
                    print(f"💾 ক্যামেরা ছবি সংরক্ষণ করা হয়েছে: {save_path}")
                
                return frame_rgb
            else:
                print("❌ ক্যামেরা থেকে ছবি ধারণ করা যায়নি")
                return None
                
        except Exception as e:
            print(f"❌ ক্যামেরা ক্যাপচার ত্রুটি: {e}")
            return None
    
    def capture_screenshot(self, region: Tuple[int, int, int, int] = None, 
                          save_path: str = None) -> Optional[Image.Image]:
        """Capture screenshot"""
        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Save if path provided
            if save_path:
                screenshot.save(save_path)
                print(f"💾 স্ক্রীনশট সংরক্ষণ করা হয়েছে: {save_path}")
            
            return screenshot
            
        except Exception as e:
            print(f"❌ স্ক্রীনশট ত্রুটি: {e}")
            return None
    
    def record_video(self, duration: int, save_path: str = "recordings") -> Optional[str]:
        """Record video from camera"""
        if self.camera is None:
            print("❌ ক্যামেরা উপলব্ধ নেই")
            return None
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"video_{timestamp}.avi")
            
            # Video writer
            frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            
            print(f"🎥 ভিডিও রেকর্ডিং শুরু ({duration} সেকেন্ড)...")
            start_time = time.time()
            
            while (time.time() - start_time) < duration:
                ret, frame = self.camera.read()
                
                if ret:
                    out.write(frame)
                else:
                    break
            
            out.release()
            print(f"✅ ভিডিও রেকর্ডিং সম্পন্ন: {filename}")
            
            return filename
            
        except Exception as e:
            print(f"❌ ভিডিও রেকর্ডিং ত্রুটি: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        return self.face_recognizer.detect_faces(image)
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize faces in image"""
        return self.face_recognizer.recognize_faces(image)
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        return self.object_detector.detect(image)
    
    def extract_text(self, image: np.ndarray, language: str = "ben") -> str:
        """Extract text from image"""
        return self.ocr_engine.extract_text(image, language)
    
    def recognize_gestures(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize hand gestures"""
        return self.gesture_recognizer.recognize(image)
    
    def analyze_emotion(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze emotion from face"""
        try:
            # Convert to BGR for DeepFace
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Analyze emotion
            analysis = DeepFace.analyze(
                img_path=image_bgr,
                actions=['emotion'],
                enforce_detection=False
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            return {
                "dominant_emotion": analysis.get('dominant_emotion', 'neutral'),
                "emotions": analysis.get('emotion', {}),
                "region": analysis.get('region', {})
            }
            
        except Exception as e:
            print(f"❌ আবেগ বিশ্লেষণ ত্রুটি: {e}")
            return {"error": str(e)}
    
    def track_object(self, image: np.ndarray, object_name: str) -> Optional[Dict[str, Any]]:
        """Track specific object in image"""
        try:
            # Get object detections
            detections = self.detect_objects(image)
            
            for detection in detections:
                if detection["label"].lower() == object_name.lower():
                    return detection
            
            return None
            
        except Exception as e:
            print(f"❌ অবজেক্ট ট্র্যাকিং ত্রুটি: {e}")
            return None
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen information"""
        try:
            monitors = get_monitors()
            screen_info = {
                "monitor_count": len(monitors),
                "monitors": []
            }
            
            for i, monitor in enumerate(monitors):
                screen_info["monitors"].append({
                    "number": i + 1,
                    "width": monitor.width,
                    "height": monitor.height,
                    "x": monitor.x,
                    "y": monitor.y,
                    "name": getattr(monitor, 'name', f"Monitor {i+1}")
                })
            
            return screen_info
            
        except Exception as e:
            print(f"❌ স্ক্রীন তথ্য পাওয়া যায়নি: {e}")
            return {"error": str(e)}
    
    def _process_vision_queue(self):
        """Process vision tasks in queue"""
        while True:
            task = self.processing_queue.get()
            if task is None:
                break
            
            try:
                task_type = task.get("type")
                
                if task_type == "capture_camera":
                    self.capture_camera(task.get("save_path"))
                elif task_type == "capture_screenshot":
                    self.capture_screenshot(task.get("region"), task.get("save_path"))
                elif task_type == "detect_faces":
                    task["callback"](self.detect_faces(task["image"]))
                elif task_type == "extract_text":
                    task["callback"](self.extract_text(task["image"], task.get("language", "ben")))
                
            except Exception as e:
                print(f"❌ ভিশন টাস্ক ত্রুটি: {e}")
            
            self.processing_queue.task_done()
    
    def queue_vision_task(self, task: Dict[str, Any]):
        """Queue vision task for processing"""
        self.processing_queue.put(task)
        print(f"📥 ভিশন টাস্ক কিউ করা হয়েছে: {task.get('type')}")

class ScreenCapture:
    """Advanced screen capture"""
    
    def __init__(self):
        self.monitors = mss.mss()
    
    def capture_full_screen(self) -> Optional[np.ndarray]:
        """Capture full screen"""
        try:
            # Get monitor information
            monitor = self.monitors.monitors[1]  # Primary monitor
            
            # Capture screen
            screenshot = self.monitors.grab(monitor)
            
            # Convert to numpy array
            img_array = np.array(screenshot)
            
            # Convert BGRA to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            
            return img_rgb
            
        except Exception as e:
            print(f"❌ ফুল স্ক্রীন ক্যাপচার ত্রুটি: {e}")
            return None
    
    def capture_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture specific region"""
        try:
            # Capture region
            screenshot = self.monitors.grab(region)
            
            # Convert to numpy array
            img_array = np.array(screenshot)
            
            # Convert BGRA to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            
            return img_rgb
            
        except Exception as e:
            print(f"❌ রিজিয়ন ক্যাপচার ত্রুটি: {e}")
            return None
    
    def capture_multiple_monitors(self) -> List[np.ndarray]:
        """Capture all monitors"""
        screenshots = []
        
        try:
            for i, monitor in enumerate(self.monitors.monitors[1:], 1):
                screenshot = self.monitors.grab(monitor)
                img_array = np.array(screenshot)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
                screenshots.append(img_rgb)
            
            return screenshots
            
        except Exception as e:
            print(f"❌ মাল্টিপল মনিটর ক্যাপচার ত্রুটি: {e}")
            return []

class ObjectDetector:
    """Object detection using YOLO"""
    
    def __init__(self, model_name: str = "yolov5s"):
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            print(f"✅ অবজেক্ট ডিটেকশন মডেল লোড করা হয়েছে: {model_name}")
        except Exception as e:
            print(f"❌ অবজেক্ট ডিটেকশন মডেল লোড করা যায়নি: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        if self.model is None:
            return []
        
        try:
            # Perform detection
            results = self.model(image)
            
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                
                detection = {
                    "label": label,
                    "confidence": float(conf),
                    "bbox": {
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3])
                    },
                    "center": {
                        "x": int((xyxy[0] + xyxy[2]) / 2),
                        "y": int((xyxy[1] + xyxy[3]) / 2)
                    }
                }
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"❌ অবজেক্ট ডিটেকশন ত্রুটি: {e}")
            return []

class FaceRecognizer:
    """Face recognition and detection"""
    
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known faces from directory"""
        try:
            faces_dir = "vision/known_faces"
            if os.path.exists(faces_dir):
                for filename in os.listdir(faces_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        # Load image
                        image_path = os.path.join(faces_dir, filename)
                        image = face_recognition.load_image_file(image_path)
                        
                        # Get face encoding
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_faces.append(encodings[0])
                            self.known_names.append(os.path.splitext(filename)[0])
                
                print(f"✅ {len(self.known_faces)}টি পরিচিত মুখ লোড করা হয়েছে")
            else:
                print("⚠️ পরিচিত মুখের ডিরেক্টরি পাওয়া যায়নি")
                
        except Exception as e:
            print(f"❌ মুখ লোড করা যায়নি: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            # Convert RGB to BGR for face_recognition
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces
            face_locations = face_recognition.face_locations(image_bgr)
            face_encodings = face_recognition.face_encodings(image_bgr, face_locations)
            
            faces = []
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face = {
                    "id": i,
                    "location": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    },
                    "size": (right - left, bottom - top),
                    "center": {
                        "x": (left + right) // 2,
                        "y": (top + bottom) // 2
                    }
                }
                
                # Get encoding if available
                if i < len(face_encodings):
                    face["encoding"] = face_encodings[i].tolist()
                
                faces.append(face)
            
            return faces
            
        except Exception as e:
            print(f"❌ মুখ শনাক্তকরণ ত্রুটি: {e}")
            return []
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize faces in image"""
        try:
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces or not self.known_faces:
                return faces
            
            # Recognize each face
            for face in faces:
                if "encoding" in face:
                    face_encoding = np.array(face["encoding"])
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_faces, 
                        face_encoding,
                        tolerance=0.6
                    )
                    
                    face_distances = face_recognition.face_distance(
                        self.known_faces, 
                        face_encoding
                    )
                    
                    # Find best match
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        face["name"] = self.known_names[best_match_index]
                        face["confidence"] = float(1 - face_distances[best_match_index])
                    else:
                        face["name"] = "অজানা"
                        face["confidence"] = 0.0
            
            return faces
            
        except Exception as e:
            print(f"❌ মুখ চেনার ত্রুটি: {e}")
            return []

class OCREngine:
    """Optical Character Recognition"""
    
    def __init__(self):
        # Configure Tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def extract_text(self, image: np.ndarray, language: str = "ben") -> str:
        """Extract text from image"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image, lang=language)
            
            # Clean text
            text = text.strip()
            
            return text
            
        except Exception as e:
            print(f"❌ টেক্সট এক্সট্র্যাকশন ত্রুটি: {e}")
            return ""

class GestureRecognizer:
    """Hand gesture recognition"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize hand gestures"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(image_rgb)
            
            gestures = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append({
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z
                        })
                    
                    # Detect gesture
                    gesture = self._detect_gesture(landmarks)
                    
                    gestures.append({
                        "landmarks": landmarks,
                        "gesture": gesture
                    })
            
            return gestures
            
        except Exception as e:
            print(f"❌ জেসচার রিকগনিশন ত্রুটি: {e}")
            return []
    
    def _detect_gesture(self, landmarks: List[Dict[str, float]]) -> str:
        """Detect specific gesture from landmarks"""
        # Simplified gesture detection
        # In production, use machine learning model
        
        # Get finger tip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Check for thumbs up
        if thumb_tip["y"] < landmarks[3]["y"]:  # Thumb is up
            # Check if other fingers are closed
            if (index_tip["y"] > landmarks[6]["y"] and  # Index finger closed
                middle_tip["y"] > landmarks[10]["y"] and  # Middle finger closed
                ring_tip["y"] > landmarks[14]["y"] and  # Ring finger closed
                pinky_tip["y"] > landmarks[18]["y"]):  # Pinky finger closed
                return "thumbs_up"
        
        # Check for peace sign (index and middle fingers up)
        if (index_tip["y"] < landmarks[6]["y"] and  # Index finger up
            middle_tip["y"] < landmarks[10]["y"] and  # Middle finger up
            ring_tip["y"] > landmarks[14]["y"] and  # Ring finger closed
            pinky_tip["y"] > landmarks[18]["y"]):  # Pinky finger closed
            return "peace"
        
        # Check for ok sign (thumb and index finger touching)
        thumb_index_distance = np.sqrt(
            (thumb_tip["x"] - index_tip["x"])**2 +
            (thumb_tip["y"] - index_tip["y"])**2
        )
        
        if thumb_index_distance < 0.05:  # Very close
            return "ok"
        
        return "unknown"

class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def resize(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize image"""
        if width is None and height is None:
            return image
        
        h, w = image.shape[:2]
        
        if width is None:
            aspect_ratio = height / h
            width = int(w * aspect_ratio)
        elif height is None:
            aspect_ratio = width / w
            height = int(h * aspect_ratio)
        
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    @staticmethod
    def apply_filter(image: np.ndarray, filter_type: str = "blur") -> np.ndarray:
        """Apply filter to image"""
        if filter_type == "blur":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_type == "edge":
            return cv2.Canny(image, 100, 200)
        else:
            return image
    
    @staticmethod
    def extract_color_histogram(image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract color histogram"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        
        return {
            "hue": hist_h.flatten(),
            "saturation": hist_s.flatten(),
            "value": hist_v.flatten()
        }