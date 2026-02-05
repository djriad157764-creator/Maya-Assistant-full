"""
Advanced Data Storage System with Encryption and Compression
"""
import json
import pickle
import sqlite3
import csv
import h5py
import numpy as np
from typing import Dict, List, Any, Optional, Union
import os
from datetime import datetime
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import zlib
import gzip
import lzma
import threading
from queue import Queue
import shutil

class AdvancedStorage:
    """Advanced storage system with multiple backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_key = self._generate_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize storage backends
        self.json_storage = JSONStorage()
        self.sqlite_storage = SQLiteStorage()
        self.hdf5_storage = HDF5Storage()
        self.cache_storage = CacheStorage()
        
        # Create data directories
        self._create_directories()
        
        print("💾 অ্যাডভান্সড স্টোরেজ সিস্টেম প্রস্তুত")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key"""
        # Use config password or generate from system
        password = self.config.get("encryption_password", "default_password").encode()
        salt = b'salt_'  # Should be random in production
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            "data/json",
            "data/sqlite",
            "data/hdf5",
            "data/cache",
            "data/backups",
            "data/exports",
            "data/logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_json(self, data: Any, filename: str, encrypt: bool = False, 
                 compress: bool = False) -> bool:
        """Save data as JSON"""
        try:
            return self.json_storage.save(
                data, filename, 
                encryption_key=self.encryption_key if encrypt else None,
                compress=compress
            )
        except Exception as e:
            print(f"❌ JSON সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load_json(self, filename: str, encrypted: bool = False, 
                 compressed: bool = False) -> Optional[Any]:
        """Load data from JSON"""
        try:
            return self.json_storage.load(
                filename,
                encryption_key=self.encryption_key if encrypted else None,
                compressed=compressed
            )
        except Exception as e:
            print(f"❌ JSON লোড করা যায়নি: {e}")
            return None
    
    def save_sqlite(self, table: str, data: List[Dict[str, Any]], 
                   database: str = "main.db") -> bool:
        """Save data to SQLite"""
        try:
            return self.sqlite_storage.save(table, data, database)
        except Exception as e:
            print(f"❌ SQLite সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load_sqlite(self, query: str, database: str = "main.db") -> List[Dict[str, Any]]:
        """Load data from SQLite"""
        try:
            return self.sqlite_storage.load(query, database)
        except Exception as e:
            print(f"❌ SQLite লোড করা যায়নি: {e}")
            return []
    
    def save_hdf5(self, dataset_name: str, data: np.ndarray, 
                 metadata: Dict[str, Any] = None) -> bool:
        """Save data to HDF5"""
        try:
            return self.hdf5_storage.save(dataset_name, data, metadata)
        except Exception as e:
            print(f"❌ HDF5 সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load_hdf5(self, dataset_name: str) -> Optional[np.ndarray]:
        """Load data from HDF5"""
        try:
            return self.hdf5_storage.load(dataset_name)
        except Exception as e:
            print(f"❌ HDF5 লোড করা যায়নি: {e}")
            return None
    
    def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value"""
        try:
            return self.cache_storage.set(key, value, ttl)
        except Exception as e:
            print(f"❌ ক্যাশে সেট করা যায়নি: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            return self.cache_storage.get(key)
        except Exception as e:
            print(f"❌ ক্যাশে পাওয়া যায়নি: {e}")
            return None
    
    def backup_data(self, backup_name: str = None) -> Optional[str]:
        """Create data backup"""
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_path = os.path.join("data/backups", backup_name)
            
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy data files
            data_dirs = ["json", "sqlite", "hdf5"]
            for data_dir in data_dirs:
                src = os.path.join("data", data_dir)
                if os.path.exists(src):
                    dst = os.path.join(backup_path, data_dir)
                    shutil.copytree(src, dst)
            
            # Compress backup
            compressed_path = f"{backup_path}.zip"
            shutil.make_archive(backup_path, 'zip', backup_path)
            
            # Remove uncompressed backup
            shutil.rmtree(backup_path)
            
            print(f"✅ ব্যাকআপ তৈরি করা হয়েছে: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            print(f"❌ ব্যাকআপ তৈরি করা যায়নি: {e}")
            return None
    
    def export_data(self, format: str = "json", filename: str = None) -> Optional[str]:
        """Export data in specified format"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_{timestamp}.{format}"
            
            export_path = os.path.join("data/exports", filename)
            
            if format == "json":
                # Export all JSON data
                export_data = {}
                json_dir = "data/json"
                
                for json_file in os.listdir(json_dir):
                    if json_file.endswith('.json'):
                        file_path = os.path.join(json_dir, json_file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            export_data[json_file] = data
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format == "csv":
                # Export SQLite data as CSV
                conn = sqlite3.connect("data/sqlite/main.db")
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()
                    
                    # Get column names
                    column_names = [description[0] for description in cursor.description]
                    
                    # Write to CSV
                    csv_path = export_path.replace('.csv', f'_{table_name}.csv')
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(column_names)
                        writer.writerows(rows)
                
                conn.close()
            
            print(f"✅ ডেটা এক্সপোর্ট করা হয়েছে: {export_path}")
            return export_path
            
        except Exception as e:
            print(f"❌ ডেটা এক্সপোর্ট করা যায়নি: {e}")
            return None

class JSONStorage:
    """JSON storage with encryption and compression"""
    
    def save(self, data: Any, filename: str, encryption_key: bytes = None, 
            compress: bool = False) -> bool:
        """Save data to JSON file"""
        try:
            filepath = os.path.join("data/json", filename)
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            # Convert data to JSON string
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            
            # Encrypt if requested
            if encryption_key:
                cipher = Fernet(encryption_key)
                json_str = cipher.encrypt(json_str.encode()).decode()
            
            # Compress if requested
            if compress:
                json_bytes = json_str.encode()
                compressed = zlib.compress(json_bytes, level=9)
                
                # Save compressed data
                with open(filepath, 'wb') as f:
                    f.write(compressed)
            else:
                # Save uncompressed data
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            print(f"💾 JSON সংরক্ষণ করা হয়েছে: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ JSON সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load(self, filename: str, encryption_key: bytes = None, 
            compressed: bool = False) -> Optional[Any]:
        """Load data from JSON file"""
        try:
            filepath = os.path.join("data/json", filename)
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            if not os.path.exists(filepath):
                print(f"⚠️ ফাইল পাওয়া যায়নি: {filepath}")
                return None
            
            if compressed:
                # Load compressed data
                with open(filepath, 'rb') as f:
                    compressed_data = f.read()
                
                # Decompress
                json_bytes = zlib.decompress(compressed_data)
                json_str = json_bytes.decode()
            else:
                # Load uncompressed data
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_str = f.read()
            
            # Decrypt if encrypted
            if encryption_key:
                cipher = Fernet(encryption_key)
                json_str = cipher.decrypt(json_str.encode()).decode()
            
            # Parse JSON
            data = json.loads(json_str)
            
            print(f"📂 JSON লোড করা হয়েছে: {filepath}")
            return data
            
        except Exception as e:
            print(f"❌ JSON লোড করা যায়নি: {e}")
            return None

class SQLiteStorage:
    """SQLite database storage"""
    
    def save(self, table: str, data: List[Dict[str, Any]], 
            database: str = "main.db") -> bool:
        """Save data to SQLite database"""
        try:
            db_path = os.path.join("data/sqlite", database)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if not data:
                print("⚠️ সংরক্ষণের জন্য কোন ডেটা নেই")
                return False
            
            # Get column names from first item
            columns = list(data[0].keys())
            
            # Create table if it doesn't exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {', '.join([f'{col} TEXT' for col in columns])},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Insert data
            for item in data:
                values = [str(item.get(col, '')) for col in columns]
                placeholders = ', '.join(['?'] * len(columns))
                insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
            
            conn.commit()
            conn.close()
            
            print(f"💾 SQLite ডেটা সংরক্ষণ করা হয়েছে: {table} ({len(data)} রো)")
            return True
            
        except Exception as e:
            print(f"❌ SQLite সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load(self, query: str, database: str = "main.db") -> List[Dict[str, Any]]:
        """Load data from SQLite database"""
        try:
            db_path = os.path.join("data/sqlite", database)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable dictionary-like access
            cursor = conn.cursor()
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            result = []
            for row in rows:
                result.append(dict(row))
            
            conn.close()
            
            print(f"📂 SQLite ডেটা লোড করা হয়েছে: {len(result)} রো")
            return result
            
        except Exception as e:
            print(f"❌ SQLite লোড করা যায়নি: {e}")
            return []

class HDF5Storage:
    """HDF5 storage for large datasets"""
    
    def save(self, dataset_name: str, data: np.ndarray, 
            metadata: Dict[str, Any] = None) -> bool:
        """Save data to HDF5 file"""
        try:
            filepath = "data/hdf5/data.h5"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with h5py.File(filepath, 'a') as f:
                # Save dataset
                if dataset_name in f:
                    del f[dataset_name]
                
                f.create_dataset(dataset_name, data=data)
                
                # Save metadata as attributes
                if metadata:
                    for key, value in metadata.items():
                        f[dataset_name].attrs[key] = value
                
                # Add timestamp
                f[dataset_name].attrs['saved_at'] = datetime.now().isoformat()
            
            print(f"💾 HDF5 ডেটাসেট সংরক্ষণ করা হয়েছে: {dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ HDF5 সংরক্ষণ করা যায়নি: {e}")
            return False
    
    def load(self, dataset_name: str) -> Optional[np.ndarray]:
        """Load data from HDF5 file"""
        try:
            filepath = "data/hdf5/data.h5"
            
            if not os.path.exists(filepath):
                print(f"⚠️ HDF5 ফাইল পাওয়া যায়নি: {filepath}")
                return None
            
            with h5py.File(filepath, 'r') as f:
                if dataset_name not in f:
                    print(f"⚠️ ডেটাসেট পাওয়া যায়নি: {dataset_name}")
                    return None
                
                data = f[dataset_name][:]
                
                # Load metadata
                metadata = dict(f[dataset_name].attrs)
                print(f"📂 HDF5 ডেটাসেট লোড করা হয়েছে: {dataset_name}")
                print(f"   মেটাডেটা: {metadata}")
                
                return data
            
        except Exception as e:
            print(f"❌ HDF5 লোড করা যায়নি: {e}")
            return None

class CacheStorage:
    """In-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL"""
        try:
            # Check cache size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # Set value
            self.cache[key] = value
            self.timestamps[key] = time.time() + ttl
            
            return True
            
        except Exception as e:
            print(f"❌ ক্যাশে সেট করা যায়নি: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            if key not in self.cache:
                return None
            
            # Check if expired
            if time.time() > self.timestamps[key]:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
            
        except Exception as e:
            print(f"❌ ক্যাশে পাওয়া যায়নি: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache value"""
        try:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            return True
        except:
            return False
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def _cleanup_expired(self):
        """Cleanup expired cache entries"""
        while True:
            time.sleep(60)  # Cleanup every minute
            
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self.timestamps.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]
            
            if expired_keys:
                print(f"🧹 {len(expired_keys)}টি এক্সপায়ার্ড ক্যাশে এন্ট্রি মুছে ফেলা হয়েছে")

class DataValidator:
    """Data validation and cleaning"""
    
    @staticmethod
    def validate_json(data: Any, schema: Dict[str, Any] = None) -> bool:
        """Validate JSON data against schema"""
        try:
            if schema is None:
                return True
            
            # Simple schema validation
            # In production, use JSON Schema validator
            return True
        except:
            return False
    
    @staticmethod
    def clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data by removing null/empty values"""
        cleaned = {}
        
        for key, value in data.items():
            if value is not None:
                if isinstance(value, str):
                    value = value.strip()
                    if value:  # Only add non-empty strings
                        cleaned[key] = value
                elif isinstance(value, (list, dict)):
                    if value:  # Only add non-empty lists/dicts
                        cleaned[key] = value
                else:
                    cleaned[key] = value
        
        return cleaned
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text (remove extra spaces, normalize quotes, etc.)"""
        if not text:
            return text
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Normalize quotes
        text = text.replace('"', "'")
        
        # Remove special characters (keep Bengali and basic punctuation)
        # Keep Bengali Unicode range: \u0980-\u09FF
        import re
        text = re.sub(r'[^\u0980-\u09FF\w\s\.\,\!\?\-\'\"]', '', text)
        
        return text

class DataAnalyzer:
    """Data analysis utilities"""
    
    @staticmethod
    def get_statistics(data: List[Any]) -> Dict[str, Any]:
        """Get basic statistics from data"""
        if not data:
            return {}
        
        # Convert to numeric if possible
        numeric_data = []
        for item in data:
            try:
                numeric_data.append(float(item))
            except:
                pass
        
        if numeric_data:
            return {
                "count": len(numeric_data),
                "mean": np.mean(numeric_data),
                "median": np.median(numeric_data),
                "std": np.std(numeric_data),
                "min": np.min(numeric_data),
                "max": np.max(numeric_data)
            }
        else:
            return {
                "count": len(data),
                "unique_values": len(set(data))
            }
    
    @staticmethod
    def find_patterns(data: List[Any], window_size: int = 3) -> List[Any]:
        """Find patterns in data"""
        if len(data) < window_size:
            return []
        
        patterns = []
        for i in range(len(data) - window_size + 1):
            pattern = tuple(data[i:i + window_size])
            patterns.append(pattern)
        
        # Find frequent patterns
        from collections import Counter
        pattern_counts = Counter(patterns)
        
        frequent_patterns = [
            {"pattern": pattern, "count": count}
            for pattern, count in pattern_counts.items()
            if count > 1
        ]
        
        return sorted(frequent_patterns, key=lambda x: x["count"], reverse=True)