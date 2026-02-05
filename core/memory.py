"""
Advanced Memory System with LSTM-based Memory Networks
"""
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json
import pickle
from collections import OrderedDict
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import heapq

@dataclass
class MemoryNode:
    """Memory node with relationships"""
    id: str
    content: str
    memory_type: str
    embedding: np.ndarray
    timestamp: datetime
    importance: float
    access_count: int
    relationships: List[str]
    metadata: Dict[str, Any]

class MemoryType(Enum):
    """Types of memories"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"

class MemoryEncoder(nn.Module):
    """Neural network for memory encoding"""
    def __init__(self, input_dim=768, hidden_dim=1024, memory_dim=512):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.memory_projection = nn.Linear(hidden_dim * 2, memory_dim)
        self.layer_norm = nn.LayerNorm(memory_dim)
        
    def forward(self, x):
        encoded, _ = self.encoder(x)
        attn_output, _ = self.attention(encoded, encoded, encoded)
        memory_embedding = self.memory_projection(attn_output[:, -1, :])
        return self.layer_norm(memory_embedding)

class AdvancedMemorySystem:
    """Advanced memory system with neural networks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_nodes = OrderedDict()
        self.memory_encoder = MemoryEncoder()
        self.retrieval_network = MemoryRetrievalNetwork()
        
        # Initialize databases
        self._init_databases()
        
        # Load existing memories
        self._load_memories()
        
        # Memory cache for fast access
        self.cache = LRUCache(capacity=1000)
        
        # Memory consolidation scheduler
        self.consolidation_scheduler = MemoryConsolidationScheduler(self)
    
    def store(self, content: str, memory_type: str, 
              importance: float = 0.5, metadata: Dict[str, Any] = None) -> str:
        """Store a new memory"""
        memory_id = self._generate_memory_id(content)
        
        # Encode content
        embedding = self._encode_content(content)
        
        # Create memory node
        node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            timestamp=datetime.now(),
            importance=importance,
            access_count=0,
            relationships=[],
            metadata=metadata or {}
        )
        
        # Store in memory
        self.memory_nodes[memory_id] = node
        self.cache.put(memory_id, node)
        
        # Update relationships
        self._update_relationships(node)
        
        # Save to database
        self._save_to_database(node)
        
        return memory_id
    
    def retrieve(self, query: str, k: int = 5, 
                 memory_type: Optional[str] = None) -> List[MemoryNode]:
        """Retrieve relevant memories"""
        # Check cache first
        cached = self.cache.get(query)
        if cached:
            return cached[:k]
        
        # Encode query
        query_embedding = self._encode_content(query)
        
        # Find similar memories
        similarities = []
        for node_id, node in self.memory_nodes.items():
            if memory_type and node.memory_type != memory_type:
                continue
            
            similarity = self._calculate_similarity(query_embedding, node.embedding)
            heapq.heappush(similarities, (similarity, node_id, node))
        
        # Get top k
        top_k = heapq.nlargest(k, similarities)
        
        # Update access counts
        for _, node_id, node in top_k:
            node.access_count += 1
            self.cache.put(node_id, node)
        
        return [node for _, _, node in top_k]
    
    def retrieve_by_association(self, concept: str, depth: int = 2) -> List[MemoryNode]:
        """Retrieve memories by association"""
        associated = []
        visited = set()
        
        def traverse(node_id: str, current_depth: int):
            if current_depth > depth or node_id in visited:
                return
            
            visited.add(node_id)
            node = self.memory_nodes.get(node_id)
            if node:
                associated.append(node)
                
                for related_id in node.relationships[:10]:  # Limit relationships
                    traverse(related_id, current_depth + 1)
        
        # Find starting nodes
        starting_nodes = []
        for node_id, node in self.memory_nodes.items():
            if concept.lower() in node.content.lower():
                starting_nodes.append(node_id)
        
        for start_id in starting_nodes[:3]:  # Limit starting points
            traverse(start_id, 0)
        
        return associated
    
    def consolidate(self):
        """Consolidate memories (move from short-term to long-term)"""
        consolidation_candidates = []
        
        for node_id, node in self.memory_nodes.items():
            if node.access_count > self.config.get("consolidation_threshold", 5):
                consolidation_candidates.append(node)
        
        # Neural consolidation process
        for node in consolidation_candidates:
            self._neural_consolidation(node)
    
    def forget(self, memory_id: str, soft: bool = True):
        """Forget a memory"""
        if soft:
            # Soft forget - reduce importance
            if memory_id in self.memory_nodes:
                node = self.memory_nodes[memory_id]
                node.importance *= 0.5
        else:
            # Hard forget - remove completely
            self.memory_nodes.pop(memory_id, None)
            self.cache.remove(memory_id)
            self._remove_from_database(memory_id)
    
    def _encode_content(self, content: str) -> np.ndarray:
        """Encode content to embedding"""
        # For now, use simple hash-based embedding
        # In production, use BERT or similar
        hash_obj = hashlib.sha256(content.encode())
        hash_bytes = hash_obj.digest()
        
        # Create 512-dim embedding from hash
        embedding = np.zeros(512)
        for i, byte in enumerate(hash_bytes[:64]):  # Use first 64 bytes
            embedding[i*8:(i+1)*8] = np.unpackbits(np.array([byte], dtype=np.uint8))
        
        return embedding
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _update_relationships(self, node: MemoryNode):
        """Update relationships between memories"""
        for other_id, other_node in self.memory_nodes.items():
            if other_id == node.id:
                continue
            
            similarity = self._calculate_similarity(node.embedding, other_node.embedding)
            if similarity > 0.7:  # Threshold for relationship
                node.relationships.append(other_id)
                other_node.relationships.append(node.id)
                
                # Limit relationships
                if len(node.relationships) > 20:
                    node.relationships.pop(0)
                if len(other_node.relationships) > 20:
                    other_node.relationships.pop(0)
    
    def _neural_consolidation(self, node: MemoryNode):
        """Neural consolidation process"""
        # Increase importance for frequently accessed memories
        node.importance = min(1.0, node.importance * 1.1)
        
        # Strengthen relationships
        for related_id in node.relationships[:5]:
            if related_id in self.memory_nodes:
                related_node = self.memory_nodes[related_id]
                related_node.importance = min(1.0, related_node.importance * 1.05)
    
    def _generate_memory_id(self, content: str) -> str:
        """Generate unique memory ID"""
        timestamp = datetime.now().isoformat()
        combined = f"{content}_{timestamp}_{np.random.random()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _init_databases(self):
        """Initialize memory databases"""
        self.conn = sqlite3.connect('data/memory.db')
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                embedding BLOB,
                timestamp DATETIME,
                importance REAL,
                access_count INTEGER,
                metadata TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                memory_id1 TEXT,
                memory_id2 TEXT,
                strength REAL,
                FOREIGN KEY (memory_id1) REFERENCES memories (id),
                FOREIGN KEY (memory_id2) REFERENCES memories (id)
            )
        ''')
        
        self.conn.commit()
    
    def _save_to_database(self, node: MemoryNode):
        """Save memory to database"""
        try:
            # Convert embedding to bytes
            embedding_bytes = node.embedding.tobytes()
            metadata_json = json.dumps(node.metadata)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, embedding, timestamp, importance, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (node.id, node.content, node.memory_type, embedding_bytes,
                  node.timestamp, node.importance, node.access_count, metadata_json))
            
            # Save relationships
            for related_id in node.relationships:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO relationships (memory_id1, memory_id2, strength)
                    VALUES (?, ?, ?)
                ''', (node.id, related_id, 1.0))
            
            self.conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
    
    def _load_memories(self):
        """Load memories from database"""
        try:
            self.cursor.execute('SELECT * FROM memories')
            rows = self.cursor.fetchall()
            
            for row in rows:
                memory_id, content, memory_type, embedding_bytes, timestamp, \
                importance, access_count, metadata_json = row
                
                # Convert bytes back to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                
                # Parse metadata
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Create memory node
                node = MemoryNode(
                    id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    embedding=embedding,
                    timestamp=datetime.fromisoformat(timestamp),
                    importance=importance,
                    access_count=access_count,
                    relationships=[],
                    metadata=metadata
                )
                
                self.memory_nodes[memory_id] = node
            
            # Load relationships
            self.cursor.execute('SELECT * FROM relationships')
            relationships = self.cursor.fetchall()
            
            for mem_id1, mem_id2, strength in relationships:
                if mem_id1 in self.memory_nodes and mem_id2 in self.memory_nodes:
                    self.memory_nodes[mem_id1].relationships.append(mem_id2)
            
            print(f"✅ {len(self.memory_nodes)}টি স্মৃতি লোড করা হয়েছে")
            
        except Exception as e:
            print(f"স্মৃতি লোড করার সময় ত্রুটি: {e}")
            self.memory_nodes = OrderedDict()
    
    def _remove_from_database(self, memory_id: str):
        """Remove memory from database"""
        try:
            self.cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            self.cursor.execute('DELETE FROM relationships WHERE memory_id1 = ? OR memory_id2 = ?', 
                               (memory_id, memory_id))
            self.conn.commit()
        except Exception as e:
            print(f"স্মৃতি মুছে ফেলার সময় ত্রুটি: {e}")

class MemoryRetrievalNetwork(nn.Module):
    """Neural network for memory retrieval"""
    def __init__(self, memory_dim=512, query_dim=768, hidden_dim=1024):
        super().__init__()
        self.query_encoder = nn.Linear(query_dim, hidden_dim)
        self.memory_encoder = nn.Linear(memory_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.scorer = nn.Linear(hidden_dim, 1)
        
    def forward(self, query_embeddings, memory_embeddings):
        query_proj = self.query_encoder(query_embeddings)
        memory_proj = self.memory_encoder(memory_embeddings)
        
        # Attention between query and memories
        attn_output, _ = self.attention(query_proj, memory_proj, memory_proj)
        
        # Score each memory
        scores = self.scorer(attn_output)
        return torch.sigmoid(scores)

class LRUCache:
    """LRU Cache for fast memory access"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str):
        """Get item from cache"""
        if key not in self.cache:
            return None
        
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def remove(self, key: str):
        """Remove item from cache"""
        self.cache.pop(key, None)

class MemoryConsolidationScheduler:
    """Schedules memory consolidation"""
    def __init__(self, memory_system: AdvancedMemorySystem):
        self.memory_system = memory_system
        self.consolidation_interval = 3600  # 1 hour
        self.last_consolidation = datetime.now()
    
    def should_consolidate(self) -> bool:
        """Check if consolidation should occur"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_consolidation).total_seconds()
        return time_diff >= self.consolidation_interval
    
    def consolidate_if_needed(self):
        """Consolidate if needed"""
        if self.should_consolidate():
            print("🔄 স্মৃতি একত্রীকরণ চলছে...")
            self.memory_system.consolidate()
            self.last_consolidation = datetime.now()
            print("✅ স্মৃতি একত্রীকরণ সম্পন্ন")

# Additional memory utilities
class MemoryAnalyzer:
    """Analyzes memory patterns and usage"""
    def __init__(self, memory_system: AdvancedMemorySystem):
        self.memory_system = memory_system
    
    def analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        access_counts = []
        importance_scores = []
        
        for node in self.memory_system.memory_nodes.values():
            access_counts.append(node.access_count)
            importance_scores.append(node.importance)
        
        return {
            "total_memories": len(access_counts),
            "avg_access_count": np.mean(access_counts) if access_counts else 0,
            "avg_importance": np.mean(importance_scores) if importance_scores else 0,
            "most_accessed": max(access_counts) if access_counts else 0,
            "least_accessed": min(access_counts) if access_counts else 0
        }
    
    def find_memory_gaps(self, topics: List[str]) -> List[str]:
        """Find gaps in knowledge/memory"""
        gaps = []
        
        for topic in topics:
            topic_memories = []
            for node in self.memory_system.memory_nodes.values():
                if topic.lower() in node.content.lower():
                    topic_memories.append(node)
            
            if len(topic_memories) < 3:  # Fewer than 3 memories about topic
                gaps.append(f"'{topic}' সম্পর্কে কম জ্ঞান")
        
        return gaps

class MemoryVisualizer:
    """Visualizes memory structures"""
    def __init__(self, memory_system: AdvancedMemorySystem):
        self.memory_system = memory_system
    
    def create_memory_graph(self, center_memory_id: str, depth: int = 3) -> Dict[str, Any]:
        """Create graph representation of memories"""
        graph = {
            "nodes": [],
            "edges": []
        }
        
        visited = set()
        
        def add_node(node_id: str, current_depth: int):
            if current_depth > depth or node_id in visited:
                return
            
            visited.add(node_id)
            node = self.memory_system.memory_nodes.get(node_id)
            if node:
                graph["nodes"].append({
                    "id": node_id,
                    "label": node.content[:50] + ("..." if len(node.content) > 50 else ""),
                    "type": node.memory_type,
                    "importance": node.importance,
                    "depth": current_depth
                })
                
                # Add edges to related nodes
                for related_id in node.relationships[:10]:
                    if related_id in self.memory_system.memory_nodes:
                        graph["edges"].append({
                            "from": node_id,
                            "to": related_id,
                            "strength": 1.0 / (current_depth + 1)
                        })
                        
                        if related_id not in visited:
                            add_node(related_id, current_depth + 1)
        
        add_node(center_memory_id, 0)
        return graph