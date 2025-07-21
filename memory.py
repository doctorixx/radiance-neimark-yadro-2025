#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated vector memory system for chat sessions using FAISS
"""

import os
import pickle
import faiss
import numpy as np
import torch
from typing import List, Dict, Optional
import logging
from utils import EmbeddingEngine
from config import config

logger = logging.getLogger(__name__)

class ChatMemory:
    """GPU-accelerated vector memory for a single chat session"""
    
    def __init__(self, session_id: str, embedding_model: str = None):
        """Initialize chat memory for a session with GPU support"""
        self.session_id = session_id
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        
        # Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(embedding_model)
        
        # GPU configuration
        self.use_gpu = config.FAISS_USE_GPU and torch.cuda.is_available()
        self.gpu_device = config.FAISS_GPU_DEVICE if self.use_gpu else None
        
        # Initialize FAISS index and message storage
        if self.use_gpu:
            cpu_index = faiss.IndexFlatIP(self.dimension)
            res = faiss.StandardGpuResources()
            res.setTempMemoryFraction(config.GPU_MEMORY_FRACTION)
            self.gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_device, cpu_index)
            self.index = self.gpu_index
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            
        self.messages = []  # Store message texts
        self.message_metadata = []  # Store metadata (timestamps, roles, etc.)
        
        logger.debug(f"Инициализирована память для сессии {session_id} ({'GPU' if self.use_gpu else 'CPU'})")
    
    def add_to_memory(self, text: str, role: str = "user", metadata: Dict = None) -> None:
        """Add message to memory with GPU acceleration"""
        if not text.strip():
            return
        
        try:
            # Generate embedding
            embedding = self.embedding_engine.encode_text(text, normalize=True)
            
            if embedding.size == 0:
                logger.error("Не удалось сгенерировать эмбеддинг для сообщения")
                return
            
            # Add to FAISS index
            embedding_vector = embedding.reshape(1, -1).astype('float32')
            self.index.add(embedding_vector)
            
            # Store message and metadata
            self.messages.append(text)
            self.message_metadata.append({
                'role': role,
                'metadata': metadata or {},
                'index': len(self.messages) - 1
            })
            
            logger.debug(f"Добавлено сообщение в память: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Ошибка добавления в память: {e}")
    
    def search_memory(self, query: str, top_k: int = None, min_similarity: float = None) -> List[str]:
        """Search for relevant messages in memory with GPU acceleration"""
        top_k = top_k or config.MEMORY_SEARCH_TOP_K
        min_similarity = min_similarity or config.SIMILARITY_THRESHOLD
        
        if not query.strip() or len(self.messages) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_engine.encode_text(query, normalize=True)
            
            if query_embedding.size == 0:
                logger.error("Не удалось сгенерировать эмбеддинг запроса")
                return []
            
            # Search in FAISS index
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            k = min(top_k, len(self.messages))
            scores, indices = self.index.search(query_vector, k)
            
            # Extract relevant messages
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i >= 0 and i < len(self.messages) and score > min_similarity:
                    message = self.messages[i]
                    metadata = self.message_metadata[i]
                    
                    # Format message with role
                    role = metadata.get('role', 'unknown')
                    formatted_message = f"[{role}]: {message}"
                    results.append(formatted_message)
            
            logger.debug(f"Найдено {len(results)} релевантных воспоминаний для запроса: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска в памяти: {e}")
            return []
    
    def get_recent_messages(self, count: int = 5) -> List[str]:
        """Get recent messages from memory"""
        if not self.messages:
            return []
        
        recent_messages = []
        start_index = max(0, len(self.messages) - count)
        
        for i in range(start_index, len(self.messages)):
            message = self.messages[i]
            metadata = self.message_metadata[i]
            role = metadata.get('role', 'unknown')
            
            formatted_message = f"[{role}]: {message}"
            recent_messages.append(formatted_message)
        
        return recent_messages
    
    def clear_memory(self):
        """Clear all memory for this session"""
        # Clear GPU memory if used
        if self.use_gpu and hasattr(self, 'gpu_index'):
            del self.gpu_index
            torch.cuda.empty_cache()
        
        # Recreate index
        if self.use_gpu:
            cpu_index = faiss.IndexFlatIP(self.dimension)
            res = faiss.StandardGpuResources()
            res.setTempMemoryFraction(config.GPU_MEMORY_FRACTION)
            self.gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_device, cpu_index)
            self.index = self.gpu_index
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            
        self.messages = []
        self.message_metadata = []
        logger.debug(f"Очищена память для сессии {self.session_id}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        stats = {
            'session_id': self.session_id,
            'total_messages': len(self.messages),
            'index_size': self.index.ntotal if self.index else 0,
            'gpu_enabled': self.use_gpu
        }
        
        if self.use_gpu and torch.cuda.is_available():
            stats['gpu_device'] = self.gpu_device
            stats['gpu_memory_used'] = torch.cuda.memory_allocated(self.gpu_device)
        
        return stats

class MemoryManager:
    """Manages GPU-accelerated memory for multiple chat sessions"""
    
    def __init__(self, memory_dir: str = "vectorstore"):
        """Initialize memory manager with GPU support"""
        self.memory_dir = memory_dir
        self.sessions = {}  # Dict[session_id, ChatMemory]
        self.embedding_engine = EmbeddingEngine()
        
        # Ensure memory directory exists
        os.makedirs(memory_dir, exist_ok=True)
        
        logger.info(f"Memory manager инициализирован ({'GPU' if config.FAISS_USE_GPU and torch.cuda.is_available() else 'CPU'})")
    
    def get_session_memory(self, session_id: str) -> ChatMemory:
        """Get or create memory for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMemory(session_id)
            logger.debug(f"Создана новая память для сессии {session_id}")
        
        return self.sessions[session_id]
    
    def add_message_to_session(self, session_id: str, message: str, 
                              role: str = "user", metadata: Dict = None):
        """Add message to session memory"""
        memory = self.get_session_memory(session_id)
        memory.add_to_memory(message, role, metadata)
    
    def search_session_memory(self, session_id: str, query: str, 
                             top_k: int = None) -> List[str]:
        """Search memory for a specific session"""
        top_k = top_k or config.MEMORY_SEARCH_TOP_K
        
        if session_id not in self.sessions:
            return []
        
        memory = self.sessions[session_id]
        return memory.search_memory(query, top_k)
    
    def get_session_recent_messages(self, session_id: str, count: int = 5) -> List[str]:
        """Get recent messages for a session"""
        if session_id not in self.sessions:
            return []
        
        memory = self.sessions[session_id]
        return memory.get_recent_messages(count)
    
    def clear_session_memory(self, session_id: str):
        """Clear memory for a specific session"""
        if session_id in self.sessions:
            self.sessions[session_id].clear_memory()
            logger.info(f"Очищена память для сессии {session_id}")
    
    def remove_session(self, session_id: str):
        """Remove session completely"""
        if session_id in self.sessions:
            # Clear GPU memory before deletion
            if hasattr(self.sessions[session_id], 'use_gpu') and self.sessions[session_id].use_gpu:
                self.sessions[session_id].clear_memory()
            
            del self.sessions[session_id]
            logger.info(f"Удалена сессия {session_id}")
    
    def get_all_session_stats(self) -> Dict[str, Dict]:
        """Get statistics for all sessions"""
        stats = {}
        for session_id, memory in self.sessions.items():
            stats[session_id] = memory.get_memory_stats()
        return stats
    
    def save_session_memory(self, session_id: str):
        """Save session memory to disk"""
        if session_id not in self.sessions:
            return False
        
        try:
            memory = self.sessions[session_id]
            memory_file = os.path.join(self.memory_dir, f"memory_{session_id}.pkl")
            
            memory_data = {
                'messages': memory.messages,
                'metadata': memory.message_metadata,
                'session_id': session_id
            }
            
            with open(memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            
            # Save FAISS index (convert from GPU to CPU if needed)
            index_file = os.path.join(self.memory_dir, f"memory_{session_id}.faiss")
            if memory.index.ntotal > 0:
                if memory.use_gpu and hasattr(memory, 'gpu_index'):
                    cpu_index = faiss.index_gpu_to_cpu(memory.gpu_index)
                    faiss.write_index(cpu_index, index_file)
                else:
                    faiss.write_index(memory.index, index_file)
            
            logger.debug(f"Сохранена память для сессии {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти для сессии {session_id}: {e}")
            return False
    
    def load_session_memory(self, session_id: str) -> bool:
        """Load session memory from disk"""
        try:
            memory_file = os.path.join(self.memory_dir, f"memory_{session_id}.pkl")
            index_file = os.path.join(self.memory_dir, f"memory_{session_id}.faiss")
            
            if not os.path.exists(memory_file):
                return False
            
            # Load memory data
            with open(memory_file, 'rb') as f:
                memory_data = pickle.load(f)
            
            # Create new memory instance
            memory = ChatMemory(session_id)
            memory.messages = memory_data.get('messages', [])
            memory.message_metadata = memory_data.get('metadata', [])
            
            # Load FAISS index if exists
            if os.path.exists(index_file):
                cpu_index = faiss.read_index(index_file)
                
                if memory.use_gpu:
                    res = faiss.StandardGpuResources()
                    res.setTempMemoryFraction(config.GPU_MEMORY_FRACTION)
                    memory.gpu_index = faiss.index_cpu_to_gpu(res, memory.gpu_device, cpu_index)
                    memory.index = memory.gpu_index
                else:
                    memory.index = cpu_index
            
            self.sessions[session_id] = memory
            logger.debug(f"Загружена память для сессии {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки памяти для сессии {session_id}: {e}")
            return False
    
    def get_gpu_info(self) -> Dict[str, any]:
        """Get GPU information for all sessions"""
        info = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_enabled': config.FAISS_USE_GPU,
            'active_sessions': len(self.sessions),
            'sessions': {}
        }
        
        for session_id, memory in self.sessions.items():
            info['sessions'][session_id] = memory.get_memory_stats()
        
        if torch.cuda.is_available():
            info['gpu_memory'] = {
                'total': torch.cuda.get_device_properties(0).total_memory,
                'allocated': torch.cuda.memory_allocated(0),
                'reserved': torch.cuda.memory_reserved(0)
            }
        
        return info
    
    def cleanup_gpu_memory(self):
        """Cleanup GPU memory for all sessions"""
        if torch.cuda.is_available():
            for session_id, memory in self.sessions.items():
                if hasattr(memory, 'use_gpu') and memory.use_gpu:
                    if hasattr(memory, 'gpu_index'):
                        del memory.gpu_index
            
            torch.cuda.empty_cache()
            logger.info("Очищена GPU память для всех сессий")

# Global memory manager instance
memory_manager = MemoryManager() 