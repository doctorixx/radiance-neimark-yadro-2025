#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for RAG Chat Bot
"""

import os
import torch
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for RAG Chat Bot"""
    
    # Mistral API Configuration (единственная приватная настройка из .env)
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', '')
    
    # Ollama Configuration (публичные настройки)
    OLLAMA_BASE_URL = 'http://localhost:11434'
    OLLAMA_MODEL = 'mistral'
    
    # GPU Configuration (автоматически определяется)
    USE_GPU = torch.cuda.is_available()
    CUDA_DEVICE = 0
    GPU_MEMORY_FRACTION = 0.8
    
    # FAISS Configuration (используем GPU если доступен)
    FAISS_USE_GPU = torch.cuda.is_available()
    FAISS_GPU_DEVICE = 0
    
    # Embedding Configuration (оптимально под RTX 4060)
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Document Processing (оптимизировано)
    MAX_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    SIMILARITY_THRESHOLD = 0.1
    
    # Memory Configuration (быстрый поиск)
    MEMORY_SEARCH_TOP_K = 3
    DOCUMENT_SEARCH_TOP_K = 5
    
    # Flask Configuration (готово к продакшену)
    FLASK_DEBUG = True
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    
    # Logging
    LOG_LEVEL = 'INFO'
    
    @classmethod
    def validate_gpu_setup(cls):
        """Validate GPU setup and configuration"""
        if not torch.cuda.is_available():
            logger.warning("CUDA не доступна, переключаемся на CPU")
            cls.USE_GPU = False
            cls.FAISS_USE_GPU = False
            cls.EMBEDDING_DEVICE = 'cpu'
            return False
            
        gpu_count = torch.cuda.device_count()
        logger.info(f"Найдено {gpu_count} GPU устройств")
        
        if cls.CUDA_DEVICE >= gpu_count:
            logger.warning(f"GPU устройство {cls.CUDA_DEVICE} не найдено, используем GPU 0")
            cls.CUDA_DEVICE = 0
            cls.FAISS_GPU_DEVICE = 0
            cls.EMBEDDING_DEVICE = 'cuda:0'
        
        # Проверяем доступную память GPU
        gpu_memory = torch.cuda.get_device_properties(cls.CUDA_DEVICE).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        logger.info(f"GPU {cls.CUDA_DEVICE}: {torch.cuda.get_device_name(cls.CUDA_DEVICE)}")
        logger.info(f"Доступно памяти: {gpu_memory_gb:.1f} GB")
        
        if gpu_memory_gb < 4:
            logger.warning("Мало памяти GPU, уменьшаем batch size")
            cls.GPU_MEMORY_FRACTION = 0.6
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        logger.info("=== 🚀 RAG Chat Bot Configuration ===")
        logger.info(f"🖥️  GPU доступно: {torch.cuda.is_available()}")
        logger.info(f"⚡ Использовать GPU: {cls.USE_GPU}")
        logger.info(f"🎯 CUDA устройство: {cls.CUDA_DEVICE}")
        logger.info(f"🔍 FAISS GPU: {cls.FAISS_USE_GPU}")
        logger.info(f"🧠 Embedding устройство: {cls.EMBEDDING_DEVICE}")
        logger.info(f"🌐 Ollama URL: {cls.OLLAMA_BASE_URL}")
        logger.info(f"🤖 Модель: {cls.OLLAMA_MODEL}")
        logger.info(f"🔑 Mistral API: {'✓ установлен' if cls.MISTRAL_API_KEY else '✗ не установлен'}")
        logger.info("=====================================")

# Initialize configuration
config = Config()
config.setup_logging()
config.validate_gpu_setup()
config.print_config() 