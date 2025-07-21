#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for text processing, embeddings, and document parsing with GPU support
"""

import re
import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import logging
from config import config

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Russian, English, numbers, and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\n]', '', text, flags=re.UNICODE)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    @staticmethod
    def split_into_chunks(text: str, max_chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        max_chunk_size = max_chunk_size or config.MAX_CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        
        if not text or len(text) <= max_chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_ends = ['.', '!', '?', '\n']
                best_end = end
                
                for i in range(max(start + max_chunk_size // 2, end - 100), end):
                    if i < len(text) and text[i] in sentence_ends:
                        best_end = i + 1
                
                end = best_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks

class DocumentParser:
    """Document parsing utilities"""
    
    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Parse TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error parsing TXT {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Parse PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Parse DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def parse_document(file_path: str) -> str:
        """Parse document based on extension"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return DocumentParser.parse_txt(file_path)
        elif ext == '.pdf':
            return DocumentParser.parse_pdf(file_path)
        elif ext == '.docx':
            return DocumentParser.parse_docx(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""

class EmbeddingEngine:
    """GPU-accelerated embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model with GPU support"""
        model_name = model_name or config.EMBEDDING_MODEL
        
        try:
            # Initialize model
            self.model = SentenceTransformer(model_name)
            
            # Move to GPU if available and configured
            if config.USE_GPU and torch.cuda.is_available():
                device = config.EMBEDDING_DEVICE
                self.model = self.model.to(device)
                logger.info(f"Загружен embedding model: {model_name} на {device}")
                
                # Optimize for GPU
                if hasattr(self.model, '_modules'):
                    for module in self.model._modules.values():
                        if hasattr(module, 'half'):
                            try:
                                module.half()  # Use FP16 for faster inference
                            except:
                                pass
            else:
                logger.info(f"Загружен embedding model: {model_name} на CPU")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_texts(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings with GPU acceleration"""
        if not texts:
            return np.array([])
        
        try:
            # Adjust batch size for GPU memory
            if config.USE_GPU and torch.cuda.is_available():
                # Calculate optimal batch size based on GPU memory
                gpu_memory_gb = torch.cuda.get_device_properties(config.CUDA_DEVICE).total_memory / (1024**3)
                optimal_batch_size = min(batch_size, max(8, int(gpu_memory_gb * 4)))
            else:
                optimal_batch_size = min(batch_size, 16)  # Conservative for CPU
            
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=normalize,
                batch_size=optimal_batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return np.array([])
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode single text to embedding"""
        if not text:
            return np.array([])
        
        try:
            embedding = self.model.encode(
                [text], 
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embedding[0]
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.array([])
    
    def get_device_info(self) -> Dict[str, str]:
        """Get device information"""
        if hasattr(self.model, 'device'):
            device = str(self.model.device)
        else:
            device = 'cpu'
            
        return {
            'device': device,
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

def load_documents_from_directory(directory: str) -> List[Dict[str, str]]:
    """Load all documents from directory with GPU-optimized chunking"""
    documents = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return documents
    
    supported_extensions = {'.txt', '.pdf', '.docx'}
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            # Check if file extension is supported
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                logger.debug(f"Skipping unsupported file: {filename}")
                continue
                
            text = DocumentParser.parse_document(file_path)
            
            if text:
                # Clean text
                cleaned_text = TextProcessor.clean_text(text)
                
                # Split into chunks with config parameters
                chunks = TextProcessor.split_into_chunks(
                    cleaned_text,
                    max_chunk_size=config.MAX_CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP
                )
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'filename': filename,
                        'chunk_id': i,
                        'text': chunk,
                        'source': f"{filename}:chunk_{i}",
                        'file_size': len(text),
                        'chunk_size': len(chunk)
                    })
    
    logger.info(f"Загружено {len(documents)} фрагментов из {directory}")
    return documents

def get_system_info() -> Dict[str, any]:
    """Get system information for debugging"""
    info = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'config': {
            'use_gpu': config.USE_GPU,
            'cuda_device': config.CUDA_DEVICE,
            'embedding_device': config.EMBEDDING_DEVICE,
            'faiss_use_gpu': config.FAISS_USE_GPU
        }
    }
    
    if torch.cuda.is_available():
        info['gpu_info'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_reserved': torch.cuda.memory_reserved(0)
        }
    
    return info 