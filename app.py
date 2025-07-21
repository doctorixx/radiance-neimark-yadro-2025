#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated RAG-enabled Flask Chat Bot with Vector Memory and Document Search
"""

from flask import Flask, render_template, request, jsonify, session
from typing import Dict, List, Optional, Tuple
import uuid
import time
import logging
from datetime import datetime
from dataclasses import dataclass

# Import our RAG modules with GPU support
from config import config
from llm_engine import llm_engine, create_system_prompt
from memory import memory_manager
from retriever import document_retriever
from utils import TextProcessor, get_system_info

# Configure logging from config
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    timestamp: str
    user_message: str
    ai_response: str
    response_time: float
    model: str
    context_used: bool
    documents_found: int

class RAGChatService:
    """Main service for GPU-accelerated RAG-enabled chat"""
    
    def __init__(self):
        """Initialize RAG chat service with GPU support"""
        self.llm_engine = llm_engine
        self.memory_manager = memory_manager
        self.document_retriever = document_retriever
        
        logger.info("RAG Chat Service инициализирован с GPU поддержкой")
        logger.info(f"Системная информация: {get_system_info()}")
    
    def process_message(self, message: str, session_id: str, model: str = None) -> Tuple[bool, str, float, Dict]:
        """Process user message with GPU-accelerated RAG pipeline"""
        model = model or config.OLLAMA_MODEL
        start_time = time.time()
        
        try:
            # Clean user message
            cleaned_message = TextProcessor.clean_text(message)
            
            # 1. Search relevant documents (GPU accelerated)
            relevant_docs = self.document_retriever.search_documents(
                cleaned_message, 
                top_k=config.DOCUMENT_SEARCH_TOP_K
            )
            
            # 2. Search memory for relevant context (GPU accelerated)
            memory_context = self.memory_manager.search_session_memory(
                session_id, 
                cleaned_message, 
                top_k=config.MEMORY_SEARCH_TOP_K
            )
            
            # 3. Create system prompt with context
            system_prompt = create_system_prompt(
                context_documents=relevant_docs,
                memory_context=memory_context
            )
            
            # 4. Generate response using LLM (Ollama or Mistral API)
            success, ai_response, llm_time = self.llm_engine.generate_response(
                prompt=cleaned_message,
                system_prompt=system_prompt,
                model=model
            )
            
            if success:
                # 5. Add user message and AI response to memory (GPU accelerated)
                self.memory_manager.add_message_to_session(
                    session_id, cleaned_message, "user"
                )
                self.memory_manager.add_message_to_session(
                    session_id, ai_response, "assistant"
                )
                
                response_time = time.time() - start_time
                
                # Prepare context info
                context_info = {
                    'documents_found': len(relevant_docs),
                    'memory_context_found': len(memory_context),
                    'context_used': len(relevant_docs) > 0 or len(memory_context) > 0,
                    'model_used': model,
                    'gpu_enabled': config.USE_GPU
                }
                
                return True, ai_response, response_time, context_info
            else:
                return False, ai_response, llm_time, {
                    'documents_found': 0, 
                    'memory_context_found': 0, 
                    'context_used': False,
                    'model_used': model,
                    'gpu_enabled': config.USE_GPU
                }
                
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            return False, f"Ошибка обработки сообщения: {str(e)}", 0.0, {
                'documents_found': 0, 
                'memory_context_found': 0, 
                'context_used': False,
                'model_used': model,
                'gpu_enabled': config.USE_GPU
            }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status information"""
        doc_stats = self.document_retriever.get_document_stats()
        memory_stats = self.memory_manager.get_all_session_stats()
        engine_info = self.llm_engine.get_engine_info()
        gpu_info = self.document_retriever.get_gpu_info()
        system_info = get_system_info()
        
        return {
            'llm_available': engine_info['ollama_available'],
            'available_models': engine_info['available_models'],
            'mistral_api_available': engine_info['mistral_api_available'],
            'documents_loaded': doc_stats,
            'active_sessions': len(memory_stats),
            'memory_stats': memory_stats,
            'gpu_info': gpu_info,
            'system_info': system_info,
            'config': {
                'use_gpu': config.USE_GPU,
                'faiss_gpu': config.FAISS_USE_GPU,
                'embedding_device': config.EMBEDDING_DEVICE,
                'chunk_size': config.MAX_CHUNK_SIZE,
                'similarity_threshold': config.SIMILARITY_THRESHOLD
            }
        }

class ConversationManager:
    """Manages conversation history for web interface"""
    
    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
    
    def add_message(self, session_id: str, user_msg: str, ai_response: str, 
                   response_time: float, model: str, context_info: Dict) -> None:
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = ChatMessage(
            timestamp=datetime.now().isoformat(),
            user_message=user_msg,
            ai_response=ai_response,
            response_time=response_time,
            model=model,
            context_used=context_info.get('context_used', False),
            documents_found=context_info.get('documents_found', 0)
        )
        
        self.conversations[session_id].append(message)
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history for web interface"""
        if session_id not in self.conversations:
            return []
        
        return [
            {
                'timestamp': msg.timestamp,
                'user': msg.user_message,
                'assistant': msg.ai_response,
                'response_time': msg.response_time,
                'model': msg.model,
                'context_used': msg.context_used,
                'documents_found': msg.documents_found
            }
            for msg in self.conversations[session_id]
        ]
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history"""
        if session_id in self.conversations:
            self.conversations[session_id] = []

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'rag-chat-bot-secret-key-change-in-production'

# Initialize services
rag_service = RAGChatService()
conversation_manager = ConversationManager()

@app.route('/')
def index():
    """Main chat page"""
    # Create unique session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Get system status
    system_status = rag_service.get_system_status()
    
    return render_template(
        'chat.html',
        system_status=system_status,
        session_id=session['session_id']
    )

@app.route('/send_message', methods=['POST'])
def send_message():
    """Process user message through GPU-accelerated RAG pipeline"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        model = data.get('model', config.OLLAMA_MODEL)
        session_id = session.get('session_id')
        
        # Validate input
        if not message:
            return jsonify({'success': False, 'error': 'Пустое сообщение'})
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Нет ID сессии'})
        
        # Process message through RAG pipeline
        success, response, response_time, context_info = rag_service.process_message(
            message, session_id, model
        )
        
        if success:
            # Save to conversation history
            conversation_manager.add_message(
                session_id, message, response, response_time, model, context_info
            )
            
            return jsonify({
                'success': True,
                'response': response,
                'response_time': response_time,
                'context_info': context_info
            })
        else:
            return jsonify({
                'success': False,
                'error': response,
                'context_info': context_info
            })
            
    except Exception as e:
        logger.error(f"Ошибка в send_message: {e}")
        return jsonify({'success': False, 'error': 'Внутренняя ошибка сервера'})

@app.route('/get_conversation')
def get_conversation():
    """Get conversation history"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify([])
    
    conversation = conversation_manager.get_conversation(session_id)
    return jsonify(conversation)

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation and memory"""
    session_id = session.get('session_id')
    if session_id:
        # Clear web conversation
        conversation_manager.clear_conversation(session_id)
        
        # Clear vector memory (GPU memory will be cleaned up)
        memory_manager.clear_session_memory(session_id)
    
    return jsonify({'success': True})

@app.route('/check_status')
def check_status():
    """Check comprehensive system status"""
    return jsonify(rag_service.get_system_status())

@app.route('/refresh_documents', methods=['POST'])
def refresh_documents():
    """Refresh document index with GPU acceleration"""
    try:
        document_retriever.refresh_index()
        return jsonify({'success': True, 'message': 'Индекс документов обновлен (GPU ускорение)'})
    except Exception as e:
        logger.error(f"Ошибка обновления документов: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload text document directly with GPU processing"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        source = data.get('source', 'manual_upload')
        
        if not text:
            return jsonify({'success': False, 'error': 'Пустой текст'})
        
        success = document_retriever.add_document_text(text, source)
        
        if success:
            return jsonify({'success': True, 'message': 'Документ добавлен в индекс (GPU обработка)'})
        else:
            return jsonify({'success': False, 'error': 'Не удалось добавить документ'})
            
    except Exception as e:
        logger.error(f"Ошибка загрузки документа: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/memory_stats')
def memory_stats():
    """Get GPU memory statistics for current session"""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({})
    
    if session_id in memory_manager.sessions:
        stats = memory_manager.sessions[session_id].get_memory_stats()
        return jsonify(stats)
    else:
        return jsonify({
            'session_id': session_id, 
            'total_messages': 0, 
            'index_size': 0,
            'gpu_enabled': config.FAISS_USE_GPU
        })

@app.route('/gpu_info')
def gpu_info():
    """Get detailed GPU information"""
    gpu_info = {
        'retriever': document_retriever.get_gpu_info(),
        'memory': memory_manager.get_gpu_info(),
        'system': get_system_info()
    }
    return jsonify(gpu_info)

@app.route('/cleanup_gpu', methods=['POST'])
def cleanup_gpu():
    """Cleanup GPU memory manually"""
    try:
        memory_manager.cleanup_gpu_memory()
        return jsonify({'success': True, 'message': 'GPU память очищена'})
    except Exception as e:
        logger.error(f"Ошибка очистки GPU памяти: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return "404 - Страница не найдена", 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Внутренняя ошибка: {error}")
    return "500 - Внутренняя ошибка сервера", 500

if __name__ == '__main__':
    logger.info("🚀 Запуск GPU-ускоренного RAG Chat Bot...")
    logger.info(f"📊 Статус системы: {rag_service.get_system_status()}")
    
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT
    ) 
