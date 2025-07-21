#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web App с новой RAG системой и локальной LLM
Интеграция rag_minimal.py, simple_rag_working.py и local_llm.py
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os

# Импорт наших RAG систем
try:
    from simple_rag_working import SimpleRAG, ADVANCED_MODE
    RAG_CLASS = SimpleRAG
    RAG_TYPE = "Advanced (Sentence Transformers)" if ADVANCED_MODE else "Basic (Text Search)"
except ImportError:
    from rag_minimal import MinimalRAG
    RAG_CLASS = MinimalRAG
    RAG_TYPE = "Minimal (TF-IDF)"

# Импорт локальной LLM
try:
    from local_llm import RAGWithLLM, LocalLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("❌ local_llm.py не найден, используем базовую генерацию")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание Flask приложения
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Глобальные переменные
rag_system = None
rag_with_llm = None
chat_sessions = {}  # Хранение сессий чата

def init_rag_system():
    """Инициализация RAG системы с LLM"""
    global rag_system, rag_with_llm
    
    if rag_system is None:
        logger.info(f"🚀 Инициализируем RAG систему: {RAG_TYPE}")
        rag_system = RAG_CLASS()
        
        # Добавляем демо документы
        demo_docs = [
            "Python - высокоуровневый язык программирования общего назначения. Поддерживает ООП, функциональное и процедурное программирование. Python известен своей простотой и читаемостью синтаксиса.",
            "RAG (Retrieval-Augmented Generation) - метод NLP, сочетающий поиск информации с генерацией текста. Использует векторный поиск для нахождения релевантных документов, а затем генерирует ответ на их основе.",
            "Векторный поиск позволяет находить семантически похожие документы используя векторные представления текста вместо точного совпадения ключевых слов. Это основа современных поисковых систем.",
            "Sentence Transformers - библиотека Python для создания качественных эмбеддингов предложений на основе моделей BERT и других трансформеров. Позволяет преобразовывать текст в векторы.",
            "Qdrant - векторная база данных с открытым исходным кодом на Rust. Предназначена для высокопроизводительного векторного поиска и ML приложений. Альтернатива FAISS.",
            "Машинное обучение включает три основных типа: обучение с учителем (supervised), без учителя (unsupervised) и обучение с подкреплением (reinforcement). Каждый тип решает разные задачи.",
            "BERT - двунаправленная модель энкодера на основе трансформеров. Предобучается на больших объемах текста для получения контекстуальных представлений слов и предложений.",
            "LangChain - фреймворк для создания приложений на основе языковых моделей. Предоставляет инструменты для цепочек, агентов и интеграций с различными LLM.",
            "Flask - микрофреймворк для создания веб-приложений на Python. Простой и гибкий для создания REST API и веб-интерфейсов. Хорошо подходит для прототипирования.",
            "TF-IDF (Term Frequency-Inverse Document Frequency) - численная статистика для оценки важности слова в документе относительно коллекции документов. Используется в поиске и анализе текста.",
            "Ollama - инструмент для запуска локальных языковых моделей. Поддерживает различные модели включая Llama, Mistral, CodeLlama. Работает без интернета.",
            "Transformer архитектура революционизировала NLP. Основана на механизме внимания (attention), позволяет обрабатывать последовательности параллельно, что делает обучение быстрее."
        ]
        
        rag_system.add_documents(demo_docs)
        logger.info("✅ RAG система инициализирована с демо документами")
        
        # Инициализация LLM если доступна
        if LLM_AVAILABLE:
            try:
                logger.info("🤖 Инициализируем локальную LLM...")
                rag_with_llm = RAGWithLLM(rag_system, "ollama", "llama3.2")
                logger.info("✅ LLM система готова")
            except Exception as e:
                logger.warning(f"⚠️ LLM недоступна: {e}")
                rag_with_llm = None
        else:
            rag_with_llm = None

def get_session_id():
    """Получение или создание session_id"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def generate_smart_answer(user_message: str, search_results: List[Dict]) -> str:
    """Генерация умного ответа с улучшенной логикой"""
    
    if not search_results:
        return "Извините, не нашел релевантных документов для вашего запроса. Попробуйте переформулировать вопрос."
    
    # Используем LLM если доступна
    if rag_with_llm:
        try:
            result = rag_with_llm.generate_rag_answer(user_message)
            return result["answer"]
        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            # Fallback на локальную генерацию
    
    # Локальная умная генерация
    question_lower = user_message.lower()
    
    # Формируем контекст из найденных документов
    context_parts = []
    for i, result in enumerate(search_results[:3], 1):
        text = result["text"]
        score = result["score"]
        context_parts.append(f"{text}")
    
    context = " ".join(context_parts)
    
    # Определяем тип вопроса и генерируем соответствующий ответ
    if any(word in question_lower for word in ["что такое", "что это", "определение", "означает"]):
        return generate_definition_answer(context, user_message, search_results)
    elif any(word in question_lower for word in ["как работает", "как", "процесс", "принцип"]):
        return generate_process_answer(context, user_message, search_results)
    elif any(word in question_lower for word in ["зачем", "почему", "для чего", "цель"]):
        return generate_purpose_answer(context, user_message, search_results)
    elif any(word in question_lower for word in ["где", "когда", "кто", "какой"]):
        return generate_factual_answer(context, user_message, search_results)
    elif "привет" in question_lower or "добр" in question_lower:
        return generate_greeting_answer(context, user_message, search_results)
    else:
        return generate_general_answer(context, user_message, search_results)

def generate_definition_answer(context: str, question: str, results: List[Dict]) -> str:
    """Генерация ответа-определения"""
    
    # Ищем определения в контексте
    sentences = context.split('.')
    definitions = []
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in [" - это", "представляет собой", "является"]):
            definitions.append(sentence.strip())
    
    if definitions:
        main_def = definitions[0]
        answer = f"{main_def}."
        
        if len(definitions) > 1:
            answer += f" {definitions[1]}."
            
        # Добавляем дополнительную информацию из результатов
        if len(results) > 1:
            additional_info = results[1]["text"].split('.')[0]
            answer += f" Также важно отметить: {additional_info}."
            
        return answer
    else:
        # Берем первый результат как основу ответа
        main_text = results[0]["text"]
        return f"Согласно найденной информации: {main_text}"

def generate_process_answer(context: str, question: str, results: List[Dict]) -> str:
    """Генерация ответа о процессе"""
    
    sentences = context.split('.')
    process_words = ["использует", "применяет", "работает", "выполняет", "сочетает", "позволяет", "основан"]
    
    process_sentences = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in process_words):
            process_sentences.append(sentence.strip())
    
    if process_sentences:
        answer = f"{process_sentences[0]}."
        if len(process_sentences) > 1:
            answer += f" {process_sentences[1]}."
        return answer
    else:
        return f"По найденной информации: {results[0]['text']}"

def generate_purpose_answer(context: str, question: str, results: List[Dict]) -> str:
    """Генерация ответа о назначении"""
    
    purpose_words = ["предназначена", "используется", "нужен", "для", "цель", "применяется"]
    sentences = context.split('.')
    
    purpose_sentences = []
    for sentence in sentences:
        if any(word in sentence.lower() for word in purpose_words):
            purpose_sentences.append(sentence.strip())
    
    if purpose_sentences:
        return f"Основная цель: {purpose_sentences[0]}."
    else:
        return f"Согласно документам: {results[0]['text']}"

def generate_factual_answer(context: str, question: str, results: List[Dict]) -> str:
    """Генерация фактуального ответа"""
    
    sentences = context.split('.')
    factual_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
    
    return f"Из найденных документов: {'. '.join(factual_sentences)}."

def generate_greeting_answer(context: str, question: str, results: List[Dict]) -> str:
    """Генерация ответа на приветствие"""
    
    greetings = [
        "Привет! Я готов помочь вам с вопросами.",
        "Добро пожаловать! Задавайте любые вопросы.",
        "Здравствуйте! Чем могу помочь?",
        "Привет! Готов ответить на ваши вопросы."
    ]
    
    import random
    base_greeting = random.choice(greetings)
    
    if results:
        topic = results[0]["text"].split('.')[0]
        return f"{base_greeting} Например, могу рассказать про: {topic}."
    else:
        return base_greeting

def generate_general_answer(context: str, question: str, results: List[Dict]) -> str:
    """Общая генерация ответа"""
    
    sentences = context.split('.')
    key_sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:2]
    
    return f"На основе найденной информации: {'. '.join(key_sentences)}."

@app.route('/')
def index():
    """Главная страница"""
    init_rag_system()
    
    # Статистика системы
    if hasattr(rag_system, 'get_stats'):
        stats = rag_system.get_stats()
    else:
        stats = {
            "total_documents": len(rag_system.documents) if hasattr(rag_system, 'documents') else 0,
            "method": RAG_TYPE
        }
    
    # Добавляем информацию о LLM
    stats["llm_available"] = LLM_AVAILABLE and rag_with_llm is not None
    
    return render_template('index_new.html', 
                         rag_type=RAG_TYPE, 
                         stats=stats)

@app.route('/chat', methods=['POST'])
def chat():
    """Обработка чат сообщений с улучшенной логикой"""
    try:
        init_rag_system()
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = get_session_id()
        
        if not user_message:
            return jsonify({'error': 'Пустое сообщение'}), 400
        
        # Засекаем время
        start_time = time.time()
        
        # Поиск документов
        search_results = rag_system.search(user_message, top_k=3)
        
        # Генерация умного ответа
        ai_response = generate_smart_answer(user_message, search_results)
        
        response_time = time.time() - start_time
        
        # Сохраняем в сессию
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'user_message': user_message,
            'ai_response': ai_response,
            'response_time': response_time,
            'documents_found': len(search_results),
            'search_results': search_results,
            'llm_used': LLM_AVAILABLE and rag_with_llm is not None
        }
        
        chat_sessions[session_id].append(chat_entry)
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'response_time': f"{response_time:.2f}s",
            'documents_found': len(search_results),
            'search_results': search_results[:3],  # Только топ-3 для UI
            'llm_used': LLM_AVAILABLE and rag_with_llm is not None
        })
        
    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search():
    """Поиск документов"""
    try:
        init_rag_system()
        
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Пустой запрос'}), 400
        
        start_time = time.time()
        results = rag_system.search(query, top_k=top_k)
        search_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'search_time': f"{search_time:.3f}s",
            'total_found': len(results)
        })
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({'error': f'Ошибка поиска: {str(e)}'}), 500

@app.route('/add_document', methods=['POST'])
def add_document():
    """Добавление нового документа"""
    try:
        init_rag_system()
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Пустой текст документа'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Документ слишком короткий (минимум 10 символов)'}), 400
        
        # Добавляем документ
        rag_system.add_documents([text])
        
        # Получаем обновленную статистику
        if hasattr(rag_system, 'get_stats'):
            stats = rag_system.get_stats()
        else:
            stats = {
                "total_documents": len(rag_system.documents) if hasattr(rag_system, 'documents') else 0
            }
        
        return jsonify({
            'success': True,
            'message': 'Документ успешно добавлен',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Ошибка добавления документа: {e}")
        return jsonify({'error': f'Ошибка добавления: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """Получение статистики системы"""
    try:
        init_rag_system()
        
        if hasattr(rag_system, 'get_stats'):
            system_stats = rag_system.get_stats()
        else:
            system_stats = {
                "total_documents": len(rag_system.documents) if hasattr(rag_system, 'documents') else 0,
                "method": RAG_TYPE
            }
        
        # Добавляем информацию о LLM
        system_stats["llm_available"] = LLM_AVAILABLE and rag_with_llm is not None
        if rag_with_llm:
            system_stats["llm_type"] = rag_with_llm.llm.model_type
            system_stats["llm_model"] = rag_with_llm.llm.model_name
        
        session_id = get_session_id()
        chat_history = chat_sessions.get(session_id, [])
        
        return jsonify({
            'system_stats': system_stats,
            'rag_type': RAG_TYPE,
            'chat_messages': len(chat_history),
            'session_id': session_id,
            'llm_available': system_stats["llm_available"]
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500

@app.route('/chat_history')
def chat_history():
    """Получение истории чата"""
    session_id = get_session_id()
    history = chat_sessions.get(session_id, [])
    
    return jsonify({
        'success': True,
        'history': history,
        'total_messages': len(history)
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Очистка истории чата"""
    session_id = get_session_id()
    if session_id in chat_sessions:
        chat_sessions[session_id] = []
    
    return jsonify({'success': True, 'message': 'История чата очищена'})

@app.route('/health')
def health():
    """Проверка здоровья приложения"""
    try:
        init_rag_system()
        return jsonify({
            'status': 'healthy',
            'rag_type': RAG_TYPE,
            'llm_available': LLM_AVAILABLE and rag_with_llm is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Запуск Flask приложения с новой RAG системой и LLM")
    logger.info(f"📊 Тип RAG: {RAG_TYPE}")
    logger.info(f"🤖 LLM доступна: {LLM_AVAILABLE}")
    
    # Инициализация при запуске
    init_rag_system()
    
    # Запуск приложения
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    ) 