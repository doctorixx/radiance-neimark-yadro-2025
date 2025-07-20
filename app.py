#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Interface for Langflow AI Chat
Enhanced with proper JSON response parsing
"""

from flask import Flask, render_template, request, jsonify, session
from typing import Dict, List, Optional, Tuple
import os
import time
import uuid
import logging
import requests
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import threading
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class for chat messages"""
    timestamp: str
    user_message: str
    ai_response: str
    response_time: float
    model: str
    message_id: str = None
    reaction: Optional[str] = None  # 'like', 'dislike', or None

class JSONStorage:
    """JSON file storage for conversations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.conversations_file = self.data_dir / "conversations.json"
        self._lock = threading.Lock()
        logger.info(f"JSONStorage initialized with data directory: {self.data_dir}")
    
    def load_conversations(self) -> Dict[str, List[ChatMessage]]:
        """Load conversations from JSON file"""
        try:
            if not self.conversations_file.exists():
                logger.info("No existing conversations file found, starting fresh")
                return {}
            
            with self._lock:
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                conversations = {}
                for session_id, messages_data in data.items():
                    conversations[session_id] = [
                        ChatMessage(**msg_data) for msg_data in messages_data
                    ]
                
                logger.info(f"Loaded {len(conversations)} conversation sessions from storage")
                return conversations
                
        except Exception as e:
            logger.error(f"Error loading conversations: {e}", exc_info=True)
            return {}
    
    def save_conversations(self, conversations: Dict[str, List[ChatMessage]]) -> None:
        """Save conversations to JSON file"""
        try:
            data = {}
            for session_id, messages in conversations.items():
                data[session_id] = [asdict(msg) for msg in messages]
            
            with self._lock:
                # Atomic write using temporary file
                temp_file = self.conversations_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Replace original file
                temp_file.replace(self.conversations_file)
            
            logger.debug(f"Saved {len(conversations)} conversation sessions to storage")
            
        except Exception as e:
            logger.error(f"Error saving conversations: {e}", exc_info=True)

class ExternalAPIService:
    """Service for sending data to external API"""
    
    API_URL = "http://192.168.44.157:5000"
    
    @classmethod
    def send_feedback_data(cls, user_query: str, ai_answer: str, reaction: Optional[str]) -> bool:
        """Send feedback data to external API"""
        try:
            # Определяем score на основе реакции
            score = 0
            if reaction == 'like':
                score = 1
            elif reaction == 'dislike':
                score = -1
            
            payload = {
                "answer": ai_answer,
                "category": "chat",  # Можно настроить категорию
                "score": score,
                "user_query": user_query,
                "version": "1.0"  # Версия приложения
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                cls.API_URL,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent feedback data to external API")
                return True
            else:
                logger.warning(f"External API returned status {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending data to external API: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to external API: {e}")
            return False

class MistralPredictionService:
    """Service for predicting next user questions using Mistral API"""
    
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get Mistral API key from environment"""
        return os.getenv("MISTRAL_API_KEY")
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if Mistral API is properly configured"""
        return bool(cls.get_api_key())
    
    @classmethod
    def predict_next_questions(cls, conversation_history: List[ChatMessage]) -> List[str]:
        """Predict next questions based on conversation history"""
        logger.info(f"Predicting questions for {len(conversation_history)} messages")
        
        if not cls.is_configured():
            logger.warning("Mistral API key not configured")
            return []
        
        if len(conversation_history) == 0:
            # Если нет истории, предлагаем стандартные вопросы
            logger.info("No history, returning default questions")
            return [
                "Как дела?",
                "Расскажи о себе",
                "Помоги с задачей"
            ]
        
        try:
            # Строим контекст из последних сообщений
            context = cls._build_context(conversation_history)
            
            system_prompt = """Ты - эксперт по анализу диалогов. Твоя задача - предсказать 5 следующих вопросов пользователя разных типов на основе истории диалога.

Типы вопросов:
1. ИССЛЕДОВАТЕЛЬСКИЙ - глубокий, аналитический вопрос
2. ПРАКТИЧЕСКИЙ - как применить, использовать  
3. ПОДРОБНЫЙ - запрос деталей, объяснений
4. БЫСТРЫЙ - короткий, конкретный вопрос
5. РАЗВИВАЮЩИЙ - логическое продолжение темы

Правила:
- Анализируй логический ход разговора
- Учитывай интересы и контекст пользователя
- Каждый вопрос должен быть в своём стиле
- Вопросы должны быть короткими (до 60 символов)
- Отвечай ТОЛЬКО JSON массивом из 5 строк, без дополнительного текста

Пример ответа:
["Как это влияет на производительность?", "Покажи код", "Расскажи подробнее про алгоритм", "Работает?", "А что насчёт безопасности?"]"""

            user_prompt = f"История диалога:\n{context}\n\nПредскажи 5 следующих вопросов пользователя разных типов (исследовательский, практический, подробный, быстрый, развивающий):"
            
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cls.get_api_key()}"
            }
            
            response = requests.post(
                cls.MISTRAL_API_URL,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content'].strip()
                logger.info(f"Raw Mistral response: {content}")
                
                # Парсим JSON ответ
                try:
                    predictions = json.loads(content)
                    if isinstance(predictions, list) and len(predictions) > 0:
                        logger.info(f"Generated {len(predictions)} question predictions")
                        return predictions[:5]  # Берём максимум 5
                    else:
                        logger.warning("Invalid predictions format from Mistral")
                        return []
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Mistral JSON: {e}")
                    logger.error(f"Content was: {content}")
                    # Fallback - пытаемся извлечь массив из текста
                    if '[' in content and ']' in content:
                        try:
                            start = content.find('[')
                            end = content.rfind(']') + 1
                            json_part = content[start:end]
                            predictions = json.loads(json_part)
                            if isinstance(predictions, list):
                                logger.info(f"Extracted {len(predictions)} predictions from text")
                                return predictions[:5]
                        except:
                            pass
                    return []
            else:
                logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Mistral API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mistral response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in question prediction: {e}")
            return []
    
    @classmethod
    def _build_context(cls, messages: List[ChatMessage]) -> str:
        """Build context string from recent messages"""
        # Берём последние 5 сообщений для контекста
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"Пользователь: {msg.user_message}")
            context_parts.append(f"AI: {msg.ai_response[:200]}...")  # Обрезаем длинные ответы
        
        return "\n".join(context_parts)

class LangflowAgentService:
    """Service for interacting with Langflow Agent"""
    
    # Configuration
    API_URL = "http://localhost:7860/api/v1/run/432ecd36-30d5-4f87-88b0-5524a717aea7"
    
    @classmethod
    def get_headers(cls) -> dict:
        """Get headers with API key"""
        api_key = os.getenv("LANGFLOW_API_KEY")
        if not api_key:
            logger.warning("LANGFLOW_API_KEY not found in environment variables")
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key or ""
        }
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if Langflow is properly configured"""
        return bool(os.getenv("LANGFLOW_API_KEY"))
    
    @classmethod
    def extract_response_text(cls, response_data: dict) -> str:
        """Extract text response from complex Langflow JSON structure"""
        try:
            # Try to navigate through the complex response structure
            if 'outputs' in response_data and isinstance(response_data['outputs'], list):
                for output in response_data['outputs']:
                    if 'outputs' in output and isinstance(output['outputs'], list):
                        for inner_output in output['outputs']:
                            if 'results' in inner_output and 'message' in inner_output['results']:
                                message = inner_output['results']['message']
                                if 'text' in message:
                                    return message['text']
                                if 'data' in message and 'text' in message['data']:
                                    return message['data']['text']
                            
                            # Alternative path for some flows
                            if 'messages' in inner_output and isinstance(inner_output['messages'], list):
                                for msg in inner_output['messages']:
                                    if 'message' in msg:
                                        return msg['message']
            
            # Fallback to string representation if structure not recognized
            return str(response_data)
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return str(response_data)
    
    @classmethod
    def send_message(
        cls, 
        message: str, 
        model: str, 
        conversation_history: List[ChatMessage]
    ) -> Tuple[bool, str, float]:
        """Send message to Langflow Agent"""
        if not cls.is_configured():
            return False, "Langflow API key not configured", 0.0
        
        payload = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": message
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                cls.API_URL,
                json=payload,
                headers=cls.get_headers(),
                timeout=120
            )
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            # Parse JSON and extract text response
            try:
                response_data = response.json()
                text_response = cls.extract_response_text(response_data)
                return True, text_response, response_time
            except json.JSONDecodeError:
                return True, response.text.strip(), response_time
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Langflow request error: {e}")
            return False, f"Langflow error: {str(e)}", time.time() - start_time
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False, f"Error: {str(e)}", time.time() - start_time

class ConversationManager:
    """Manages conversation sessions with JSON storage"""
    
    def __init__(self, storage: JSONStorage):
        self.storage = storage
        self.conversations: Dict[str, List[ChatMessage]] = self.storage.load_conversations()
        logger.info(f"ConversationManager initialized with {len(self.conversations)} existing sessions")
    
    def _save_to_storage(self) -> None:
        """Save current conversations to storage"""
        self.storage.save_conversations(self.conversations)
    
    def add_message(self, session_id: str, user_msg: str, ai_response: str, 
                   response_time: float, model: str) -> str:
        """Add message to conversation"""
        if session_id not in self.conversations:
            logger.debug(f"New conversation session started: {session_id}")
            self.conversations[session_id] = []
        
        message_id = str(uuid.uuid4())
        message = ChatMessage(
            timestamp=datetime.now().isoformat(),
            user_message=user_msg,
            ai_response=ai_response,
            response_time=response_time,
            model=model,
            message_id=message_id
        )
        
        self.conversations[session_id].append(message)
        self._save_to_storage()  # Auto-save after adding message
        logger.debug(f"Message added to session {session_id} and saved to storage")
        return message_id
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        if session_id not in self.conversations:
            return []
        
        return [
            {
                'timestamp': msg.timestamp,
                'user': msg.user_message,
                'assistant': msg.ai_response,
                'response_time': msg.response_time,
                'model': msg.model,
                'message_id': msg.message_id,
                'reaction': msg.reaction
            }
            for msg in self.conversations[session_id]
        ]
    
    def add_reaction(self, session_id: str, message_id: str, reaction: str) -> bool:
        """Add reaction to a message"""
        if session_id not in self.conversations:
            return False
        
        for message in self.conversations[session_id]:
            if message.message_id == message_id:
                # Сохраняем реакцию локально
                message.reaction = reaction if reaction in ['like', 'dislike'] else None
                self._save_to_storage()  # Auto-save after reaction change
                
                # Отправляем данные на внешний API параллельно (в фоне)
                threading.Thread(
                    target=self._send_to_external_api,
                    args=(message.user_message, message.ai_response, message.reaction),
                    daemon=True
                ).start()
                
                logger.info(f"Reaction '{reaction}' added to message {message_id} in session {session_id} and saved")
                return True
        
        return False
    
    def _send_to_external_api(self, user_query: str, ai_answer: str, reaction: Optional[str]) -> None:
        """Send data to external API in background thread"""
        success = ExternalAPIService.send_feedback_data(user_query, ai_answer, reaction)
        if success:
            logger.debug("Data successfully sent to external API")
        else:
            logger.warning("Failed to send data to external API (will continue with local storage)")
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history"""
        if session_id in self.conversations:
            self.conversations[session_id] = []
            self._save_to_storage()  # Auto-save after clearing
            logger.info(f"Conversation cleared for session {session_id} and saved to storage")
    
    def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get raw messages for prompt building"""
        return self.conversations.get(session_id, [])

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
logger.info("Flask application initialized")

# Initialize services
langflow_service = LangflowAgentService()
json_storage = JSONStorage()
external_api_service = ExternalAPIService()
mistral_service = MistralPredictionService()
conversation_manager = ConversationManager(json_storage)

@app.route('/')
def index():
    """Main page"""
    try:
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            logger.info(f"New session created: {session['session_id']}")
        
        langflow_configured = langflow_service.is_configured()
        logger.debug(f"Rendering index page for session {session['session_id']}")
        
        return render_template(
            'index.html',
            models=["Langflow Agent"],
            langflow_configured=langflow_configured,
            session_id=session['session_id']
        )
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        return render_template('error.html'), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    """Send message to AI"""
    try:
        logger.debug("Received send_message request")
        data = request.get_json()
        message = data.get('message', '').strip()
        model = data.get('model', 'Langflow Agent')
        session_id = session.get('session_id')
        
        if not message:
            logger.warning("Empty message received")
            return jsonify({'success': False, 'error': 'Empty message'})
        
        if not session_id:
            logger.error("No session ID found")
            return jsonify({'success': False, 'error': 'No session ID'})
        
        logger.info(f"Processing message for session {session_id}")
        
        history = conversation_manager.get_messages(session_id)
        success, response, response_time = langflow_service.send_message(
            message, model, history
        )
        
        if success:
            message_id = conversation_manager.add_message(
                session_id, message, response, response_time, model
            )
            logger.info(f"Message processed successfully in {response_time:.2f}s")
            return jsonify({
                'success': True,
                'response': response,
                'response_time': response_time,
                'message_id': message_id
            })
        else:
            logger.error(f"Failed to process message: {response}")
            return jsonify({
                'success': False,
                'error': response
            })
            
    except Exception as e:
        logger.error(f"Error in send_message: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        })

@app.route('/get_conversation')
def get_conversation():
    """Get conversation history"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify([])
        
        conversation = conversation_manager.get_conversation(session_id)
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Error in get_conversation: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get conversation'}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    try:
        session_id = session.get('session_id')
        if session_id:
            conversation_manager.clear_conversation(session_id)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'No session ID'}), 400
    except Exception as e:
        logger.error(f"Error in clear_conversation: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/add_reaction', methods=['POST'])
def add_reaction():
    """Add reaction to a message"""
    try:
        data = request.get_json()
        message_id = data.get('message_id')
        reaction = data.get('reaction')  # 'like', 'dislike', or None
        session_id = session.get('session_id')
        
        if not message_id:
            return jsonify({'success': False, 'error': 'Missing message_id'}), 400
        
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID'}), 400
        
        success = conversation_manager.add_reaction(session_id, message_id, reaction)
        
        if success:
            logger.info(f"Reaction '{reaction}' added to message {message_id}")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Message not found'}), 404
            
    except Exception as e:
        logger.error(f"Error in add_reaction: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_predictions')
def get_predictions():
    """Get predicted next questions"""
    try:
        session_id = session.get('session_id')
        logger.info(f"Get predictions request for session: {session_id}")
        
        if not session_id:
            logger.warning("No session ID found")
            return jsonify({'predictions': []})
        
        # Получаем историю диалога для данной сессии
        conversation_history = conversation_manager.get_messages(session_id)
        logger.info(f"Conversation history length: {len(conversation_history)}")
        
        # Генерируем предсказания
        predictions = mistral_service.predict_next_questions(conversation_history)
        logger.info(f"Generated predictions: {predictions}")
        
        result = {
            'predictions': predictions,
            'mistral_configured': mistral_service.is_configured()
        }
        logger.info(f"Sending response: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_predictions: {e}", exc_info=True)
        return jsonify({'predictions': [], 'error': str(e)})

@app.route('/check_status')
def check_status():
    """Check Langflow status"""
    try:
        status = {
            'langflow_configured': langflow_service.is_configured(),
            'mistral_configured': mistral_service.is_configured(),
            'models': ["Langflow Agent"]
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in check_status: {e}", exc_info=True)
        return jsonify({'error': 'Failed to check status'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 Internal Server Error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting application")
    app.run(debug=True, host='0.0.0.0', port=5000)