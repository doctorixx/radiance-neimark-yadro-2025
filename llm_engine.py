#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Engine for interacting with Mistral via Ollama API and external Mistral API
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from config import config

logger = logging.getLogger(__name__)

class OllamaLLMEngine:
    """Engine for interacting with Ollama API and external Mistral API"""
    
    def __init__(self, base_url: str = None):
        """Initialize Ollama LLM Engine with configuration"""
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.models_url = f"{self.base_url}/api/tags"
        self.default_model = config.OLLAMA_MODEL
        
        # Mistral API configuration
        self.mistral_api_key = config.MISTRAL_API_KEY
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        
        logger.info(f"LLM Engine инициализирован с Ollama URL: {self.base_url}")
        if self.mistral_api_key:
            logger.info("✓ Mistral API ключ найден")
        else:
            logger.info("✗ Mistral API ключ не установлен, используется только Ollama")
        
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(self.models_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama сервис недоступен: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                # Add Mistral API model if key is available
                if self.mistral_api_key:
                    models.extend(['mistral-api', 'mistral-small', 'mistral-medium', 'mistral-large'])
                
                return models
            else:
                logger.error(f"Не удалось получить модели: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка получения моделей: {e}")
            return []
    
    def generate_response(self, prompt: str, system_prompt: str = "", 
                         model: str = None, max_tokens: int = 2048,
                         temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using appropriate API"""
        model = model or self.default_model
        
        # Check if it's a Mistral API model
        if model.startswith('mistral-') and self.mistral_api_key:
            return self._generate_mistral_api_response(
                prompt, system_prompt, model, max_tokens, temperature
            )
        else:
            return self._generate_ollama_response(
                prompt, system_prompt, model, max_tokens, temperature
            )
    
    def _generate_ollama_response(self, prompt: str, system_prompt: str = "", 
                                 model: str = "mistral", max_tokens: int = 2048,
                                 temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using Ollama API"""
        
        # Build full prompt with system context
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,
                headers={'Content-Type': 'application/json'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('response', '').strip()
                
                if generated_text:
                    return True, generated_text, response_time
                else:
                    return False, "Пустой ответ от модели", response_time
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Ошибка Ollama API: {error_msg}")
                return False, error_msg, response_time
                
        except requests.exceptions.Timeout:
            return False, "Таймаут запроса", 0.0
        except requests.exceptions.ConnectionError:
            return False, "Ошибка соединения - Ollama запущен?", 0.0
        except Exception as e:
            logger.error(f"Ошибка вызова Ollama API: {e}")
            return False, f"Ошибка: {str(e)}", 0.0
    
    def _generate_mistral_api_response(self, prompt: str, system_prompt: str = "",
                                      model: str = "mistral-small", max_tokens: int = 2048,
                                      temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using external Mistral API"""
        
        if not self.mistral_api_key:
            return False, "Mistral API ключ не установлен", 0.0
        
        # Map model names
        model_mapping = {
            'mistral-api': 'mistral-small-latest',
            'mistral-small': 'mistral-small-latest',
            'mistral-medium': 'mistral-medium-latest', 
            'mistral-large': 'mistral-large-latest'
        }
        
        api_model = model_mapping.get(model, 'mistral-small-latest')
        
        # Build messages for chat API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.mistral_api_url,
                json=payload,
                headers=headers,
                timeout=120
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data['choices'][0]['message']['content'].strip()
                
                if generated_text:
                    return True, generated_text, response_time
                else:
                    return False, "Пустой ответ от Mistral API", response_time
            else:
                error_msg = f"Mistral API HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                return False, error_msg, response_time
                
        except requests.exceptions.Timeout:
            return False, "Таймаут Mistral API", 0.0
        except requests.exceptions.ConnectionError:
            return False, "Ошибка соединения с Mistral API", 0.0
        except Exception as e:
            logger.error(f"Ошибка вызова Mistral API: {e}")
            return False, f"Ошибка Mistral API: {str(e)}", 0.0
    
    def generate_chat_response(self, messages: List[Dict[str, str]], 
                              model: str = None, 
                              temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using chat API format"""
        model = model or self.default_model
        
        # Check if it's a Mistral API model
        if model.startswith('mistral-') and self.mistral_api_key:
            return self._generate_mistral_chat_response(messages, model, temperature)
        else:
            return self._generate_ollama_chat_response(messages, model, temperature)
    
    def _generate_ollama_chat_response(self, messages: List[Dict[str, str]], 
                                      model: str = "mistral", 
                                      temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using Ollama chat API format"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120,
                headers={'Content-Type': 'application/json'}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('message', {}).get('content', '').strip()
                
                if generated_text:
                    return True, generated_text, response_time
                else:
                    return False, "Пустой ответ от модели", response_time
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Ошибка Ollama chat API: {error_msg}")
                return False, error_msg, response_time
                
        except requests.exceptions.Timeout:
            return False, "Таймаут запроса", 0.0
        except requests.exceptions.ConnectionError:
            return False, "Ошибка соединения - Ollama запущен?", 0.0
        except Exception as e:
            logger.error(f"Ошибка вызова Ollama chat API: {e}")
            return False, f"Ошибка: {str(e)}", 0.0
    
    def _generate_mistral_chat_response(self, messages: List[Dict[str, str]],
                                       model: str = "mistral-small",
                                       temperature: float = 0.7) -> Tuple[bool, str, float]:
        """Generate response using external Mistral chat API"""
        
        if not self.mistral_api_key:
            return False, "Mistral API ключ не установлен", 0.0
        
        # Map model names
        model_mapping = {
            'mistral-api': 'mistral-small-latest',
            'mistral-small': 'mistral-small-latest',
            'mistral-medium': 'mistral-medium-latest',
            'mistral-large': 'mistral-large-latest'
        }
        
        api_model = model_mapping.get(model, 'mistral-small-latest')
        
        payload = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
            "top_p": 0.9
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.mistral_api_url,
                json=payload,
                headers=headers,
                timeout=120
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data['choices'][0]['message']['content'].strip()
                
                if generated_text:
                    return True, generated_text, response_time
                else:
                    return False, "Пустой ответ от Mistral API", response_time
            else:
                error_msg = f"Mistral API HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                return False, error_msg, response_time
                
        except requests.exceptions.Timeout:
            return False, "Таймаут Mistral API", 0.0
        except requests.exceptions.ConnectionError:
            return False, "Ошибка соединения с Mistral API", 0.0
        except Exception as e:
            logger.error(f"Ошибка вызова Mistral API: {e}")
            return False, f"Ошибка Mistral API: {str(e)}", 0.0
    
    def get_engine_info(self) -> Dict[str, any]:
        """Get engine information"""
        return {
            'ollama_available': self.is_available(),
            'ollama_url': self.base_url,
            'mistral_api_available': bool(self.mistral_api_key),
            'default_model': self.default_model,
            'available_models': self.get_available_models()
        }

def create_system_prompt(context_documents: List[str] = None, 
                        memory_context: List[str] = None) -> str:
    """Create system prompt for RAG-enabled chat"""
    
    system_prompt = """Ты — полезный AI-ассистент. Ты ведёшь диалог с человеком и помогаешь ему, используя:

1. Его прошлые реплики (если есть)
2. Информацию из документов (если релевантно) 
3. Свежий вопрос пользователя

Отвечай чётко, по существу, не выдумывай. Если чего-то не знаешь — попроси уточнение. Будь дружелюбным и кратким."""

    # Add document context if available
    if context_documents:
        system_prompt += "\n\nРЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:\n"
        for i, doc in enumerate(context_documents, 1):
            system_prompt += f"\n[Документ {i}]: {doc}\n"
    
    # Add memory context if available  
    if memory_context:
        system_prompt += "\n\nКОНТЕКСТ ИЗ ПРЕДЫДУЩИХ СООБЩЕНИЙ:\n"
        for i, memory in enumerate(memory_context, 1):
            system_prompt += f"\n[Контекст {i}]: {memory}\n"
    
    if context_documents or memory_context:
        system_prompt += "\n\nИспользуй эту информацию для формирования ответа, но отвечай естественно и не ссылайся на номера документов."
    
    return system_prompt

# Global instance
llm_engine = OllamaLLMEngine() 