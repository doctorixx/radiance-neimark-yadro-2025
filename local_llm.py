#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальная LLM для генерации ответов в RAG системе
Поддерживает Ollama, GPT4All и Hugging Face модели
"""

import requests
import json
import logging
from typing import Optional, Dict, List
import os

logger = logging.getLogger(__name__)

class LocalLLM:
    """Класс для работы с локальными языковыми моделями"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "llama3.2"):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.base_url = "http://localhost:11434"  # Ollama default
        
        # Проверяем доступность модели
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Проверка доступности модели"""
        if self.model_type == "ollama":
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [m["name"] for m in models]
                    
                    if self.model_name not in available_models:
                        logger.warning(f"Модель {self.model_name} не найдена. Доступные: {available_models}")
                        # Используем первую доступную модель
                        if available_models:
                            self.model_name = available_models[0]
                            logger.info(f"Переключаемся на модель: {self.model_name}")
                    
                    logger.info(f"✅ Ollama доступен, модель: {self.model_name}")
                else:
                    raise Exception("Ollama не отвечает")
                    
            except Exception as e:
                logger.error(f"❌ Ollama недоступен: {e}")
                logger.info("Переключаемся на встроенную модель...")
                self.model_type = "builtin"
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Генерация ответа от LLM"""
        
        if self.model_type == "ollama":
            return self._generate_ollama(prompt, max_tokens)
        elif self.model_type == "huggingface":
            return self._generate_huggingface(prompt, max_tokens)
        else:
            return self._generate_builtin(prompt)
    
    def _generate_ollama(self, prompt: str, max_tokens: int) -> str:
        """Генерация через Ollama"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Ошибка генерации ответа").strip()
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return self._generate_builtin(prompt)
                
        except Exception as e:
            logger.error(f"Ошибка Ollama: {e}")
            return self._generate_builtin(prompt)
    
    def _generate_huggingface(self, prompt: str, max_tokens: int) -> str:
        """Генерация через Hugging Face (локальная модель)"""
        try:
            # Попытка использовать transformers
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            
            # Используем небольшую модель для локального запуска
            model_name = "microsoft/DialoGPT-medium"
            
            if not hasattr(self, '_hf_generator'):
                logger.info("Загружаем Hugging Face модель...")
                self._hf_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    max_length=max_tokens,
                    device=-1  # CPU
                )
            
            # Генерируем ответ
            result = self._hf_generator(
                prompt,
                max_length=len(prompt.split()) + max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            # Убираем исходный промпт из ответа
            response = generated_text[len(prompt):].strip()
            
            return response if response else "Не удалось сгенерировать ответ"
            
        except ImportError:
            logger.warning("Transformers не установлен")
            return self._generate_builtin(prompt)
        except Exception as e:
            logger.error(f"Ошибка Hugging Face: {e}")
            return self._generate_builtin(prompt)
    
    def _generate_builtin(self, prompt: str) -> str:
        """Встроенная генерация ответов (улучшенная логика)"""
        
        # Извлекаем контекст и вопрос из промпта
        if "КОНТЕКСТ:" in prompt and "ВОПРОС:" in prompt:
            try:
                context_start = prompt.find("КОНТЕКСТ:") + len("КОНТЕКСТ:")
                question_start = prompt.find("ВОПРОС:") + len("ВОПРОС:")
                
                context = prompt[context_start:prompt.find("ВОПРОС:")].strip()
                question = prompt[question_start:].strip()
                
                # Генерируем ответ на основе контекста
                return self._smart_response_generation(context, question)
                
            except Exception as e:
                logger.error(f"Ошибка парсинга промпта: {e}")
        
        # Простая генерация если не удалось распарсить
        return self._simple_response_generation(prompt)
    
    def _smart_response_generation(self, context: str, question: str) -> str:
        """Умная генерация ответов на основе контекста"""
        
        # Ключевые слова для анализа вопроса
        question_lower = question.lower()
        
        # Определяем тип вопроса
        if any(word in question_lower for word in ["что такое", "что это", "определение"]):
            return self._generate_definition_answer(context, question)
        elif any(word in question_lower for word in ["как", "каким образом", "процесс"]):
            return self._generate_process_answer(context, question)
        elif any(word in question_lower for word in ["зачем", "почему", "для чего"]):
            return self._generate_purpose_answer(context, question)
        elif any(word in question_lower for word in ["где", "когда", "кто"]):
            return self._generate_factual_answer(context, question)
        else:
            return self._generate_general_answer(context, question)
    
    def _generate_definition_answer(self, context: str, question: str) -> str:
        """Генерация ответа-определения"""
        
        # Ищем определения в контексте
        sentences = context.split('.')
        definitions = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in [" - это", "представляет собой", "является"]):
                definitions.append(sentence.strip())
        
        if definitions:
            main_def = definitions[0]
            answer = f"Согласно найденным документам, {main_def}"
            
            if len(definitions) > 1:
                answer += f"\n\nТакже стоит отметить: {definitions[1]}"
                
            return answer
        else:
            # Извлекаем ключевые предложения
            key_sentences = sentences[:2]
            return f"На основе документов: {'. '.join(key_sentences)}."
    
    def _generate_process_answer(self, context: str, question: str) -> str:
        """Генерация ответа о процессе"""
        
        # Ищем описания процессов
        sentences = context.split('.')
        process_words = ["использует", "применяет", "работает", "выполняет", "сочетает", "позволяет"]
        
        process_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in process_words):
                process_sentences.append(sentence.strip())
        
        if process_sentences:
            return f"По данным из документов: {'. '.join(process_sentences[:2])}."
        else:
            return f"Согласно найденной информации: {sentences[0].strip()}."
    
    def _generate_purpose_answer(self, context: str, question: str) -> str:
        """Генерация ответа о назначении"""
        
        purpose_words = ["предназначена", "используется", "нужен", "для", "цель", "применяется"]
        sentences = context.split('.')
        
        purpose_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in purpose_words):
                purpose_sentences.append(sentence.strip())
        
        if purpose_sentences:
            return f"Основное назначение: {'. '.join(purpose_sentences[:2])}."
        else:
            return f"Согласно документам: {sentences[0].strip()}."
    
    def _generate_factual_answer(self, context: str, question: str) -> str:
        """Генерация фактуального ответа"""
        
        sentences = context.split('.')
        # Берем наиболее информативные предложения
        factual_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
        
        return f"Согласно найденным документам: {'. '.join(factual_sentences)}."
    
    def _generate_general_answer(self, context: str, question: str) -> str:
        """Общая генерация ответа"""
        
        sentences = context.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:3]
        
        return f"На основе анализа документов: {'. '.join(key_sentences)}."
    
    def _simple_response_generation(self, prompt: str) -> str:
        """Простая генерация если не удалось распарсить промпт"""
        
        if "python" in prompt.lower():
            return "Python - это высокоуровневый язык программирования, известный своей простотой и читаемостью."
        elif "rag" in prompt.lower():
            return "RAG (Retrieval-Augmented Generation) - это метод, который сочетает поиск информации с генерацией текста."
        elif "машинное обучение" in prompt.lower():
            return "Машинное обучение - это область искусственного интеллекта, которая позволяет компьютерам обучаться на данных."
        else:
            return "Извините, не могу предоставить точный ответ на основе доступной информации."


class RAGWithLLM:
    """RAG система с интеграцией локальной LLM"""
    
    def __init__(self, rag_system, llm_type: str = "ollama", model_name: str = "llama3.2"):
        self.rag_system = rag_system
        self.llm = LocalLLM(llm_type, model_name)
        
    def generate_rag_answer(self, question: str, max_context_length: int = 2000) -> Dict:
        """Генерация полного RAG ответа с LLM"""
        
        # 1. Поиск релевантных документов
        search_results = self.rag_system.search(question, top_k=3)
        
        if not search_results:
            return {
                "question": question,
                "answer": "Извините, не найдено релевантных документов для вашего вопроса.",
                "context": "",
                "search_results": [],
                "llm_used": False
            }
        
        # 2. Формирование контекста
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(search_results, 1):
            text = result["text"]
            score = result["score"]
            
            formatted = f"Документ {i} (релевантность: {score:.3f}):\n{text}"
            
            if total_length + len(formatted) > max_context_length:
                break
            
            context_parts.append(formatted)
            total_length += len(formatted)
        
        context = "\n\n".join(context_parts)
        
        # 3. Создание промпта для LLM
        prompt = f"""Ты - полезный ассистент. Ответь на вопрос пользователя, используя только информацию из предоставленного контекста. Отвечай на русском языке четко и по существу.

КОНТЕКСТ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
        
        # 4. Генерация ответа через LLM
        try:
            llm_answer = self.llm.generate_response(prompt)
            
            # Улучшаем ответ если он слишком короткий
            if len(llm_answer.strip()) < 20:
                llm_answer = self.llm._smart_response_generation(context, question)
            
            return {
                "question": question,
                "answer": llm_answer,
                "context": context,
                "search_results": search_results,
                "llm_used": True,
                "model_type": self.llm.model_type,
                "model_name": self.llm.model_name
            }
            
        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            # Fallback на встроенную генерацию
            fallback_answer = self.llm._smart_response_generation(context, question)
            
            return {
                "question": question,
                "answer": fallback_answer,
                "context": context,
                "search_results": search_results,
                "llm_used": False,
                "error": str(e)
            }


# Простой тест
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тест локальной LLM
    llm = LocalLLM("ollama", "llama3.2")
    
    test_prompt = """Ты - полезный ассистент. Ответь на вопрос пользователя.

КОНТЕКСТ:
Python - высокоуровневый язык программирования общего назначения. Поддерживает ООП, функциональное и процедурное программирование.

ВОПРОС: Что такое Python?

ОТВЕТ:"""
    
    response = llm.generate_response(test_prompt)
    print(f"Ответ LLM: {response}") 