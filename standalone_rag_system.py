 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полная автономная RAG система с Qdrant
Автоматически устанавливает зависимости и запускает демонстрацию
"""

import os
import sys
import subprocess
import time
import logging
import uuid
import json
import pickle
from typing import List, Dict, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Установка пакета с проверкой"""
    try:
        __import__(package_name.split('==')[0].replace('-', '_'))
        logger.info(f"✅ {package_name} уже установлен")
        return True
    except ImportError:
        logger.info(f"🔧 Устанавливаем {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            logger.info(f"✅ {package_name} установлен")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Ошибка установки {package_name}: {e}")
            return False

def install_dependencies():
    """Установка всех необходимых зависимостей"""
    logger.info("🚀 Проверяем и устанавливаем зависимости...")
    
    required_packages = [
        "numpy",
        "torch",
        "sentence-transformers",
        "qdrant-client",
        "requests"
    ]
    
    for package in required_packages:
        if not install_package(package):
            logger.error(f"❌ Критическая ошибка: не удалось установить {package}")
            return False
    
    logger.info("✅ Все зависимости установлены")
    return True

# Устанавливаем зависимости
if not install_dependencies():
    logger.error("❌ Не удалось установить зависимости")
    sys.exit(1)

# Импортируем после установки
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import requests

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Qdrant client не доступен, используем файловое хранилище")
    QDRANT_AVAILABLE = False


class LocalVectorStore:
    """Локальное векторное хранилище как fallback для Qdrant"""
    
    def __init__(self, storage_path="vector_storage"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.vectors_file = os.path.join(storage_path, "vectors.npy")
        self.metadata_file = os.path.join(storage_path, "metadata.json")
        
        self.vectors = []
        self.metadata = []
        self.load_data()
    
    def load_data(self):
        """Загрузка данных из файлов"""
        try:
            if os.path.exists(self.vectors_file):
                self.vectors = np.load(self.vectors_file).tolist()
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            logger.info(f"📂 Загружено {len(self.vectors)} векторов из локального хранилища")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки данных: {e}")
            self.vectors = []
            self.metadata = []
    
    def save_data(self):
        """Сохранение данных в файлы"""
        try:
            if self.vectors:
                np.save(self.vectors_file, np.array(self.vectors))
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных: {e}")
    
    def add_vectors(self, vectors, metadata):
        """Добавление векторов"""
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)
        self.save_data()
    
    def search(self, query_vector, top_k=5):
        """Поиск похожих векторов"""
        if not self.vectors:
            return []
        
        vectors_array = np.array(self.vectors)
        query_array = np.array(query_vector)
        
        # Косинусное сходство
        similarities = np.dot(vectors_array, query_array) / (
            np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query_array)
        )
        
        # Топ-K результатов
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx],
                "index": int(idx)
            })
        
        return results


class SmartRAGSystem:
    """Умная RAG система с автоматическим выбором хранилища"""
    
    def __init__(self):
        logger.info("🚀 Инициализация Smart RAG системы...")
        
        # Инициализация модели эмбеддингов
        self._init_embedding_model()
        
        # Попытка подключения к Qdrant
        self.qdrant_client = None
        self.use_qdrant = self._try_connect_qdrant()
        
        if not self.use_qdrant:
            logger.info("📁 Используем локальное векторное хранилище")
            self.local_store = LocalVectorStore()
        
        self.collection_name = "smart_rag_documents"
    
    def _init_embedding_model(self):
        """Инициализация модели эмбеддингов"""
        logger.info("🧠 Загружаем модель эмбеддингов...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ Устройство: {device}")
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ Модель загружена, размерность: {self.vector_size}")
    
    def _try_connect_qdrant(self):
        """Попытка подключения к Qdrant"""
        if not QDRANT_AVAILABLE:
            return False
        
        try:
            # Проверяем, доступен ли Qdrant
            response = requests.get("http://localhost:6333/health", timeout=2)
            if response.status_code == 200:
                self.qdrant_client = QdrantClient(host="localhost", port=6333)
                self._create_qdrant_collection()
                logger.info("✅ Подключено к Qdrant")
                return True
        except:
            pass
        
        # Попытка запустить Qdrant через Docker
        try:
            logger.info("🐳 Пытаемся запустить Qdrant через Docker...")
            subprocess.run([
                "docker", "run", "-d", "--name", "qdrant_smart_rag", 
                "-p", "6333:6333", "qdrant/qdrant"
            ], check=True, capture_output=True)
            
            time.sleep(5)  # Ждем запуска
            
            response = requests.get("http://localhost:6333/health", timeout=5)
            if response.status_code == 200:
                self.qdrant_client = QdrantClient(host="localhost", port=6333)
                self._create_qdrant_collection()
                logger.info("✅ Qdrant запущен и подключен")
                return True
        except:
            pass
        
        logger.warning("⚠️ Qdrant недоступен, используем локальное хранилище")
        return False
    
    def _create_qdrant_collection(self):
        """Создание коллекции в Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"📦 Коллекция '{self.collection_name}' создана")
        except Exception as e:
            logger.error(f"❌ Ошибка создания коллекции: {e}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None) -> bool:
        """Добавление документов в хранилище"""
        if not texts:
            return False
        
        if metadata is None:
            metadata = [{"index": i, "text_preview": text[:100]} for i, text in enumerate(texts)]
        
        logger.info(f"📚 Добавляем {len(texts)} документов...")
        
        # Генерация эмбеддингов
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        if self.use_qdrant:
            return self._add_to_qdrant(texts, embeddings, metadata)
        else:
            return self._add_to_local(texts, embeddings, metadata)
    
    def _add_to_qdrant(self, texts, embeddings, metadata):
        """Добавление в Qdrant"""
        try:
            points = []
            for text, emb, meta in zip(texts, embeddings, metadata):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={"text": text, "metadata": meta}
                )
                points.append(point)
            
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"✅ {len(points)} документов добавлено в Qdrant")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в Qdrant: {e}")
            return False
    
    def _add_to_local(self, texts, embeddings, metadata):
        """Добавление в локальное хранилище"""
        try:
            local_metadata = []
            for text, meta in zip(texts, metadata):
                local_metadata.append({
                    "text": text,
                    "metadata": meta,
                    "id": str(uuid.uuid4())
                })
            
            self.local_store.add_vectors(embeddings.tolist(), local_metadata)
            logger.info(f"✅ {len(texts)} документов добавлено в локальное хранилище")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления в локальное хранилище: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск документов"""
        logger.info(f"🔍 Поиск: '{query}'")
        
        # Генерация эмбеддинга запроса
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        if self.use_qdrant:
            return self._search_qdrant(query_embedding, top_k)
        else:
            return self._search_local(query_embedding, top_k)
    
    def _search_qdrant(self, query_embedding, top_k):
        """Поиск в Qdrant"""
        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload.get("metadata", {})
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в Qdrant: {e}")
            return []
    
    def _search_local(self, query_embedding, top_k):
        """Поиск в локальном хранилище"""
        try:
            search_results = self.local_store.search(query_embedding, top_k)
            
            results = []
            for result in search_results:
                meta = result["metadata"]
                results.append({
                    "id": meta.get("id", "unknown"),
                    "score": result["score"],
                    "text": meta.get("text", ""),
                    "metadata": meta.get("metadata", {})
                })
            
            return results
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в локальном хранилище: {e}")
            return []
    
    def generate_rag_response(self, query: str) -> Dict:
        """Генерация RAG ответа"""
        # Поиск релевантных документов
        search_results = self.search(query, top_k=3)
        
        # Формирование контекста
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Документ {i} (релевантность: {result['score']:.3f}):\n{result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Создание промпта
        prompt = f"""Используй следующий контекст для ответа на вопрос.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ: На основе предоставленного контекста, """
        
        return {
            "query": query,
            "context": context,
            "prompt": prompt,
            "search_results": search_results,
            "storage_type": "Qdrant" if self.use_qdrant else "Локальное хранилище"
        }
    
    def get_stats(self) -> Dict:
        """Получение статистики системы"""
        if self.use_qdrant:
            try:
                info = self.qdrant_client.get_collection(self.collection_name)
                return {
                    "storage_type": "Qdrant",
                    "documents_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status
                }
            except:
                return {"storage_type": "Qdrant", "status": "error"}
        else:
            return {
                "storage_type": "Локальное хранилище",
                "documents_count": len(self.local_store.metadata),
                "vectors_count": len(self.local_store.vectors),
                "status": "active"
            }


def create_sample_documents():
    """Создание примеров документов"""
    return [
        "Python - высокоуровневый язык программирования общего назначения. Поддерживает ООП, функциональное и процедурное программирование. Имеет простой синтаксис и обширную экосистему библиотек.",
        
        "Qdrant - современная векторная база данных, написанная на Rust. Предназначена для высокопроизводительного векторного поиска и ML приложений. Поддерживает различные метрики расстояния.",
        
        "RAG (Retrieval-Augmented Generation) - архитектура, сочетающая поиск информации с генерацией текста. Использует векторный поиск для нахождения релевантных документов как контекста для LLM.",
        
        "BERT - модель глубокого обучения для понимания естественного языка. Использует механизм внимания и двунаправленное кодирование для создания контекстуальных представлений текста.",
        
        "Sentence Transformers - библиотека для создания семантических эмбеддингов предложений. Основана на BERT и других трансформерах. Позволяет находить семантически похожие тексты.",
        
        "Машинное обучение включает обучение с учителем (supervised), без учителя (unsupervised) и с подкреплением (reinforcement). Каждый тип решает разные классы задач.",
        
        "Векторный поиск превосходит традиционный поиск по ключевым словам, так как учитывает семантический смысл. Используется в рекомендательных системах и поисковиках.",
        
        "LangChain - фреймворк для создания приложений с языковыми моделями. Предоставляет компоненты для создания цепочек, агентов и сложных workflow с LLM."
    ]


def run_demonstration():
    """Запуск полной демонстрации"""
    logger.info("🎬 === ДЕМОНСТРАЦИЯ SMART RAG СИСТЕМЫ ===")
    logger.info("=" * 60)
    
    try:
        # Инициализация системы
        rag_system = SmartRAGSystem()
        
        # Статистика системы
        stats = rag_system.get_stats()
        logger.info(f"📊 Тип хранилища: {stats['storage_type']}")
        logger.info(f"📄 Документов: {stats.get('documents_count', 0)}")
        
        # Добавление примеров документов
        sample_docs = create_sample_documents()
        metadata = [
            {"topic": "programming", "language": "python"},
            {"topic": "database", "type": "vector"},
            {"topic": "ai", "method": "rag"},
            {"topic": "ai", "model": "bert"},
            {"topic": "ml", "library": "sentence-transformers"},
            {"topic": "ai", "type": "learning"},
            {"topic": "search", "method": "vector"},
            {"topic": "framework", "type": "langchain"}
        ]
        
        success = rag_system.add_documents(sample_docs, metadata)
        if not success:
            logger.error("❌ Не удалось добавить документы")
            return False
        
        # Тестовые запросы
        test_queries = [
            "Что такое Python и его основные особенности?",
            "Объясни принцип работы RAG",
            "Какие типы машинного обучения существуют?",
            "Как работает векторный поиск?",
            "Что такое BERT и как он используется?"
        ]
        
        logger.info(f"\n🔍 Тестируем {len(test_queries)} запросов...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"❓ Запрос {i}/{len(test_queries)}: {query}")
            logger.info("-" * 80)
            
            # Выполнение RAG запроса
            response = rag_system.generate_rag_response(query)
            
            logger.info(f"🏪 Хранилище: {response['storage_type']}")
            logger.info(f"📋 Найдено документов: {len(response['search_results'])}")
            
            # Показываем топ результаты
            for j, result in enumerate(response['search_results'][:2], 1):
                text_preview = result['text'][:120] + "..." if len(result['text']) > 120 else result['text']
                logger.info(f"  {j}. Релевантность: {result['score']:.3f}")
                logger.info(f"     {text_preview}")
            
            # Короткий промпт
            prompt_preview = response['prompt'][:200] + "..." if len(response['prompt']) > 200 else response['prompt']
            logger.info(f"💭 Промпт (начало): {prompt_preview}")
            
            time.sleep(0.5)  # Небольшая пауза для читаемости
        
        # Финальная статистика
        final_stats = rag_system.get_stats()
        logger.info(f"\n📊 === ФИНАЛЬНАЯ СТАТИСТИКА ===")
        logger.info(f"🏪 Тип хранилища: {final_stats['storage_type']}")
        logger.info(f"📄 Всего документов: {final_stats.get('documents_count', 0)}")
        logger.info(f"🔢 Векторов: {final_stats.get('vectors_count', 0)}")
        logger.info(f"⚡ Статус: {final_stats.get('status', 'unknown')}")
        
        logger.info(f"\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        
        # Дополнительная информация
        if rag_system.use_qdrant:
            logger.info("🌐 Qdrant dashboard: http://localhost:6333/dashboard")
        else:
            logger.info("📁 Данные сохранены в папке: vector_storage/")
        
        logger.info("💡 Система готова к использованию!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        logger.exception("Детали ошибки:")
        return False


if __name__ == "__main__":
    logger.info("🚀 Запуск Smart RAG System")
    
    success = run_demonstration()
    
    if success:
        logger.info("✅ Программа завершена успешно")
    else:
        logger.error("❌ Программа завершена с ошибками")
        sys.exit(1)