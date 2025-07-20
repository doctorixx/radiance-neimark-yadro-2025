#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Interface for Local AI Chat
Clean architecture with proper error handling and type hints
"""

import logging
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, jsonify, session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Data class for chat messages"""

    timestamp: str
    user_message: str
    ai_response: str
    response_time: float
    model: str


class OllamaService:
    """Service for interacting with Ollama"""

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Ollama list failed: {result.stderr}")
                return []

            models = []
            for line in result.stdout.strip().split("\n")[1:]:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)

            return models

        except subprocess.TimeoutExpired:
            logger.error("Timeout getting models")
            return []
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    @staticmethod
    def is_running() -> bool:
        """Check if Ollama is running"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def send_message(
        message: str, model: str, conversation_history: List[ChatMessage]
    ) -> Tuple[bool, str, float]:
        """Send message to AI model"""
        try:
            # Build prompt with history
            prompt = OllamaService._build_prompt(message, conversation_history)

            start_time = time.time()

            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=600,
                encoding="utf-8",
                errors="ignore",
            )

            response_time = time.time() - start_time

            if result.returncode == 0:
                response = result.stdout.strip()
                return True, response, response_time
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return False, f"Error: {error_msg}", response_time

        except subprocess.TimeoutExpired:
            return False, "Request timeout", 0.0
        except Exception as e:
            return False, f"Error: {str(e)}", 0.0

    @staticmethod
    def _build_prompt(message: str, history: List[ChatMessage]) -> str:
        """Build prompt with conversation history"""
        prompt = "Ты полезный AI помощник. Отвечай на русском языке, будь дружелюбным и полезным.\n\n"

        # Add recent history (last 3 messages)
        for msg in history[-3:]:
            prompt += f"Пользователь: {msg.user_message}\n"
            prompt += f"Ассистент: {msg.ai_response}\n\n"

        # Add current message
        prompt += f"Пользователь: {message}\n"
        prompt += "Ассистент: "

        return prompt


class ConversationManager:
    """Manages conversation sessions"""

    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}

    def add_message(
        self,
        session_id: str,
        user_msg: str,
        ai_response: str,
        response_time: float,
        model: str,
    ) -> None:
        """Add message to conversation"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        message = ChatMessage(
            timestamp=datetime.now().isoformat(),
            user_message=user_msg,
            ai_response=ai_response,
            response_time=response_time,
            model=model,
        )

        self.conversations[session_id].append(message)

    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        if session_id not in self.conversations:
            return []

        return [
            {
                "timestamp": msg.timestamp,
                "user": msg.user_message,
                "assistant": msg.ai_response,
                "response_time": msg.response_time,
                "model": msg.model,
            }
            for msg in self.conversations[session_id]
        ]

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history"""
        if session_id in self.conversations:
            self.conversations[session_id] = []

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get raw messages for prompt building"""
        return self.conversations.get(session_id, [])


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-change-this-in-production"

# Initialize services
ollama_service = OllamaService()
conversation_manager = ConversationManager()


@app.route("/")
def index():
    """Main page"""
    # Create unique session ID
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    # Get available models and status
    models = ollama_service.get_available_models()
    ollama_running = ollama_service.is_running()

    return render_template(
        "index.html",
        models=models,
        ollama_running=ollama_running,
        session_id=session["session_id"],
    )


@app.route("/send_message", methods=["POST"])
def send_message():
    """Send message to AI"""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        model = data.get("model", "qwen2.5:7b")
        session_id = session.get("session_id")

        # Validate input
        if not message:
            return jsonify({"success": False, "error": "Empty message"})

        if not session_id:
            return jsonify({"success": False, "error": "No session ID"})

        # Get conversation history
        history = conversation_manager.get_messages(session_id)

        # Send message to AI
        success, response, response_time = ollama_service.send_message(
            message, model, history
        )

        if success:
            # Save to conversation
            conversation_manager.add_message(
                session_id, message, response, response_time, model
            )

            return jsonify(
                {"success": True, "response": response, "response_time": response_time}
            )
        else:
            return jsonify({"success": False, "error": response})

    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return jsonify({"success": False, "error": "Internal server error"})


@app.route("/get_conversation")
def get_conversation():
    """Get conversation history"""
    session_id = session.get("session_id")
    if not session_id:
        return jsonify([])

    conversation = conversation_manager.get_conversation(session_id)
    return jsonify(conversation)


@app.route("/clear_conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history"""
    session_id = session.get("session_id")
    if session_id:
        conversation_manager.clear_conversation(session_id)
    return jsonify({"success": True})


@app.route("/check_status")
def check_status():
    """Check Ollama status"""
    return jsonify(
        {
            "ollama_running": ollama_service.is_running(),
            "models": ollama_service.get_available_models(),
        }
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return "404 - Page not found", 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
