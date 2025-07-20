#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Interface for Langflow AI Chat
Enhanced with proper JSON response parsing
"""

from flask import Flask, render_template, request, jsonify, session
from typing import Dict, List, Tuple
import os
import time
import uuid
import logging
import requests
import json
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
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
        return {"Content-Type": "application/json", "x-api-key": api_key or ""}

    @classmethod
    def is_configured(cls) -> bool:
        """Check if Langflow is properly configured"""
        return bool(os.getenv("LANGFLOW_API_KEY"))

    @classmethod
    def extract_response_text(cls, response_data: dict) -> str:
        """Extract text response from complex Langflow JSON structure"""
        try:
            # Try to navigate through the complex response structure
            if "outputs" in response_data and isinstance(
                response_data["outputs"], list
            ):
                for output in response_data["outputs"]:
                    if "outputs" in output and isinstance(output["outputs"], list):
                        for inner_output in output["outputs"]:
                            if (
                                "results" in inner_output
                                and "message" in inner_output["results"]
                            ):
                                message = inner_output["results"]["message"]
                                if "text" in message:
                                    return message["text"]
                                if "data" in message and "text" in message["data"]:
                                    return message["data"]["text"]

                            # Alternative path for some flows
                            if "messages" in inner_output and isinstance(
                                inner_output["messages"], list
                            ):
                                for msg in inner_output["messages"]:
                                    if "message" in msg:
                                        return msg["message"]

            # Fallback to string representation if structure not recognized
            return str(response_data)
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return str(response_data)

    @classmethod
    def send_message(
        cls, message: str, model: str, conversation_history: List[ChatMessage]
    ) -> Tuple[bool, str, float]:
        """Send message to Langflow Agent"""
        if not cls.is_configured():
            return False, "Langflow API key not configured", 0.0

        payload = {"output_type": "chat", "input_type": "chat", "input_value": message}

        start_time = time.time()
        try:
            response = requests.post(
                cls.API_URL, json=payload, headers=cls.get_headers(), timeout=120
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
    """Manages conversation sessions"""

    def __init__(self):
        self.conversations: Dict[str, List[ChatMessage]] = {}
        logger.info("ConversationManager initialized")

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
            logger.debug(f"New conversation session started: {session_id}")
            self.conversations[session_id] = []

        message = ChatMessage(
            timestamp=datetime.now().isoformat(),
            user_message=user_msg,
            ai_response=ai_response,
            response_time=response_time,
            model=model,
        )

        self.conversations[session_id].append(message)
        logger.debug(f"Message added to session {session_id}")

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
            logger.info(f"Conversation cleared for session {session_id}")

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get raw messages for prompt building"""
        return self.conversations.get(session_id, [])


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
logger.info("Flask application initialized")

# Initialize services
langflow_service = LangflowAgentService()
conversation_manager = ConversationManager()


@app.route("/")
def index():
    """Main page"""
    try:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
            logger.info(f"New session created: {session['session_id']}")

        langflow_configured = langflow_service.is_configured()
        logger.debug(f"Rendering index page for session {session['session_id']}")

        return render_template(
            "index.html",
            models=["Langflow Agent"],
            langflow_configured=langflow_configured,
            session_id=session["session_id"],
        )
    except Exception as e:
        logger.error(f"Error in index route: {e}", exc_info=True)
        return render_template("error.html"), 500


@app.route("/send_message", methods=["POST"])
def send_message():
    """Send message to AI"""
    try:
        logger.debug("Received send_message request")
        data = request.get_json()
        message = data.get("message", "").strip()
        model = data.get("model", "Langflow Agent")
        session_id = session.get("session_id")

        if not message:
            logger.warning("Empty message received")
            return jsonify({"success": False, "error": "Empty message"})

        if not session_id:
            logger.error("No session ID found")
            return jsonify({"success": False, "error": "No session ID"})

        logger.info(f"Processing message for session {session_id}")

        history = conversation_manager.get_messages(session_id)
        success, response, response_time = langflow_service.send_message(
            message, model, history
        )

        if success:
            conversation_manager.add_message(
                session_id, message, response, response_time, model
            )
            logger.info(f"Message processed successfully in {response_time:.2f}s")
            return jsonify(
                {"success": True, "response": response, "response_time": response_time}
            )
        else:
            logger.error(f"Failed to process message: {response}")
            return jsonify({"success": False, "error": response})

    except Exception as e:
        logger.error(f"Error in send_message: {e}", exc_info=True)
        return jsonify(
            {"success": False, "error": "Internal server error", "details": str(e)}
        )


@app.route("/get_conversation")
def get_conversation():
    """Get conversation history"""
    try:
        session_id = session.get("session_id")
        if not session_id:
            return jsonify([])

        conversation = conversation_manager.get_conversation(session_id)
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Error in get_conversation: {e}", exc_info=True)
        return jsonify({"error": "Failed to get conversation"}), 500


@app.route("/clear_conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history"""
    try:
        session_id = session.get("session_id")
        if session_id:
            conversation_manager.clear_conversation(session_id)
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "No session ID"}), 400
    except Exception as e:
        logger.error(f"Error in clear_conversation: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/check_status")
def check_status():
    """Check Langflow status"""
    try:
        status = {
            "langflow_configured": langflow_service.is_configured(),
            "models": ["Langflow Agent"],
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in check_status: {e}", exc_info=True)
        return jsonify({"error": "Failed to check status"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 Not Found: {request.url}")
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 Internal Server Error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting application")
    app.run(debug=True, host="0.0.0.0", port=5000)
