import streamlit as st
import requests
import os
import dotenv
import uuid

from feedback import change_feedback, send_feedback
from models import AnswerUpdate, Answer
from categorize import categorize

dotenv.load_dotenv("../config.env")


def run_langflow_api(input_message: str, session_id: str, output_type: str = "chat", input_type: str = "chat") -> dict:
    try:
        api_key = os.environ["LANGFLOW_API_KEY"]
        flow = os.environ["YADRO_FLOW"]
    except KeyError:
        raise EnvironmentError(
            "LANGFLOW_API_KEY environment variable not found. "
            "Please set your API key in the environment variables."
        )

    url = f"http://localhost:7860/api/v1/run/{flow}"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "output_type": output_type,
        "input_type": input_type,
        "input_value": input_message,
        "session_id": session_id
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"API request failed: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Invalid JSON response: {str(e)}") from e


def extract_response_text(api_response: dict) -> str:
    """
    Extract text from LangFlow API response
    """
    try:
        return api_response['outputs'][0]['outputs'][0]['results']['message']['text'].strip()
    except (KeyError, IndexError, TypeError):
        try:
            return api_response['outputs'][0]['outputs'][0]['outputs']['message']['message'].strip()
        except (KeyError, IndexError, TypeError):
            def find_text(data):
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k == 'text' and isinstance(v, str):
                            return v
                        result = find_text(v)
                        if result:
                            return result
                elif isinstance(data, list):
                    for item in data:
                        result = find_text(item)
                        if result:
                            return result
                return None

            text = find_text(api_response)
            return text.strip() if text else "Answer not found"


# Callback-функции для обработки оценок
def set_rating(message_key, rating, _answer: str, _question: str, _category: str) -> None:
    st.session_state.ratings[message_key] = rating
    answer = _answer
    score = 1 if rating == "like" else 0
    user_query = _question
    version = "v1.0"
    category = _category

    answer_id = st.session_state.feedbacks.get(message_key)
    print(answer_id)

    print({
        "id": answer_id,
        "answer": answer,
        "category": category,
        "score": score,
        "user_query": user_query,
        "version": version,
    })
    if answer_id:
        params = AnswerUpdate.model_validate(
            {
                "id": answer_id,
                "answer": answer,
                "category": category,
                "score": score,
                "user_query": user_query,
                "version": version,
            }
        )
        ans_id = change_feedback(params)
    else:
        params = Answer.model_validate(
            {
                "answer": answer,
                "category": category,
                "score": score,
                "user_query": user_query,
                "version": version,
            }
        )
        ans_id = send_feedback(params)
        st.session_state.feedbacks[message_key] = ans_id.id


st.title("Помощник для Tatlin.Unified")

# Инициализация состояния сессии
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = {}

if "categories" not in st.session_state:
    st.session_state.categories = {}

# Кнопка для сброса сессии
if st.button("Сбросить чат"):
    st.session_state.clear()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.ratings = {}
    st.session_state.feedbacks = {}
    st.session_state.categories = {}
    st.experimental_rerun()

# Контейнер для истории чата
chat_container = st.container()


# Отображение истории сообщений
def display_messages():
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            if role == "AI":
                message_key = f"{st.session_state.session_id}_{i}"
                current_rating = st.session_state.ratings.get(message_key, None)

                if current_rating == "like":
                    st.caption("👍 Вы оценили этот ответ положительно")
                elif current_rating == "dislike":
                    st.caption("👎 Вы оценили этот ответ отрицательно")

                _message = st.session_state.messages[i - 1]
                _content = _message["content"]
                _category = st.session_state.categories.get(message_key)
                # Кнопки для оценки
                col1, col2 = st.columns([1, 10])
                with col1:
                    if st.button("👍", key=f"like_{message_key}"):
                        set_rating(message_key, "like", content, _content, _category)
                with col2:
                    if st.button("👎", key=f"dislike_{message_key}"):
                        set_rating(message_key, "dislike", content, _content, _category)


# Первоначальное отображение сообщений
with chat_container:
    display_messages()

# Обработка ввода пользователя
if prompt := st.chat_input("Ваше сообщение"):
    # Добавление сообщения пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Обновляем контейнер с новым сообщением
    with chat_container:
        # Отображаем только новое сообщение пользователя
        with st.chat_message("user"):
            st.markdown(prompt)

    # Генерация ответа AI
    try:
        response = run_langflow_api(prompt, st.session_state.session_id)
        text = extract_response_text(response)
        # Добавление ответа AI в историю
        st.session_state.messages.append({"role": "AI", "content": text})
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.messages.append({"role": "AI", "content": error_msg})

    category = categorize(prompt)

    # Обновляем контейнер с ответом AI
    with chat_container:
        # Отображаем только последнее сообщение AI
        last_message = st.session_state.messages[-1]

        message_key = f"{st.session_state.session_id}_{len(st.session_state.messages) - 1}"

        st.session_state.categories[message_key] = category
        with st.chat_message(last_message["role"]):
            st.markdown(last_message["content"])

            if last_message["role"] == "AI":
                current_rating = st.session_state.ratings.get(message_key, None)

                if current_rating == "like":
                    st.caption("👍 Вы оценили этот ответ положительно")
                elif current_rating == "dislike":
                    st.caption("👎 Вы оценили этот ответ отрицательно")

                _message = st.session_state.messages[-2]
                _content = _message["content"]

                # Кнопки для оценки
                col1, col2 = st.columns([1, 10])
                with col1:
                    if st.button("👍", key=f"like_{message_key}"):
                        set_rating(message_key, "like", last_message["content"], _content, category)
                with col2:
                    if st.button("👎", key=f"dislike_{message_key}"):
                        set_rating(message_key, "dislike", last_message["content"], _content, category)
