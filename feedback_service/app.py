from flask import Flask, request, jsonify, redirect
from functools import wraps
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from flasgger import Swagger

app = Flask(__name__)

# Конфигурация Swagger
app.config['SWAGGER'] = {
    'title': 'Answers API',
    'uiversion': 3,
    'version': '1.0',
    'description': 'API для управления ответами в базе данных (psycopg2 версия)',
    'termsOfService': '',
    'tags': [
        {
            'name': 'answers',
            'description': 'Операции с ответами'
        }
    ]
}
# swagger = Swagger(app)

# Конфигурация базы данных
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'metrics'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'admin'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '25565')
}


def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


# Токен для аутентификации
API_TOKEN = os.getenv('API_TOKEN', 'default-secret-token')


# Декоратор для проверки токена
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        # if not token or token != f'Bearer {API_TOKEN}':
        #     return jsonify({'message': 'Unauthorized'}), 401

        return f(*args, **kwargs)

    return decorated


# Отключаем CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


# Ручка для создания ответа
@app.route('/create', methods=['POST'])
# @token_required
def create_answer():
    """
    Создание нового ответа
    ---
    tags:
      - answers
    security:
      - Bearer: []
    parameters:
      - in: header
        name: Authorization
        required: true
        schema:
          type: string
        description: Bearer token для аутентификации
      - in: body
        name: body
        required: true
        schema:
          id: Answer
          required:
            - user_query
            - answer
            - score
            - version
            - category
          properties:
            user_query:
              type: string
              description: Запрос пользователя
            answer:
              type: string
              description: Ответ на запрос
            score:
              type: integer
              description: Оценка ответа (1-10)
            version:
              type: string
              description: Версия ответа
            category:
              type: string
              description: Категория ответа
    responses:
      201:
        description: Ответ успешно создан
        schema:
          $ref: '#/definitions/Answer'
      400:
        description: Неверные входные данные
      401:
        description: Не авторизован
      500:
        description: Ошибка сервера
    """
    data = request.get_json()

    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    required_fields = ['user_query', 'answer', 'score', 'version', 'category']
    for field in required_fields:
        if field not in data:
            return jsonify({'message': f'Missing required field: {field}'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = sql.SQL("""
            INSERT INTO answers (user_query, answer, score, version, category)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, user_query, answer, score, version, category
        """)

        cursor.execute(query, (
            data['user_query'],
            data['answer'],
            data['score'],
            data['version'],
            data['category']
        ))

        conn.commit()
        new_answer = cursor.fetchone()

        # cursor.close()
        # conn.close()

        return jsonify(new_answer), 201

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return jsonify({'message': str(e)}), 500


# Ручка для получения всех ответов
@app.route('/answers', methods=['GET'])
def get_answers():
    """
    Получение списка всех ответов
    ---
    tags:
      - answers
    responses:
      200:
        description: Список ответов
        schema:
          type: array
          items:
            $ref: '#/definitions/Answer'
      500:
        description: Ошибка сервера
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("SELECT * FROM answers")
        answers = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(answers), 200

    except Exception as e:
        if 'conn' in locals():
            cursor.close()
            conn.close()
        return jsonify({'message': str(e)}), 500


# Добавьте этот endpoint в ваш Flask-приложение
@app.route('/answers/<int:answer_id>', methods=['PUT'])
# @token_required
def update_answer(answer_id):
    """
    Обновление существующего ответа
    ---
    tags:
      - answers
    security:
      - Bearer: []
    parameters:
      - in: path
        name: answer_id
        required: true
        type: integer
        description: ID ответа для обновления
      - in: header
        name: Authorization
        required: true
        schema:
          type: string
        description: Bearer token для аутентификации
      - in: body
        name: body
        required: true
        schema:
          id: AnswerUpdate
          properties:
            user_query:
              type: string
              description: Новый запрос пользователя
            answer:
              type: string
              description: Новый ответ
            score:
              type: integer
              description: Новая оценка
            version:
              type: string
              description: Новая версия
            category:
              type: string
              description: Новая категория
    responses:
      200:
        description: Ответ успешно обновлен
        schema:
          $ref: '#/definitions/Answer'
      400:
        description: Неверные входные данные
      404:
        description: Ответ не найден
      500:
        description: Ошибка сервера
    """
    data = request.get_json()

    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    # Проверяем, что есть хотя бы одно поле для обновления
    updatable_fields = ['user_query', 'answer', 'score', 'version', 'category']
    if not any(field in data for field in updatable_fields):
        return jsonify({'message': 'No fields to update provided'}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Собираем динамический запрос на основе переданных полей
        set_parts = []
        params = []

        for field in updatable_fields:
            if field in data:
                set_parts.append(sql.SQL("{} = %s").format(sql.Identifier(field)))
                params.append(data[field])

        params.append(answer_id)

        query = sql.SQL("""
            UPDATE answers
            SET {set_clause}
            WHERE id = %s
            RETURNING id, user_query, answer, score, version, category
        """).format(
            set_clause=sql.SQL(', ').join(set_parts)
        )

        cursor.execute(query, params)
        updated_answer = cursor.fetchone()

        if not updated_answer:
            cursor.close()
            conn.close()
            return jsonify({'message': 'Answer not found'}), 404

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify(updated_answer), 200

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return jsonify({'message': str(e)}), 500

# Ручка для проверки работы сервиса
@app.route('/health', methods=['GET'])
def health_check():
    """
    Проверка работоспособности сервиса
    ---
    tags:
      - health
    responses:
      200:
        description: Сервис работает
        schema:
          type: object
          properties:
            status:
              type: string
            db_status:
              type: string
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return jsonify({'status': 'ok', 'db_status': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'ok', 'db_status': str(e)}), 500


# Определение схемы для Swagger
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Answers API (psycopg2)",
        "description": "API для управления ответами в базе данных с использованием psycopg2",
        "version": "1.0"
    },
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Авторизация с использованием Bearer token. Пример: 'Bearer {token}'"
        }
    },
    "definitions": {
        "Answer": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer"
                },
                "user_query": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                },
                "score": {
                    "type": "integer"
                },
                "version": {
                    "type": "string"
                },
                "category": {
                    "type": "string"
                }
            }
        },
"AnswerUpdate": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                },
                "score": {
                    "type": "integer"
                },
                "version": {
                    "type": "string"
                },
                "category": {
                    "type": "string"
                }
            }
        }
    }
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/"
}

@app.route("/")
def index():
    return redirect("/swagger")

swagger = Swagger(app, template=swagger_template, config=swagger_config)


def init_db():
    """Создание таблицы если её нет"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id SERIAL PRIMARY KEY,
                user_query TEXT,
                answer TEXT,
                score INTEGER,
                version TEXT,
                category TEXT
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("Таблица 'answers' создана или уже существует")
    except Exception as e:
        print(f"Ошибка при создании таблицы: {str(e)}")
        raise

init_db()
if __name__ == '__main__':
    app.run(debug=os.getenv('DEBUG', False), host="0.0.0.0")