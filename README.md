

> [!NOTE]  
> Решние было создано в рамках хакатона НЕЙМАРК x YADRO (2025)

## 🧑‍💻 Команда Radiance

| Имя             | Роль                  |
| --------------- | --------------------- |
| Сергей Катцын   | ML Engineer, Frontend |
| Полина Никулина | Дизайнер, Аналитик    |
| Ярослав Суслов  | DevOps                |
| Даниил Мошкин   | Backend-разработчик   |
| Артём Мухин     | Backend-разработчик   |

---

# ✨ AI-помощник для технической поддержки СХД
![img_1.png](images/img_1.png)
Интеллектуальное решение на базе искусственного интеллекта, разработанное командой **Radiance team** в рамках кейса YADRO. Система автоматизирует обработку заявок, отвечает на вопросы о СХД и снижает нагрузку на инженеров, экономя время и бюджет компании.

> 🤖 *От рутинных задач — к умной поддержке 24/7*

---

## 🔍 Зачем это нужно?

* 📉 **Сокращение нагрузки** на технических специалистов до 70%
* 🚀 **Ускорение реакции** на типовые инциденты на 30–50%
* 💸 **Снижение затрат** на поддержку клиентов до 50%
* 🌙 **Круглосуточная работа** без участия сотрудников

---

## 🧠 Архитектура
![img_2.png](images/img_2.png)
* LangFlow (AI Flow-based агент, RAG, классификация)
* Feedback microservice
* База данных + сбор аналитики (PostgreSQL + Prometheus + Grafana)
* Frontend (Streamlit)
* CI-пайплайн и организация задач по Kanban (GitHub Projects)

---

## 🚀 Быстрый старт

> 🔒 **Внимание**: защита от прямых пушей в `main`, все изменения через pull requests

### 1. 📊 Monitoring: Grafana + Prometheus

* Перейти в директорию `feedback_service`
* Запустить:

  ```bash
  docker compose up --build
  ```
* Подключить базу данных PostgreSQL со следующими данными:

  ```
  login: postgres
  password: admin
  host: host.docker.internal:25565
  TLS/SSL Mode: disabled
  ```
* Импортировать дашборд из:
  `_monitoring/grafana_dashboard_export.json`

---

### 2. 🛢️ Database

В директории `db`:

```bash
docker compose up --build
```

---

### 3. 🗣️ Feedback microservice

* Перейти в `feedback_service`
* Установить зависимости:

  ```bash
  pip install -r requirements.txt
  ```
* Запустить:

  ```bash
  python3 app.py
  ```

---

### 4. 🎨 Frontend (Streamlit)

* Перейти в директорию `streamlit`
* Создать и активировать виртуальное окружение `venv`
* Установить зависимости:

  ```bash
  pip install -r requirements.txt
  ```
* Создать файл `.env` с конфигурацией:

  ```
  LANGFLOW_API_KEY=<<КЛЮЧ API LANGFLOW>>
  YADRO_FLOW=<<Имя flow для RAG>>
  CATEGORY_FLOW=<<Имя flow для Category>>
  FEEDBACK_TOKEN=default-secret-token
  ```
* Запустить:

  ```bash
  streamlit run main.py
  ```

---

### 5. 🤖 LangFlow

#### RAG Agent:

* Импортировать `/langflow/Vector Store RAG.json`
* Добавить API-токены MistralAI
* Загрузить файлы из `/langflow/files` в модуль File
* Запустить ChromaDB

#### Category Classifier:

* Импортировать `/langflow/Category.json`
* Указать токены MistralAI

---

## 📁 Работа с кодом

* 🛡️ Защита: запрет на push в `main`
* ✅ CI: автоматические проверки и линтеры
* 📌 Задачи организованы по **Kanban** в GitHub Projects
  ![](images/1122.png)
  ![](images/img.png)

---


## 📈 Влияние на бизнес

* 🕒 Сокращение AHT на **30–50%**
* 🤖 Обработка до **70%** рутинных запросов автоматически
* 🌐 Круглосуточная доступность
* 💰 Снижение затрат на поддержку на **до 50%**

---

## 📌 Roadmap

| Стадия    | Задачи                                |
| --------- | ------------------------------------- |
| 📍 Этап 1 | Поиск проблемы, анализ, выделение ЦА  |
| ⚙️ Этап 2 | Проверка инженерами, добавление в RAG |
| 🚀 Этап 3 | Релиз, мониторинг, сбор фидбека       |

---

Готов объединить это в `README.md` файл, если нужно — могу экспортировать или оформить под конкретный стиль проекта.

Хочешь добавить бейджики GitHub Actions, лицензии или кнопку запуска?
