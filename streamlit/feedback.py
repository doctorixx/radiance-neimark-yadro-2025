import requests
import dotenv
import os
import json

from models import Answer, AnswerUpdate

dotenv.load_dotenv("../config.env")


def send_feedback(_answer: Answer):
    response = requests.post(
        "http://192.168.44.157:5000/create",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(os.getenv("FEEDBACK_TOKEN"))
        },
        data=_answer.model_dump_json()
    )
    print("send_feedback")
    return AnswerUpdate.model_validate(response.json())


def change_feedback(_answer_update: AnswerUpdate):
    print(_answer_update)
    response = requests.put(
        f"http://192.168.44.157:5000/answers/{_answer_update.id}",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {os.getenv('FEEDBACK_TOKEN')}"
        },
        data=json.dumps({
            "answer": _answer_update.answer,
            "category": _answer_update.category,
            "score": _answer_update.score,
            "user_query": _answer_update.user_query,
            "version": _answer_update.version,
        })
    )

    print("change_feedback")
    print(response.text)
    print(response.status_code)
    print("change_feedback")
    answer = response.json()
    print(answer)
    return AnswerUpdate.model_validate(answer)


if __name__ == '__main__':
    data = {
        'id': 9,
        'answer': 'Привет! Рад помочь. Если у тебя есть вопросы по Tatlin.Unified, не стесняйся задавать. Я постараюсь предоставить максимально полезную информацию, учитывая твой уровень знаний и раскрыв смежные темы.\n\nЕсли у тебя есть конкретные вопросы или темы, которые тебя интересуют, дай знать, и я постараюсь помочь с максимально полным и исчерпывающим ответом.\n\n---\n\nКонтакты YADRO:\n\n• 123376, г. Москва, ул. Рочдельская 15, стр. 15\n\n• +7 495 540 50 55\n\n• info@yadro.com\n\nТехническая поддержка:\n\n• +7 800 777 06 11\n\n• support@yadro.com',
        'category': 'Эксплуатация',
        'score': 1,
        'user_query': 'Привет',
        'version': 'v1.0'
    }
    change_feedback(AnswerUpdate(**data))
