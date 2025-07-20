from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    category: str
    score: int
    user_query: str
    version: str

class AnswerUpdate(BaseModel):
    id: int
    answer: str
    category: str
    score: int
    user_query: str
    version: str