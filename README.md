> Radiance team

# Работа с кодом

- Запрет на push в main (Security)
- CI для тестов кода (линтеры)
![img.png](images/1122.png)
- Задачи организованы в Github Project (По методологии kanban)
![img.png](images/img.png)


# Запуск

## Grafana + Prometheus
- перейти в feedback service
- запустить
```shell
docker compose up --build
```

## Database

- перейти в feedback_service
- запустить
```shell
docker compose up --build
```
## Feedback microservice

- перейти в feedback service
- установить бибилиотеки
```shell
pip instal -r requirements
```
- запустить
```shell
python3 app.py
```
## Frontendback service

...