# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /code

COPY ./Requirements.txt /code/Requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/Requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]