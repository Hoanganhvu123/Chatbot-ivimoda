# Dockerfile

# pull the official docker image
FROM python:3.11.3-slim

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OPENAI_API_KEY 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

# install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install ez_setup
RUN pip install -r requirements.txt

# copy project
COPY . .