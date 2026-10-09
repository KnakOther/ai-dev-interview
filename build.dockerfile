FROM python:3.12.7-bullseye

RUN pip install poetry==2.5.1
RUN apt-get -y update
RUN apt-get install -y python3-full

COPY --from=timoreymann/deterministic-zip:4.0.1 /bin/deterministic-zip /usr/bin/deterministic-zip

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

RUN poetry install --only main --no-root
RUN mkdir -p dist && mkdir -p dist/_python_dependencies && mkdir /asset && mkdir /asset-output
RUN cp --recursive .venv/lib/python*/site-packages/* dist/_python_dependencies
RUN cd dist/_python_dependencies && deterministic-zip -r ../python_dependencies.zip .
WORKDIR /app
RUN rm -rf dist/_python_dependencies

COPY src ./dist/src
COPY lambda_handler.py dist/

RUN cd dist && deterministic-zip -r ../dist.zip . && cd ..
