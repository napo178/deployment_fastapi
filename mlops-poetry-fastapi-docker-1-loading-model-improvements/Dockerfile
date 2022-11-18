###############################################
# Base Image
###############################################

FROM python:3.8-slim as python-base
LABEL maintainer="Mihlos Code"

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PIP_NO_CACHE_DIR=off 
ENV PIP_DISABLE_PIP_VERSION_CHECK=on 
ENV PIP_DEFAULT_TIMEOUT=100 
ENV POETRY_VERSION=1.1.7 
ENV POETRY_HOME="/opt/poetry" 
ENV POETRY_VIRTUALENVS_IN_PROJECT=true 
ENV POETRY_NO_INTERACTION=1 
ENV PYSETUP_PATH="/opt/pysetup" 
ENV VENV_PATH="/opt/pysetup/.venv"

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

###############################################
# Builder Image
###############################################

FROM python-base as builder-base
RUN apt-get update && apt-get install --no-install-recommends -y curl build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --no-dev

###############################################
# Production Image
###############################################

FROM python-base as production
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
COPY ./mlops /mlops/

CMD ["uvicorn", "mlops.app:app", "--host", "0.0.0.0", "--port", "80"]