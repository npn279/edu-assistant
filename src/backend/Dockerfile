FROM python:3.11

# Common dependencies
RUN apt-get update -qqy && \
    apt-get install -y --no-install-recommends \
      ssh \
      git \
      gcc \
      g++ \
      poppler-utils \
      libpoppler-dev \
      unzip \
      curl \
      cargo

# Keeps Python from generating .pyc files in the container
# Turns off buffering for easier container logging
# Force UTF8 encoding for funky character handling
# Needed so imports function properly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONPATH=/workspace/src/
# Keep the venv name and location predictable
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# "Activate" the venv manually for the context of the container
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

# Need to expose port in ENV to use in CMD
ARG PORT=8000
ENV PORT=${PORT}

# Build with community packages
ARG INSTALL_COMMUNITY_DEPS

# Copy dependency files to avoid cache invalidations
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip3 install --no-cache-dir poetry==1.6.1
RUN pip3 install graphrag

# Conditional installation of dependencies
RUN if [ "$INSTALL_COMMUNITY_DEPS" = "true" ]; then \
      poetry install --with dev,community; \
    else \
      poetry install --with dev; \
    fi

COPY src/backend/ src/backend/
COPY src/community src/community/

# Install graphrag using pip after Poetry installs
RUN $VIRTUAL_ENV/bin/pip install graphrag

# Copy environment variables optionally 
# IMPORTANT: Can't be put in the docker-compose, will break tests
COPY .en[v] .env

EXPOSE ${PORT}

CMD uvicorn backend.main:app --reload --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 300