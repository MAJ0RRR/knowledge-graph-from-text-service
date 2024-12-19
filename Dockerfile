FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN set -x && \
    apt-get update && \
    apt-get install --no-install-recommends --assume-yes \
      build-essential \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create user
ARG UID=1000
ARG GID=1000
RUN set -x && \
    groupadd -g "${GID}" python && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python &&\
    chown python:python -R /app

USER python

# Install python dependencies within the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download NLTK data to a directory with write permissions
RUN python -m nltk.downloader -d /home/python/nltk_data punkt_tab
RUN python -m nltk.downloader -d /home/python/nltk_data averaged_perceptron_tagger_eng

# Copy the entire project
COPY . .

# don't buffer Python output
ENV PYTHONUNBUFFERED=1
# Add pip's user base to PATH
ENV PATH="$PATH:/home/python/.local/bin"
# Add NLTK data to NLTK_DATA environment variable
ENV NLTK_DATA="/home/python/nltk_data"

# expose port
EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]