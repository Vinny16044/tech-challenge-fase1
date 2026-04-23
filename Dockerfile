FROM python:3.10-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry
RUN pip install poetry && \
    poetry config virtualenvs.create false

# Copia arquivos de dependência primeiro (cache de build)
COPY pyproject.toml poetry.lock* ./

# Instala dependências (incluindo dev para Jupyter)
RUN poetry install --no-interaction --no-ansi

# Copia o projeto
COPY . .

# Expõe a porta do Jupyter
EXPOSE 8888

# Comando para iniciar o Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
