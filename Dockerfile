# forex-predicter/Dockerfile

# Use the official Debian 12 base image
FROM debian:12

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies and dependencies for TA-Lib
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-venv \
    python3-pip \
    tar \
    && apt-get clean

# Install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install required Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY src /app/src

# Set the working directory to /app
WORKDIR /app

# Set PYTHONPATH to include /app to recognize src as a module
ENV PYTHONPATH="/app"

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
