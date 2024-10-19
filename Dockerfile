# Use the official Debian 12 base image
FROM debian:12

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-venv \
    python3-pip \
    && apt-get clean

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install required Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY src /app/src

# Set the working directory
WORKDIR /app/src

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
