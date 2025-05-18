# syntax=docker/dockerfile:1

# Select base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy necessary files to container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy remaining files
COPY . .

# Expose Flask port
EXPOSE 5000

# Define commands to run application
CMD ["python3", "app/service_api.py"]
