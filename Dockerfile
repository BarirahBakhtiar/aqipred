# Use an official slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all files from your local folder into the container
COPY . /app/

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Tell Docker what command to run when container starts
CMD ["streamlit", "run", "model.py", "--server.port=8501", "--server.address=0.0.0.0"]
