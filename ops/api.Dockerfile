# Set base image (host OS)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./ops/requirements.api.txt ./requirements.txt

# Install any dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD [ "uvicorn", "app:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000" ]