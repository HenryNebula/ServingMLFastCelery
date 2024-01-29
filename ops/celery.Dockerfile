# Set base image (host OS)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ./ops/requirements.celery.txt ./requirements.txt

# Install any dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && apt-get -y install curl libgomp1

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD [ "celery", "-A", "celery_task_app.worker", "worker", "--concurrency", "2", "--prefetch-multiplier", "1", "-l", "info" ]