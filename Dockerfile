# Use the Python 3.8.2 image as the base
FROM python:3.8.2-slim-buster

# Update and install necessary packages
RUN apt-get install openjdk-8-jdk -y \
    && apt-get install python3-pip -y \
    && export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/ \

# Set environment variables
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Create a directory for your application
RUN mkdir /app

# Copy your application code into the container
COPY . /app/

# Set the working directory
WORKDIR /app/

# Install the Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Specify the command to run when the container starts
CMD ["python3", "app.py"]