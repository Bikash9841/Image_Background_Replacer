#this sets up the container with Python 3.10 installed.
FROM python:3.10-slim

# this copies everything in your current directory
COPY . /app

# Thid sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT or COPY instructions that follow
WORKDIR /app

# this runs pip install for all the packages listed in your requirements.txt file
RUN pip install -r requirements.txt

# This tells Docker to listen on port 8501 at runtime. Port 8501 is the streamlit's default port.
EXPOSE 8501

# this tells Docker how to test a container to check that it is still working.

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# this allows to configure a container that will run as an executable. it containes entire streamlit run command for the app

ENTRYPOINT [ "streamlit","run","main.py","--server.port=8501" ]
