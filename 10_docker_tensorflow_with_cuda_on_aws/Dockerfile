# VERSION 1.0.0
# BUILD: docker build --rm -t cuda_app .

FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Install basic libraries
RUN apt-get update -y --fix-missing
RUN apt-get install -y software-properties-common apt-utils locales tzdata

# Install required packages
RUN apt-get install -y python3-pip python3-setuptools python3-tk \
                        libpq-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev \
                        gcc && \
    apt -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# App Home directory
ARG APP_HOME=/home/ubuntu/app

# Install Python modules
ADD requirements.txt ${APP_HOME}/
RUN pip3 install -r ${APP_HOME}/requirements.txt
# RUN pip3 install --upgrade awscli

# Bundle app source
ADD /src ${APP_HOME}/src

# Add entrypoint shell file that determines which job to run
COPY entrypoint.sh /entrypoint.sh
RUN ["chmod", "+x", "/entrypoint.sh"]

# Define entrypoint
WORKDIR ${APP_HOME}/src
ENV PYTHONPATH ${APP_HOME}/src
# ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "$APP_HOME/src/run.py"]
