# Getting base image ubuntu
FROM python:3.6

MAINTAINER Yong Joon Thoo <yongjoon.thoo@gmail.com>

CMD ["echo", "---- Container is created from image ----"]

# Set working directory
WORKDIR src/

# Install required libraries
COPY ./requirements.txt /src
RUN pip install -r requirements.txt

COPY ./MovieGenreClassifier /src/MovieGenreClassifier 
COPY ./MovieGenreClassifier/MovieGenreClassifier.py /src

# Container will run as an executable
ENTRYPOINT ["python", "MovieGenreClassifier.py"]