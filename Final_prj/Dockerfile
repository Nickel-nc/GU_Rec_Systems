FROM python:3.7-stretch

MAINTAINER nickelnc <gm.nickel.nc@gmail.com>

# set a directory for the app
WORKDIR /usr/app

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# app run
CMD ["python", "Run_1lvl.py"]
