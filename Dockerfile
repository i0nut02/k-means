FROM ubuntu:latest

# Install necessary dependencies, including MPI and OpenMP
RUN apt update && apt install -y \
    build-essential \
    mpich \
    libopenmpi-dev \
    gcc \
    make \
    gfortran \
    && apt clean

# Install OpenMP if needed (optional based on your Dockerfile)
RUN apt install -y libomp-dev

WORKDIR /home/ubuntu/kmeans

COPY . /home/ubuntu/kmeans