FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim \
    python3 \
    python3-pip \
    python3-tk \
    tzdata \
    libx11-dev
ENV DISPLAY="host.docker.internal:0.0"

RUN pip3 install numpy matplotlib cvxpy cvxopt MOSEK
ENV MOSEKLM_LICENSE_FILE="/usr/local/lib/python3.8/dist-packages/mosek"
COPY mosek.lic /usr/local/lib/python3.8/dist-packages/mosek

CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]
