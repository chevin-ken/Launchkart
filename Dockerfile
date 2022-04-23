################################################################
FROM ubuntu:xenial-20201030 AS base


# Setup environment variables in a single layer
ENV \
    # Prevent dpkg from prompting for user input during package setup
    DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    # mupen64plus will be installed in /usr/games; add to the $PATH
    PATH=$PATH:/usr/games \
    # Set default DISPLAY
    DISPLAY=:0


################################################################
FROM base AS buildstuff

RUN apt-get update && \
    apt-get install -y \
        build-essential dpkg-dev libwebkitgtk-dev libjpeg-dev libtiff-dev libgtk2.0-dev \
        libsdl1.2-dev libgstreamer-plugins-base0.10-dev libnotify-dev freeglut3 freeglut3-dev \
        libjson-c2 libjson-c-dev \
        git

# clone, build, and install the input bot
# (explicitly specifying commit hash to attempt to guarantee behavior within this container)
WORKDIR /src/mupen64plus-src
RUN git clone https://github.com/chevin-ken/mupen64plus-core && \
        cd mupen64plus-core && \
        git reset --hard 12d136dd9a54e8b895026a104db7c076609d11ff && \
    cd .. && \
    git clone https://github.com/chevin-ken/mupen64plus-input-bot && \
        cd mupen64plus-input-bot && \
        git reset --hard 0a1432035e2884576671ef9777a2047dc6c717a2 && \
    make all && \
    make install

################################################################
FROM base

# Update package cache and install dependencies
RUN apt-get update && \
    apt-get install -y \
        libjson-c2 \
        wget \
        xvfb libxv1 x11vnc \
        imagemagick \
        mupen64plus \
        nano \
        ffmpeg \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        liblzma-dev \
        lzma \
        rsync \
        git

# Install Python 3.7.9
RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz && \
    tar -xvf Python-3.7.9.tgz && \
    cd Python-3.7.9 && \
    ./configure --enable-optimizations && \
    make && \
    make install && \
    cd .. && \
    rm -rf Python*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install VirtualGL (provides vglrun to allow us to run the emulator in XVFB)
# (Check for new releases here: https://github.com/VirtualGL/virtualgl/releases)
ENV VIRTUALGL_VERSION=2.5.2
RUN wget "https://sourceforge.net/projects/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb" && \
    apt install ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb && \
    rm virtualgl_${VIRTUALGL_VERSION}_amd64.deb

# Copy compiled input plugin from buildstuff layer
COPY --from=buildstuff /usr/local/lib/mupen64plus/mupen64plus-input-bot.so /usr/local/lib/mupen64plus/

RUN pip3 install torch \
    tensorboard \
    ray[rllib]

VOLUME /src/launchkart/

WORKDIR /src
RUN git clone https://github.com/chevin-ken/gym-mupen64plus

#Skip the cache for the below command to make sure we git pull if there are updates to the environment
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Install requirements & this package
RUN cd /src/gym-mupen64plus && \
    git pull && \
    pip3 install -e . && \
    pip3 install inputs

COPY ./ROMs/marioKart.n64 /src/gym-mupen64plus/gym_mupen64plus/ROMs

WORKDIR /src

# Expose the default VNC port for connecting with a client/viewer outside the container
EXPOSE 5900