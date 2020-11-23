FROM tiangolo/uwsgi-nginx-flask:python3.7

# Install base packages
RUN apt-get update --fix-missing && \
    apt-get install -y \
        vim wget bzip2 ca-certificates curl git grep sed dpkg rsync zip unzip
        
# Install application-specific packages
RUN apt-get install -y \
    gfortran libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 \
    libxcomposite1 libasound2 libxi6 libxtst6 unixodbc unixodbc-dev libmpfr-dev \
    libmpc-dev ghostscript poppler-utils

# Copy project files
COPY ./app /app

# Timezone
ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Flask configuration file
ENV APP_CONFIG_FILE=/app/config/env.py


# Miniconda
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $HOME/miniconda && \
    $HOME/miniconda/bin/conda init bash && \
    export CONDA_PIP=$HOME/miniconda/bin/pip
WORKDIR /app

# Python version 3.7.4
RUN $HOME/miniconda/bin/conda install -y python=3.7.4
# PIP
RUN $HOME/miniconda/bin/pip install -r /app/config/requirements.txt
# Conda
RUN $HOME/miniconda/bin/conda install --file /app/config/requirements_conda.txt
RUN $HOME/miniconda/bin/conda install -y -c intel mkl_fft==1.0.14 mkl_random==1.1.0
