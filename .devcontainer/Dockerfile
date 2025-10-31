FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=N \
    DISPLAY=${DISPLAY}

USER root
RUN apt-get update && apt-get install -y vim git python3-pip sudo && rm -rf /var/lib/apt/lists/*

# Allow isaac-sim user to use sudo without password (optional)
RUN echo "isaac-sim ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER isaac-sim
WORKDIR /isaac-sim

# Adding git autocompletion
RUN echo 'if [ -f /usr/share/bash-completion/completions/git ]; then . /usr/share/bash-completion/completions/git; fi' >> /isaac-sim/.bashrc