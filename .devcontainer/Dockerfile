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

# Set environment to run VS Code Remote container properly

# If you have additional dev-dependencies (vim, python, ros, etc), install here
# For example:
# RUN sudo apt-get update && sudo apt-get install -y python3-pip

# (Optional) Add VS Code remote-container configuration, e.g. setting up shell or dotfiles