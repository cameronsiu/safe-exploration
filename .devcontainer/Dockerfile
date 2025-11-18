FROM nvcr.io/nvidia/isaac-lab:2.3.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=N \
    DISPLAY=${DISPLAY}

USER root

# Install development utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim \
        git \
        python3-pip \
        sudo \
        bash-completion && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Allow root user passwordless sudo (Isaac Lab uses root by default)
RUN echo "root ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Enable git autocompletion for bash
RUN echo 'if [ -f /usr/share/bash-completion/completions/git ]; then . /usr/share/bash-completion/completions/git; fi' >> ~/.bashrc

# Default shell
SHELL ["/bin/bash", "-c"]

# Final message
CMD ["bash"]