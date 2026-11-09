# Sandbox image for agent code. Changing this is an experimental change: record it.
FROM python:3.14-slim

# Fixed scientific stack. Whatever is listed here must also be listed in the
# system prompt so agents know what is available.
RUN pip install --no-cache-dir numpy scipy

# Unprivileged user; agents never need root.
RUN useradd --create-home --shell /bin/bash agent
USER agent
WORKDIR /work
