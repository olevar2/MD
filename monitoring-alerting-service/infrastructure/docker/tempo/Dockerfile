# Use official Tempo image
FROM grafana/tempo:latest

# Copy Tempo configuration
COPY ./config/tempo-config.yml /etc/tempo/config.yml

# Create directories for tempo data
USER root
RUN mkdir -p /tmp/tempo/blocks && \
    mkdir -p /tmp/tempo/generator/wal && \
    chown -R tempo:tempo /tmp/tempo

# Switch back to tempo user
USER tempo

# Expose Tempo ports
EXPOSE 3200 14250 14268 55681 4317
