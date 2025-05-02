# Use official Loki image
FROM grafana/loki:latest

# Copy Loki configuration
COPY ./config/loki-config.yml /etc/loki/local-config.yaml

# Set ownership for loki user
USER root
RUN mkdir -p /tmp/loki && \
    chown -R loki:loki /tmp/loki

# Switch back to loki user
USER loki

# Expose Loki ports
EXPOSE 3100
