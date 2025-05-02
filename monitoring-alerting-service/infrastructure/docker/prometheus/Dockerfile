# Use official Prometheus image
FROM prom/prometheus:latest

# Copy prometheus configuration
COPY ./config/prometheus.yml /etc/prometheus/
COPY ./alerts/*.yml /etc/prometheus/alerts/

# Set ownership for prometheus user
USER root
RUN chown -R nobody:nobody /etc/prometheus

# Switch back to prometheus user
USER nobody

# Expose Prometheus port
EXPOSE 9090
