# Use multi-stage build for optimal image size
FROM grafana/grafana-oss:latest

# Copy dashboards
COPY ./dashboards /etc/grafana/provisioning/dashboards

# Copy datasource configuration
COPY ./config/grafana-datasources.yaml /etc/grafana/provisioning/datasources/

# Set environment variables
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=forex123
ENV GF_USERS_ALLOW_SIGN_UP=false

# Expose Grafana port
EXPOSE 3000
