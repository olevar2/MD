# Terraform configuration for the production environment

environment = "production"
aws_region = "us-east-1"  # Primary region

# Multi-region support (for disaster recovery)
secondary_region = "us-west-2"
enable_multi_region = true

# VPC Configuration
vpc_cidr = "10.0.0.0/16"  # Production CIDR range
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]  # Use all AZs for high availability

# Database Configuration
db_instance_type = "db.m5.large"  # More powerful instance for production
db_username = "forex_prod_user"
# Note: Password managed through secrets
enable_db_encryption = true
enable_db_backup = true
backup_retention_period = 30  # Days
multi_az = true  # Enable Multi-AZ deployment for RDS

# Compute Configuration
compute_instance_type = "m5.large"  # More powerful instance for production

# Scaling Configuration
min_capacity = 3  # Minimum instances for high availability
max_capacity = 10
desired_capacity = 3

# Feature Flags
enable_monitoring = true
enable_alerting = true
enable_backups = true
enable_waf = true  # Web Application Firewall
enable_cloudfront = true  # CDN for global content delivery

# Security Configuration
enable_vpc_flow_logs = true
enable_cloudtrail = true
ssl_certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/example-cert"

# Performance Configuration
enable_elasticache = true  # Redis caching
enable_read_replicas = true  # Database read replicas
read_replica_count = 2
