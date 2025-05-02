# Terraform configuration for the staging environment

environment = "staging"
aws_region = "us-east-1"

# VPC Configuration
vpc_cidr = "10.1.0.0/16"  # Separate CIDR range from dev/prod
availability_zones = ["us-east-1a", "us-east-1b"]

# Database Configuration
db_instance_type = "db.t3.medium"
db_username = "forex_staging_user"
# Note: Password should be injected through environment variables or secrets management
# db_password = Managed through GitHub secrets

# Compute Configuration
compute_instance_type = "t3.medium"

# Scaling Configuration
min_capacity = 2
max_capacity = 4
desired_capacity = 2

# Feature Flags
enable_monitoring = true
enable_alerting = true
enable_backups = true

# Seeding Configuration
enable_test_data_seeding = true
test_data_seed_script = "s3://forex-platform-assets/seeds/staging-seed-data.sql"

# TTL for resources (for cost optimization)
resource_ttl_days = 7  # Auto-terminate after 7 days of inactivity
