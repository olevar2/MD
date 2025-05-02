terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"

  backend "s3" {
    # These values would be provided during initialization
    # terraform init -backend-config="bucket=your-state-bucket" -backend-config="key=forex/terraform.tfstate" ...
    # bucket = "forex-platform-terraform-state"
    # key    = "forex/terraform.tfstate"
    # region = "us-east-1"
    # dynamodb_table = "forex-terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Forex Trading Platform"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Remote state data sources for shared infrastructure
# data "terraform_remote_state" "shared" {
#   backend = "s3"
#   config = {
#     bucket = "forex-platform-terraform-state"
#     key    = "shared/terraform.tfstate"
#     region = var.aws_region
#   }
# }

# Networking module for VPC and subnets
module "networking" {
  source = "./modules/networking"
  
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
}

# Database module for TimescaleDB
module "database" {
  source = "./modules/database"
  
  environment     = var.environment
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
  instance_type   = var.db_instance_type
  db_name         = "forex_${var.environment}"
  db_username     = var.db_username
  db_password     = var.db_password
}

# Compute module for ECS/Kubernetes
module "compute" {
  source = "./modules/compute"
  
  environment     = var.environment
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
  instance_type   = var.compute_instance_type
}

# Security module for IAM and Security Groups
module "security" {
  source = "./modules/security"
  
  environment     = var.environment
  vpc_id          = module.networking.vpc_id
}

# Monitoring module for CloudWatch/Prometheus/Grafana
module "monitoring" {
  source = "./modules/monitoring"
  
  environment     = var.environment
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
}
