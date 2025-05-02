variable "environment" {
  description = "Deployment environment (dev, test, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region where resources will be deployed"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "db_instance_type" {
  description = "Instance type for the database"
  type        = string
  default     = "db.t3.medium"
}

variable "db_username" {
  description = "Username for database access"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Password for database access"
  type        = string
  sensitive   = true
}

variable "compute_instance_type" {
  description = "Instance type for compute resources"
  type        = string
  default     = "t3.medium"
}
