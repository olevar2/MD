# Installation Guide

This guide provides detailed instructions for installing and setting up the Forex Trading Platform.

## Prerequisites

Before installing the platform, ensure you have the following prerequisites:

- **Python 3.10+**: Required for running all services
- **PostgreSQL 14+ with TimescaleDB**: Required for storing time-series data
- **Docker and Docker Compose** (optional): For containerized deployment
- **Redis** (optional): For caching and pub/sub messaging
- **Kafka** (optional): For event streaming

### Installing Prerequisites

#### Python

Install Python 3.10 or later:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
brew install python
```

**Windows:**
Download and install from [python.org](https://www.python.org/downloads/)

#### PostgreSQL with TimescaleDB

**Ubuntu/Debian:**
```bash
# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update

# Install PostgreSQL and TimescaleDB
sudo apt install postgresql-14 timescaledb-2-postgresql-14

# Enable TimescaleDB
sudo sh -c "echo \"shared_preload_libraries = 'timescaledb'\" >> /etc/postgresql/14/main/postgresql.conf"
sudo systemctl restart postgresql
```

**macOS:**
```bash
brew install postgresql@14
brew install timescaledb

# Enable TimescaleDB
echo "shared_preload_libraries = 'timescaledb'" >> $(brew --prefix)/var/postgres/postgresql.conf
brew services restart postgresql@14
```

**Windows:**
1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Download and install TimescaleDB from [timescale.com](https://docs.timescale.com/install/latest/self-hosted/installation-windows/)

#### Docker and Docker Compose

**Ubuntu/Debian:**
```bash
# Install Docker
sudo apt update
sudo apt install docker.io

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to the docker group
sudo usermod -aG docker $USER
```

**macOS:**
```bash
brew install docker docker-compose
```

**Windows:**
Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)

#### Redis

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
```

**macOS:**
```bash
brew install redis
```

**Windows:**
Download and install Redis from [redis.io](https://redis.io/download)

#### Kafka

**Ubuntu/Debian:**
```bash
# Download Kafka
wget https://downloads.apache.org/kafka/3.4.0/kafka_2.13-3.4.0.tgz
tar -xzf kafka_2.13-3.4.0.tgz
cd kafka_2.13-3.4.0

# Start ZooKeeper
bin/zookeeper-server-start.sh config/zookeeper.properties &

# Start Kafka
bin/kafka-server-start.sh config/server.properties &
```

**macOS:**
```bash
brew install kafka
brew services start zookeeper
brew services start kafka
```

**Windows:**
Download and install Kafka from [kafka.apache.org](https://kafka.apache.org/downloads)

## Installation Methods

There are three ways to install the Forex Trading Platform:

1. **Using the Setup Script** (Recommended)
2. **Manual Installation**
3. **Docker Installation**

### 1. Using the Setup Script

The platform includes a setup script that automates the installation process:

```bash
# Clone the repository
git clone https://github.com/olevar2/MD.git
cd MD

# Run the setup script
./scripts/setup_platform.sh
```

The setup script will:
- Install required dependencies
- Set up the database
- Generate environment files
- Initialize the platform

### 2. Manual Installation

#### Clone the Repository

```bash
git clone https://github.com/olevar2/MD.git
cd MD
```

#### Set Up Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Set Up the Database

```bash
# Run the database setup script
python scripts/setup_database.py
```

#### Generate Environment Files

```bash
# Generate environment files for development
python scripts/generate_env_files.py --env development
```

#### Install Service Dependencies

```bash
# Install dependencies for each service
for service in */; do
    if [ -f "$service/requirements.txt" ]; then
        pip install -r "$service/requirements.txt"
    fi
done
```

### 3. Docker Installation

#### Clone the Repository

```bash
git clone https://github.com/olevar2/MD.git
cd MD
```

#### Build and Start Docker Containers

```bash
docker-compose up -d
```

This will build and start all the services defined in the `docker-compose.yml` file.

## Post-Installation Steps

### Load Initial Data

After installation, you can load sample data into the platform:

```bash
./scripts/load_initial_data.sh
```

### Start the Platform

To start the platform:

```bash
./scripts/start_platform.sh
```

### Verify Installation

To verify that the platform is running correctly:

```bash
./scripts/check_platform_health.sh
```

## Configuration

The platform can be configured through environment variables or configuration files. For detailed configuration options, see the [Configuration Guide](configuration.md).

## Troubleshooting

If you encounter issues during installation, check the [Troubleshooting Guide](troubleshooting.md) for common problems and solutions.

## Next Steps

After installation, you can:

- [Configure the platform](configuration.md)
- [Learn how to use the platform](user_guide.md)
- [Explore the API reference](api_reference.md)
- [Develop new features](development.md)
