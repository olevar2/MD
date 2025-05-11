# Naming Conventions

This document defines the naming conventions for the forex trading platform. Consistent naming conventions make the codebase more readable, maintainable, and easier to understand.

## Table of Contents

- [General Principles](#general-principles)
- [Python Conventions](#python-conventions)
  - [Modules and Packages](#modules-and-packages)
  - [Classes](#classes)
  - [Functions and Methods](#functions-and-methods)
  - [Variables and Constants](#variables-and-constants)
  - [Type Hints](#type-hints)
  - [Exceptions](#exceptions)
- [Database Conventions](#database-conventions)
  - [Tables](#tables)
  - [Columns](#columns)
  - [Indexes](#indexes)
  - [Constraints](#constraints)
- [API Conventions](#api-conventions)
  - [Endpoints](#endpoints)
  - [Request Parameters](#request-parameters)
  - [Response Fields](#response-fields)
- [File and Directory Conventions](#file-and-directory-conventions)
  - [Directory Structure](#directory-structure)
  - [File Names](#file-names)
  - [Configuration Files](#configuration-files)
- [Service Conventions](#service-conventions)
  - [Service Names](#service-names)
  - [Component Names](#component-names)
  - [Interface Names](#interface-names)
  - [Adapter Names](#adapter-names)
- [Event Conventions](#event-conventions)
  - [Event Names](#event-names)
  - [Event Fields](#event-fields)

## General Principles

1. **Consistency**: Be consistent with naming conventions throughout the codebase.
2. **Clarity**: Names should be clear, descriptive, and unambiguous.
3. **Conciseness**: Names should be concise while still being descriptive.
4. **Avoid Abbreviations**: Avoid abbreviations unless they are widely understood.
5. **Use English**: All names should be in English.

## Python Conventions

### Modules and Packages

- Use lowercase with underscores for module and package names.
- Module names should be short, descriptive, and singular.
- Package names should be short, descriptive, and singular.

```python
# Good
import market_data
from common_lib import config
from data_pipeline_service.adapters import market_data_adapter

# Bad
import MarketData
from common_lib import Config
from data_pipeline_service.Adapters import marketDataAdapter
```

### Classes

- Use CapWords (PascalCase) for class names.
- Class names should be singular and descriptive.
- Interface classes should start with `I` (e.g., `IMarketDataProvider`).
- Abstract classes should start with `Abstract` (e.g., `AbstractTokenizer`).
- Exception classes should end with `Error` (e.g., `ValidationError`).

```python
# Good
class MarketDataProvider:
    pass

class IMarketDataProvider:
    pass

class AbstractTokenizer:
    pass

class ValidationError(Exception):
    pass

# Bad
class marketDataProvider:
    pass

class Market_Data_Provider:
    pass

class marketdataprovider:
    pass
```

### Functions and Methods

- Use lowercase with underscores for function and method names.
- Function and method names should be descriptive and action-oriented.
- Private methods should start with a single underscore.
- "Magic" methods should start and end with double underscores.

```python
# Good
def calculate_moving_average(data, period):
    pass

def _validate_input(data):
    pass

def __init__(self, name):
    self.name = name

# Bad
def CalculateMovingAverage(data, period):
    pass

def calculate_MA(data, period):
    pass

def validateInput(data):
    pass
```

### Variables and Constants

- Use lowercase with underscores for variable names.
- Use UPPERCASE with underscores for constants.
- Use descriptive names that indicate the purpose of the variable.
- Avoid single-letter variable names except for loop counters.

```python
# Good
user_name = "John"
MAX_RETRY_COUNT = 3
for i in range(10):
    print(i)

# Bad
userName = "John"
max_retry_count = 3
for x in range(10):
    print(x)
```

### Type Hints

- Use CapWords (PascalCase) for type names.
- Use lowercase with underscores for type variable names.
- Use descriptive names for type variables.

```python
# Good
from typing import Dict, List, Optional, TypeVar

T = TypeVar('T')
UserID = int
UserDict = Dict[UserID, str]

def get_user(user_id: UserID) -> Optional[str]:
    pass

# Bad
from typing import Dict, List, Optional, TypeVar

t = TypeVar('t')
userId = int
userDict = Dict[userId, str]

def get_user(user_id: userId) -> Optional[str]:
    pass
```

### Exceptions

- Use CapWords (PascalCase) for exception names.
- Exception names should end with `Error`.
- Use descriptive names that indicate the type of error.

```python
# Good
class ValidationError(Exception):
    pass

class DatabaseConnectionError(Exception):
    pass

# Bad
class Validation_Error(Exception):
    pass

class DBConnError(Exception):
    pass
```

## Database Conventions

### Tables

- Use lowercase with underscores for table names.
- Table names should be plural and descriptive.
- Junction tables should be named after both tables they connect, in alphabetical order.

```sql
-- Good
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id)
);

CREATE TABLE orders_products (
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    PRIMARY KEY (order_id, product_id)
);

-- Bad
CREATE TABLE User (
    ID SERIAL PRIMARY KEY,
    Name TEXT NOT NULL
);

CREATE TABLE Order (
    ID SERIAL PRIMARY KEY,
    UserID INTEGER REFERENCES User(ID)
);

CREATE TABLE OrderProduct (
    OrderID INTEGER REFERENCES Order(ID),
    ProductID INTEGER REFERENCES Product(ID),
    PRIMARY KEY (OrderID, ProductID)
);
```

### Columns

- Use lowercase with underscores for column names.
- Primary key columns should be named `id`.
- Foreign key columns should be named `<table_name>_id` (singular).
- Boolean columns should be named with a prefix like `is_`, `has_`, or `can_`.
- Timestamp columns should be named with a suffix like `_at`.

```sql
-- Good
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total_amount DECIMAL(10, 2) NOT NULL,
    is_paid BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Bad
CREATE TABLE Users (
    ID SERIAL PRIMARY KEY,
    Name TEXT NOT NULL,
    Email TEXT UNIQUE NOT NULL,
    Active BOOLEAN DEFAULT TRUE,
    Created TIMESTAMP DEFAULT NOW(),
    Updated TIMESTAMP DEFAULT NOW()
);

CREATE TABLE Orders (
    ID SERIAL PRIMARY KEY,
    UserID INTEGER REFERENCES Users(ID),
    TotalAmount DECIMAL(10, 2) NOT NULL,
    Paid BOOLEAN DEFAULT FALSE,
    Created TIMESTAMP DEFAULT NOW()
);
```

### Indexes

- Use lowercase with underscores for index names.
- Index names should follow the pattern `<table_name>_<column_name(s)>_idx`.

```sql
-- Good
CREATE INDEX users_email_idx ON users (email);
CREATE INDEX orders_user_id_created_at_idx ON orders (user_id, created_at);

-- Bad
CREATE INDEX UserEmailIndex ON Users (Email);
CREATE INDEX idx_orders_user_created ON Orders (UserID, Created);
```

### Constraints

- Use lowercase with underscores for constraint names.
- Constraint names should follow the pattern `<table_name>_<column_name(s)>_<constraint_type>`.

```sql
-- Good
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);
ALTER TABLE orders ADD CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id);

-- Bad
ALTER TABLE Users ADD CONSTRAINT UniqueEmail UNIQUE (Email);
ALTER TABLE Orders ADD CONSTRAINT FK_Orders_Users FOREIGN KEY (UserID) REFERENCES Users(ID);
```

## API Conventions

### Endpoints

- Use lowercase with hyphens for endpoint paths.
- Use plural nouns for resource names.
- Use nested paths for related resources.
- Use verbs for actions that don't map to CRUD operations.

```
# Good
GET /api/v1/users
GET /api/v1/users/{id}
GET /api/v1/users/{id}/orders
POST /api/v1/users/{id}/reset-password

# Bad
GET /api/v1/User
GET /api/v1/User/{id}
GET /api/v1/User/{id}/Order
POST /api/v1/User/{id}/ResetPassword
```

### Request Parameters

- Use camelCase for request parameters.
- Use descriptive names that indicate the purpose of the parameter.

```json
// Good
{
  "userName": "john_doe",
  "email": "john@example.com",
  "isActive": true
}

// Bad
{
  "user_name": "john_doe",
  "Email": "john@example.com",
  "active": true
}
```

### Response Fields

- Use camelCase for response fields.
- Use descriptive names that indicate the purpose of the field.
- Use consistent field names across different endpoints.

```json
// Good
{
  "id": 123,
  "userName": "john_doe",
  "email": "john@example.com",
  "isActive": true,
  "createdAt": "2023-01-01T00:00:00Z"
}

// Bad
{
  "ID": 123,
  "user_name": "john_doe",
  "Email": "john@example.com",
  "active": true,
  "created": "2023-01-01T00:00:00Z"
}
```

## File and Directory Conventions

### Directory Structure

- Use lowercase with hyphens for directory names.
- Use descriptive names that indicate the purpose of the directory.
- Follow a consistent directory structure across services.

```
# Good
forex-trading-platform/
  common-lib/
  data-pipeline-service/
  feature-store-service/
  analysis-engine-service/
  trading-gateway-service/
  docs/
    standards/
    architecture/
  tests/
    unit/
    integration/
    performance/

# Bad
forex-trading-platform/
  CommonLib/
  DataPipelineService/
  feature_store_service/
  AnalysisEngine/
  trading-gateway/
  Docs/
    Standards/
    Architecture/
  Tests/
    Unit/
    Integration/
    Performance/
```

### File Names

- Use lowercase with underscores for Python file names.
- Use lowercase with hyphens for other file names.
- Use descriptive names that indicate the purpose of the file.
- Use appropriate file extensions.

```
# Good
market_data_provider.py
config_manager.py
README.md
docker-compose.yml
requirements.txt

# Bad
MarketDataProvider.py
ConfigManager.py
readme.md
dockercompose.yml
requirements
```

### Configuration Files

- Use lowercase with hyphens for configuration file names.
- Use YAML or JSON for configuration files.
- Use descriptive names that indicate the purpose of the configuration file.

```
# Good
config.yaml
development-config.yaml
production-config.yaml
logging-config.yaml

# Bad
Config.yaml
dev_config.yaml
prod.config.yaml
logging.conf
```

## Service Conventions

### Service Names

- Use lowercase with hyphens for service names.
- Use descriptive names that indicate the purpose of the service.
- End service names with `-service`.

```
# Good
market-data-service
feature-store-service
analysis-engine-service
trading-gateway-service

# Bad
MarketDataService
feature_store_service
AnalysisEngine
trading-gateway
```

### Component Names

- Use CapWords (PascalCase) for component names.
- Use descriptive names that indicate the purpose of the component.
- End component names with the component type (e.g., `Service`, `Repository`, `Controller`).

```python
# Good
class MarketDataService:
    pass

class UserRepository:
    pass

class AuthenticationController:
    pass

# Bad
class marketDataService:
    pass

class User_Repository:
    pass

class authController:
    pass
```

### Interface Names

- Use CapWords (PascalCase) for interface names.
- Start interface names with `I`.
- Use descriptive names that indicate the purpose of the interface.

```python
# Good
class IMarketDataProvider:
    pass

class IUserRepository:
    pass

class IAuthenticationService:
    pass

# Bad
class MarketDataProviderInterface:
    pass

class UserRepositoryI:
    pass

class iauthenticationservice:
    pass
```

### Adapter Names

- Use CapWords (PascalCase) for adapter names.
- End adapter names with `Adapter`.
- Use descriptive names that indicate the purpose of the adapter.

```python
# Good
class MarketDataProviderAdapter:
    pass

class FeatureStoreAdapter:
    pass

class AnalysisEngineAdapter:
    pass

# Bad
class MarketDataProviderAdpt:
    pass

class Feature_Store_Adapter:
    pass

class analysisEngineAdapter:
    pass
```

## Event Conventions

### Event Names

- Use CapWords (PascalCase) for event names.
- End event names with the event type (e.g., `Created`, `Updated`, `Deleted`).
- Use descriptive names that indicate the purpose of the event.

```python
# Good
class UserCreated:
    pass

class OrderPlaced:
    pass

class PaymentProcessed:
    pass

# Bad
class user_created:
    pass

class OrderPlacedEvent:
    pass

class payment_processed_event:
    pass
```

### Event Fields

- Use lowercase with underscores for event fields.
- Use descriptive names that indicate the purpose of the field.
- Include metadata fields like `event_id`, `event_type`, `timestamp`, and `version`.

```python
# Good
class UserCreated:
    def __init__(self, user_id, username, email, timestamp):
        self.event_id = str(uuid.uuid4())
        self.event_type = "user_created"
        self.timestamp = timestamp
        self.version = "1.0"
        self.user_id = user_id
        self.username = username
        self.email = email

# Bad
class UserCreated:
    def __init__(self, userId, userName, Email, Timestamp):
        self.EventId = str(uuid.uuid4())
        self.EventType = "UserCreated"
        self.Timestamp = Timestamp
        self.Version = "1.0"
        self.UserId = userId
        self.UserName = userName
        self.Email = Email
```
