# Forex Trading Platform Code Review Guidelines

This document defines guidelines for code reviews in the Forex Trading Platform. These guidelines are designed to ensure code quality, maintainability, and alignment with domain concepts.

## Table of Contents

1. [Code Review Principles](#code-review-principles)
2. [Code Review Process](#code-review-process)
3. [Code Review Checklist](#code-review-checklist)
4. [Domain-Specific Review Guidelines](#domain-specific-review-guidelines)
5. [Feedback Guidelines](#feedback-guidelines)
6. [Resolving Disagreements](#resolving-disagreements)
7. [Phased Adoption Approach](#phased-adoption-approach)

## Code Review Principles

1. **Focus on Education**: Code reviews should be a learning opportunity for both the author and the reviewer
2. **Constructive Feedback**: Provide constructive feedback that helps improve the code
3. **Domain Alignment**: Ensure code aligns with domain concepts and terminology
4. **Maintainability First**: Prioritize maintainability over clever or overly complex solutions
5. **Respect**: Treat code authors with respect and assume good intentions
6. **Timeliness**: Review code promptly to avoid blocking progress
7. **Thoroughness**: Review code thoroughly, but don't let perfect be the enemy of good

## Code Review Process

### Pre-Review Preparation

**For Code Authors:**

1. **Self-Review**: Review your own code before submitting for review
2. **Tests**: Ensure all tests pass and new code has appropriate test coverage
3. **Documentation**: Update documentation as needed
4. **Description**: Provide a clear description of the changes and their purpose
5. **Scope**: Keep changes focused and reasonably sized (ideally < 400 lines)

**For Code Reviewers:**

1. **Context**: Understand the purpose and context of the changes
2. **Domain Knowledge**: Ensure you have sufficient domain knowledge to review the code
3. **Time Allocation**: Allocate sufficient time for a thorough review
4. **Tools**: Use automated tools to assist with the review

### Review Process

1. **Automated Checks**: Run automated checks (linting, formatting, tests)
2. **High-Level Review**: Review the overall design and architecture
3. **Detailed Review**: Review the code in detail, focusing on maintainability
4. **Documentation Review**: Review documentation for clarity and completeness
5. **Test Review**: Review tests for coverage and correctness
6. **Feedback**: Provide constructive feedback with clear explanations
7. **Follow-Up**: Follow up on feedback and ensure issues are addressed

### Post-Review

1. **Approval**: Approve the changes once all issues are addressed
2. **Merge**: Merge the changes into the target branch
3. **Deployment**: Deploy the changes to the appropriate environment
4. **Monitoring**: Monitor the changes for any issues
5. **Learning**: Identify learning opportunities for the team

## Code Review Checklist

### General

- [ ] Code follows the established coding standards
- [ ] Code is well-organized and follows the established file structure
- [ ] Code is easy to understand and maintain
- [ ] Code has appropriate error handling
- [ ] Code has appropriate logging
- [ ] Code has appropriate documentation
- [ ] Code has appropriate tests
- [ ] Code has appropriate performance characteristics
- [ ] Code has appropriate security measures

### Domain Alignment

- [ ] Code uses domain terminology consistently
- [ ] Code follows domain-driven design principles
- [ ] Code respects domain boundaries
- [ ] Code implements domain logic correctly
- [ ] Code handles domain-specific edge cases

### Maintainability

- [ ] Code is modular and follows single responsibility principle
- [ ] Code avoids duplication
- [ ] Code uses appropriate abstractions
- [ ] Code is testable
- [ ] Code is extensible
- [ ] Code is configurable
- [ ] Code avoids unnecessary complexity
- [ ] Code has appropriate comments for complex logic
- [ ] Code uses meaningful names for variables, functions, and classes

### Performance

- [ ] Code has appropriate performance characteristics for its use case
- [ ] Code avoids unnecessary database queries
- [ ] Code uses appropriate caching strategies
- [ ] Code uses appropriate data structures and algorithms
- [ ] Code avoids unnecessary memory usage
- [ ] Code avoids unnecessary CPU usage
- [ ] Code avoids unnecessary network calls

### Security

- [ ] Code validates input appropriately
- [ ] Code sanitizes output appropriately
- [ ] Code handles sensitive data appropriately
- [ ] Code uses appropriate authentication and authorization
- [ ] Code avoids common security vulnerabilities
- [ ] Code uses secure defaults
- [ ] Code follows the principle of least privilege

## Domain-Specific Review Guidelines

### Market Data Domain

- [ ] Code handles different timeframes correctly
- [ ] Code handles different data types correctly (ticks, OHLCV, etc.)
- [ ] Code handles missing data appropriately
- [ ] Code handles data quality issues appropriately
- [ ] Code uses appropriate data structures for time series data
- [ ] Code handles timezone issues correctly
- [ ] Code uses appropriate precision for financial data

### Trading Domain

- [ ] Code handles different order types correctly
- [ ] Code handles different position types correctly
- [ ] Code handles risk management appropriately
- [ ] Code handles order execution correctly
- [ ] Code handles order lifecycle correctly
- [ ] Code handles position tracking correctly
- [ ] Code handles account management correctly
- [ ] Code uses appropriate precision for financial calculations

### Analysis Domain

- [ ] Code implements technical indicators correctly
- [ ] Code implements pattern recognition correctly
- [ ] Code implements signal generation correctly
- [ ] Code handles different timeframes correctly
- [ ] Code handles different data types correctly
- [ ] Code uses appropriate data structures for analysis
- [ ] Code handles edge cases appropriately
- [ ] Code uses appropriate precision for calculations

## Feedback Guidelines

### Constructive Feedback

- **Be Specific**: Provide specific examples and suggestions
- **Be Objective**: Focus on the code, not the person
- **Be Constructive**: Suggest improvements, not just point out issues
- **Be Respectful**: Assume good intentions and treat others with respect
- **Be Clear**: Communicate clearly and avoid ambiguity
- **Be Helpful**: Provide resources and examples when appropriate
- **Be Balanced**: Acknowledge good aspects as well as areas for improvement

### Feedback Examples

#### Positive Examples

- "This implementation of the order validation logic is clear and handles all the edge cases we discussed."
- "I like how you've organized the market data processing pipeline. It's easy to follow and maintain."
- "The way you've implemented the caching strategy for indicators is efficient and handles invalidation correctly."

#### Constructive Examples

- "The order validation logic could be more maintainable if we extract the validation rules into separate functions. This would make it easier to test and extend."
- "The market data processing pipeline might be more efficient if we use a streaming approach instead of loading all data into memory. This would help with large datasets."
- "The caching strategy for indicators might miss some invalidation cases. Consider adding a check for timeframe changes as well."

#### Negative Examples (Avoid)

- "This code is a mess. You should rewrite it."
- "Why would you implement it this way? It's obviously wrong."
- "This is not how we do things here. You should know better."

## Resolving Disagreements

1. **Understand the Other Perspective**: Try to understand the other person's perspective
2. **Focus on Principles**: Focus on principles and goals rather than personal preferences
3. **Provide Evidence**: Support your position with evidence and examples
4. **Consider Alternatives**: Consider alternative approaches that address both perspectives
5. **Seek Mediation**: If necessary, seek mediation from a third party
6. **Document Decisions**: Document the decision and the rationale for future reference
7. **Move Forward**: Once a decision is made, move forward constructively

## Phased Adoption Approach

### Phase 1: Education and Awareness

- Share code review guidelines with the team
- Conduct training sessions on code review best practices
- Start with informal code reviews focused on learning

### Phase 2: Structured Reviews for Critical Components

- Implement structured code reviews for critical components
- Use the code review checklist for these reviews
- Focus on high-impact areas like financial calculations and risk management

### Phase 3: Team-Wide Adoption

- Extend structured code reviews to all components
- Integrate code reviews into the development process
- Collect feedback and refine the process

### Phase 4: Continuous Improvement

- Regularly review and update the code review guidelines
- Collect metrics on code review effectiveness
- Identify areas for improvement
- Share learnings across teams

## Code Review Examples

### Example 1: Order Service Implementation

#### Code Submission

```python
class OrderService:
    def __init__(self, broker_client, db_client):
        self.broker = broker_client
        self.db = db_client
        
    def place_order(self, order_data):
        # Create order in database
        order_id = self.db.create_order(order_data)
        
        # Send order to broker
        broker_order_id = self.broker.place_order(order_data)
        
        # Update order with broker ID
        self.db.update_order(order_id, {"broker_order_id": broker_order_id})
        
        return order_id
```

#### Review Feedback

**Maintainability:**
- The `place_order` method has multiple responsibilities (creating order, sending to broker, updating order). Consider separating these concerns.
- There's no error handling for broker communication failures or database operations.
- The method doesn't validate the order data before processing.

**Domain Alignment:**
- The method doesn't use domain terminology consistently. Consider using a domain model for orders instead of generic `order_data`.
- The method doesn't handle different order types or validation rules that might be specific to order types.

**Suggested Improvements:**
```python
from common_lib.models import Order
from common_lib.errors import OrderValidationError, BrokerError, DatabaseError

class OrderService:
    def __init__(self, broker_client, db_client, validator):
        self.broker = broker_client
        self.db = db_client
        self.validator = validator
        self.logger = logging.getLogger(__name__)
        
    def place_order(self, order_data: dict) -> str:
        """
        Place a new order with the broker.
        
        Args:
            order_data: The order data
            
        Returns:
            The order ID
            
        Raises:
            OrderValidationError: If the order data is invalid
            BrokerError: If there's an error communicating with the broker
            DatabaseError: If there's an error with the database
        """
        try:
            # Validate order data
            self.validator.validate_order(order_data)
            
            # Create order domain model
            order = Order.from_dict(order_data)
            
            # Create order in database
            try:
                order_id = self.db.create_order(order)
                order.id = order_id
            except Exception as e:
                self.logger.error(f"Failed to create order in database: {e}")
                raise DatabaseError(f"Failed to create order: {e}")
            
            # Send order to broker
            try:
                broker_order_id = self.broker.place_order(order)
                order.broker_order_id = broker_order_id
            except Exception as e:
                self.logger.error(f"Failed to place order with broker: {e}")
                # Update order status to FAILED
                self.db.update_order(order_id, {"status": "FAILED", "error": str(e)})
                raise BrokerError(f"Failed to place order with broker: {e}")
            
            # Update order with broker ID
            try:
                self.db.update_order(order_id, {"broker_order_id": broker_order_id, "status": "PENDING"})
            except Exception as e:
                self.logger.error(f"Failed to update order in database: {e}")
                # Order is placed with broker but not updated in database
                # This requires manual reconciliation
                self.logger.critical(f"Order placed with broker but not updated in database. Order ID: {order_id}, Broker Order ID: {broker_order_id}")
                raise DatabaseError(f"Failed to update order: {e}")
            
            self.logger.info(f"Order placed successfully. Order ID: {order_id}, Broker Order ID: {broker_order_id}")
            return order_id
            
        except OrderValidationError as e:
            self.logger.warning(f"Order validation failed: {e}")
            raise
        except (BrokerError, DatabaseError) as e:
            self.logger.error(f"Order placement failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during order placement: {e}")
            raise OrderValidationError(f"Unexpected error: {e}")
```

### Example 2: Market Data Processing

#### Code Submission

```python
def process_market_data(data, timeframe):
    result = []
    for item in data:
        timestamp = item["timestamp"]
        open_price = item["open"]
        high_price = item["high"]
        low_price = item["low"]
        close_price = item["close"]
        volume = item["volume"]
        
        # Convert timestamp to timeframe
        if timeframe == "1h":
            timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        elif timeframe == "1d":
            timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if we need to create a new candle or update existing
        if not result or result[-1]["timestamp"] != timestamp:
            result.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        else:
            # Update existing candle
            result[-1]["high"] = max(result[-1]["high"], high_price)
            result[-1]["low"] = min(result[-1]["low"], low_price)
            result[-1]["close"] = close_price
            result[-1]["volume"] += volume
    
    return result
```

#### Review Feedback

**Maintainability:**
- The function handles multiple responsibilities (timestamp conversion, candle creation/updating). Consider separating these concerns.
- The timeframe handling is limited to hardcoded values ("1h", "1d"). This won't scale to other timeframes.
- There's no validation of input data or handling of missing fields.

**Domain Alignment:**
- The function doesn't use domain terminology consistently. Consider using domain models for OHLCV data.
- The function doesn't handle different timeframe conventions that might exist in the forex domain.

**Suggested Improvements:**
```python
from datetime import datetime
from typing import List, Dict, Any
from common_lib.models import OHLCV, Timeframe
from common_lib.errors import MarketDataError

def align_timestamp_to_timeframe(timestamp: datetime, timeframe: Timeframe) -> datetime:
    """
    Align a timestamp to a specific timeframe.
    
    Args:
        timestamp: The timestamp to align
        timeframe: The timeframe to align to
        
    Returns:
        The aligned timestamp
        
    Raises:
        ValueError: If the timeframe is invalid
    """
    if timeframe == Timeframe.MINUTE_1:
        return timestamp.replace(second=0, microsecond=0)
    elif timeframe == Timeframe.MINUTE_5:
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 5, second=0, microsecond=0)
    elif timeframe == Timeframe.MINUTE_15:
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 15, second=0, microsecond=0)
    elif timeframe == Timeframe.MINUTE_30:
        return timestamp.replace(minute=timestamp.minute - timestamp.minute % 30, second=0, microsecond=0)
    elif timeframe == Timeframe.HOUR_1:
        return timestamp.replace(minute=0, second=0, microsecond=0)
    elif timeframe == Timeframe.HOUR_4:
        return timestamp.replace(hour=timestamp.hour - timestamp.hour % 4, minute=0, second=0, microsecond=0)
    elif timeframe == Timeframe.DAY_1:
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def process_market_data(data: List[Dict[str, Any]], timeframe: Timeframe) -> List[OHLCV]:
    """
    Process raw market data into OHLCV data for a specific timeframe.
    
    Args:
        data: The raw market data
        timeframe: The timeframe to process for
        
    Returns:
        The processed OHLCV data
        
    Raises:
        MarketDataError: If the data is invalid or processing fails
    """
    if not data:
        return []
    
    result = []
    
    try:
        for item in data:
            # Validate required fields
            required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
            for field in required_fields:
                if field not in item:
                    raise MarketDataError(f"Missing required field: {field}")
            
            # Parse timestamp if it's a string
            timestamp = item["timestamp"]
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    raise MarketDataError(f"Invalid timestamp format: {timestamp}")
            
            # Align timestamp to timeframe
            aligned_timestamp = align_timestamp_to_timeframe(timestamp, timeframe)
            
            # Create or update candle
            if not result or result[-1].timestamp != aligned_timestamp:
                # Create new candle
                candle = OHLCV(
                    timestamp=aligned_timestamp,
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"])
                )
                result.append(candle)
            else:
                # Update existing candle
                result[-1].high = max(result[-1].high, float(item["high"]))
                result[-1].low = min(result[-1].low, float(item["low"]))
                result[-1].close = float(item["close"])
                result[-1].volume += float(item["volume"])
        
        return result
    
    except MarketDataError:
        raise
    except Exception as e:
        raise MarketDataError(f"Failed to process market data: {e}")
```

## Conclusion

Code reviews are a critical part of our development process. They help ensure code quality, maintainability, and alignment with domain concepts. By following these guidelines, we can make code reviews a positive and productive experience for everyone involved.

Remember that the goal of code reviews is not to find fault, but to improve the code and help each other grow as developers. Approach code reviews with a mindset of collaboration and learning, and they will be a valuable tool for improving our codebase and our team.