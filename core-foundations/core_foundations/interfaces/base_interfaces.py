"""
Base Interfaces Module.

Defines abstract base classes and interfaces to be implemented by different services.
"""

import abc
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

# Generic type variables
T = TypeVar('T')
ID = TypeVar('ID')


class IService(abc.ABC):
    """
    Base interface for all services in the forex trading platform.
    
    Services represent the business logic layer and coordinate operations
    across one or more repositories or other dependencies.
    """
    
    @abc.abstractmethod
    def get_service_name(self) -> str:
        """
        Get the name of the service.
        
        Returns:
            Name of the service
        """
        pass
    
    @abc.abstractmethod
    def get_service_version(self) -> str:
        """
        Get the version of the service.
        
        Returns:
            Version string
        """
        pass
    
    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the service.
        
        Returns:
            Dictionary with health check results
        """
        pass


class IRepository(Generic[T, ID], abc.ABC):
    """
    Base interface for data repositories in the forex trading platform.
    
    Repositories are responsible for data storage and retrieval from
    a specific data source (e.g., database, API).
    """
    
    @abc.abstractmethod
    def find_by_id(self, entity_id: ID) -> Optional[T]:
        """
        Find an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to find
            
        Returns:
            The entity if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    def find_all(self) -> List[T]:
        """
        Find all entities.
        
        Returns:
            List of all entities
        """
        pass
    
    @abc.abstractmethod
    def save(self, entity: T) -> T:
        """
        Save an entity.
        
        Args:
            entity: The entity to save
            
        Returns:
            The saved entity (potentially with updated fields)
        """
        pass
    
    @abc.abstractmethod
    def delete(self, entity_id: ID) -> bool:
        """
        Delete an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        pass


class IEntity(abc.ABC):
    """
    Base interface for domain entities in the forex trading platform.
    
    Entities represent persistent domain objects with identity.
    """
    
    @abc.abstractproperty
    def id(self) -> Any:
        """
        Get the unique identifier of the entity.
        
        Returns:
            The entity's ID
        """
        pass


class IDataSource(abc.ABC):
    """
    Base interface for data sources in the forex trading platform.
    
    Data sources represent external systems that provide data
    (e.g., market data providers, brokers).
    """
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the data source is connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the data source.
        
        Returns:
            Dictionary with health check results
        """
        pass


class IMessageHandler(Generic[T], abc.ABC):
    """
    Base interface for message handlers in the forex trading platform.
    
    Message handlers are responsible for processing messages from queues or streams.
    """
    
    @abc.abstractmethod
    def handle(self, message: T) -> None:
        """
        Handle a received message.
        
        Args:
            message: The message to handle
        """
        pass
    
    @abc.abstractmethod
    def can_handle(self, message_type: str) -> bool:
        """
        Check if this handler can handle a specific message type.
        
        Args:
            message_type: The type of message
            
        Returns:
            True if this handler can handle the message type, False otherwise
        """
        pass


class IStrategyComponent(abc.ABC):
    """
    Base interface for trading strategy components.
    
    Strategy components represent parts of a trading strategy that
    can be composed to build complete strategies.
    """
    
    @abc.abstractmethod
    def compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute strategy component output based on input data.
        
        Args:
            data: Input data for computation
            
        Returns:
            Computation results
        """
        pass
    
    @abc.abstractmethod
    def get_required_inputs(self) -> List[str]:
        """
        Get the list of required input keys for this component.
        
        Returns:
            List of required input keys
        """
        pass
    
    @abc.abstractmethod
    def get_provided_outputs(self) -> List[str]:
        """
        Get the list of output keys this component provides.
        
        Returns:
            List of provided output keys
        """
        pass


class IModelTrainer(Generic[T], abc.ABC):
    """
    Base interface for machine learning model trainers.

    Model trainers are responsible for training machine learning models
    with specific algorithms and hyperparameters.
    """

    @abc.abstractmethod
    def train(self, training_data: T) -> Dict[str, Any]:
        """
        Train a model using the provided training data.

        Args:
            training_data: Data to use for training

        Returns:
            Training results including model performance metrics
        """
        pass

    @abc.abstractmethod
    def validate(self, validation_data: T) -> Dict[str, Any]:
        """
        Validate a trained model using the provided validation data.

        Args:
            validation_data: Data to use for validation

        Returns:
            Validation results including model performance metrics
        """
        pass

    @abc.abstractmethod
    def save_model(self, path: str) -> bool:
        """
        Save the trained model to the specified path.

        Args:
            path: Path where the model should be saved

        Returns:
            True if the model was saved successfully, False otherwise
        """
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from the specified path.

        Args:
            path: Path from where the model should be loaded

        Returns:
            True if the model was loaded successfully, False otherwise
        """
        pass

    @abc.abstractmethod
    def get_model(self) -> Optional[Any]:
        """
        Get the underlying trained model object.

        Returns:
            The trained model object or None if not trained/loaded.
        """
        pass

    @abc.abstractmethod
    def predict(self, input_data: Any) -> Any:
        """
        Make predictions using the trained model.

        Args:
            input_data: Data to make predictions on.

        Returns:
            The prediction results.
        """
        pass