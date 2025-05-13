"""
Service layer for the Model Registry Service.
"""
from typing import List, Dict, Optional, Any, BinaryIO, Union
from datetime import datetime, timedelta
import uuid

from models.model import (
    ModelMetadata,
    ModelVersion,
    ModelType,
    ModelStage,
    ModelFramework,
    ModelMetrics
)
from models.repository import ModelRegistryRepository
from core.abtest import ABTest
from models.exceptions import (
    ModelRegistryError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    InvalidModelError,
    ArtifactNotFoundError,
    InvalidStageTransitionError
)

class ModelRegistryService:
    """
    Core service for managing ML model versioning, tracking, and deployment.
    """

    def __init__(self, repository: ModelRegistryRepository):
    """
      init  .
    
    Args:
        repository: Description of repository
    
    """

        self.repository = repository

    async def register_model(
        self,
        name: str,
        model_type: Union[ModelType, str],
        description: Optional[str] = None,
        created_by: str = "system",
        tags: Dict[str, str] = None,
        business_domain: Optional[str] = None,
        purpose: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ModelMetadata:
        """Register a new model"""
        # Convert string model_type to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        model_id = str(uuid.uuid4())
        now = datetime.utcnow()

        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            model_type=model_type,
            description=description,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            business_domain=business_domain,
            purpose=purpose,
            tags=tags or {},
            metadata=metadata or {}
        )

        await self.repository.save_model_metadata(metadata)
        return metadata

    async def create_model_version(
        self,
        model_id: str,
        model_data: bytes,
        framework: Union[ModelFramework, str],
        framework_version: str,
        created_by: str,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: List[str] = None,
        target_names: List[str] = None,
        experiment_id: Optional[str] = None,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> ModelVersion:
        """Create a new version of an existing model"""
        # Get model metadata
        model = await self.repository.get_model_metadata(model_id)
        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        # Convert string framework to enum if needed
        if isinstance(framework, str):
            framework = ModelFramework(framework)

        # Generate version ID and number
        version_id = str(uuid.uuid4())
        version_number = len(model.versions) + 1
        now = datetime.utcnow()

        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version_number=version_number,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            description=description,
            framework=framework,
            framework_version=framework_version,
            metrics=metrics,
            hyperparameters=hyperparameters,
            feature_names=feature_names or [],
            target_names=target_names or [],
            stage=stage,
            experiment_id=experiment_id,
            tags=tags or {},
            metadata=metadata or {}
        )

        # Save model artifact
        artifact_uri = await self.repository.save_model_artifact(
            version_id=version_id,
            artifact_data=model_data,
            artifact_name="model.joblib"
        )
        version.artifact_uri = artifact_uri

        # Save version
        await self.repository.save_model_version(version)

        # Update model metadata
        model.versions.append(version)
        model.latest_version_id = version_id
        model.updated_at = now
        await self.repository.save_model_metadata(model)

        return version

    async def get_model(self, model_id: str) -> ModelMetadata:
        """Get model metadata by ID"""
        model = await self.repository.get_model_metadata(model_id)
        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")
        return model

    async def get_model_version(
        self,
        version_id: str,
        with_artifact: bool = False
    ) -> ModelVersion:
        """Get model version by ID"""
        version = await self.repository.get_model_version(version_id)
        if not version:
            raise ModelVersionNotFoundError(f"Version {version_id} not found")

        if with_artifact and version.artifact_uri:
            artifact = await self.repository.get_model_artifact(version_id, "model.joblib")
            if not artifact:
                raise ArtifactNotFoundError(f"Artifact not found for version {version_id}")
            # Add artifact data to metadata
            version.metadata["artifact_data"] = artifact

        return version

    async def list_models(
        self,
        name_filter: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        tags: Optional[Dict[str, str]] = None,
        business_domain: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List models matching criteria"""
        return await self.repository.list_models(
            name_filter=name_filter,
            model_type=model_type,
            tags=tags,
            business_domain=business_domain
        )

    async def update_version_stage(
        self,
        version_id: str,
        stage: Union[ModelStage, str]
    ) -> ModelVersion:
        """Update stage of a model version"""
        # Convert string stage to enum if needed
        if isinstance(stage, str):
            stage = ModelStage(stage)

        # Get version
        version = await self.repository.get_model_version(version_id)
        if not version:
            raise ModelVersionNotFoundError(f"Version {version_id} not found")

        # Get model metadata
        model = await self.repository.get_model_metadata(version.model_id)
        if not model:
            raise ModelNotFoundError(f"Model {version.model_id} not found")

        # Update version stage
        old_stage = version.stage
        version.stage = stage
        version.updated_at = datetime.utcnow()

        # Update model metadata
        if stage == ModelStage.PRODUCTION:
            # Only one version can be in production
            for v in model.versions:
                if v.version_id != version_id and v.stage == ModelStage.PRODUCTION:
                    v.stage = ModelStage.ARCHIVED
                    await self.repository.save_model_version(v)
            model.production_version_id = version_id
        elif stage == ModelStage.STAGING:
            model.staging_version_id = version_id
        elif old_stage == ModelStage.PRODUCTION:
            model.production_version_id = None
        elif old_stage == ModelStage.STAGING:
            model.staging_version_id = None

        # Save changes
        await self.repository.save_model_version(version)
        await self.repository.save_model_metadata(model)

        return version

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions"""
        # Check if model exists
        model = await self.repository.get_model_metadata(model_id)
        if not model:
            raise ModelNotFoundError(f"Model {model_id} not found")

        # Check if any version is in production
        for version in model.versions:
            if version.stage == ModelStage.PRODUCTION:
                raise ModelRegistryError(
                    f"Cannot delete model {model_id} because version {version.version_id} is in production"
                )

        # Delete model and all its versions
        success = await self.repository.delete_model(model_id)
        return success

    async def delete_model_version(self, version_id: str) -> bool:
        """Delete a specific model version"""
        # Get version
        version = await self.repository.get_model_version(version_id)
        if not version:
            raise ModelVersionNotFoundError(f"Version {version_id} not found")

        # Cannot delete production version
        if version.stage == ModelStage.PRODUCTION:
            raise ModelRegistryError(
                f"Cannot delete version {version_id} because it is in production"
            )

        # Delete version
        success = await self.repository.delete_model_version(version_id)
        if success:
            # Update model metadata
            model = await self.repository.get_model_metadata(version.model_id)
            if model:
                model.versions = [v for v in model.versions if v.version_id != version_id]
                if model.latest_version_id == version_id:
                    model.latest_version_id = model.versions[-1].version_id if model.versions else None
                if model.staging_version_id == version_id:
                    model.staging_version_id = None
                model.updated_at = datetime.utcnow()
                await self.repository.save_model_metadata(model)

        return success

    async def create_ab_test(
        self,
        model_id: str,
        version_ids: List[str],
        traffic_split: List[float],
        duration_days: Optional[int] = None,
        description: Optional[str] = None
    ) -> ABTest:
        """Create a new A/B test for a model's versions"""
        # Validate model exists
        model = await self.get_model(model_id)

        # Validate versions exist and belong to model
        versions = []
        for vid in version_ids:
            version = await self.get_model_version(vid)
            if version.model_id != model_id:
                raise InvalidModelError(f"Version {vid} does not belong to model {model_id}")
            versions.append(version)

        # Verify versions aren't archived
        for v in versions:
            if v.stage == ModelStage.ARCHIVED:
                raise InvalidModelError(f"Cannot use archived version {v.version_id} in A/B test")

        # Create A/B test
        test_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(days=duration_days) if duration_days else None
        
        test = ABTest(
            test_id=test_id,
            model_id=model_id,
            version_ids=version_ids,
            traffic_split=traffic_split,
            start_time=start_time,
            end_time=end_time,
            status="active"
        )

        # Save test
        await self.repository.save_ab_test(test)
        return test

    async def get_ab_test(self, test_id: str) -> ABTest:
        """Get an A/B test by ID"""
        test = await self.repository.get_ab_test(test_id)
        if not test:
            raise ModelRegistryError(f"A/B test {test_id} not found")
        return test

    async def list_ab_tests(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ABTest]:
        """List A/B tests matching criteria"""
        return await self.repository.list_ab_tests(model_id=model_id, status=status)

    async def update_ab_test(
        self,
        test_id: str,
        traffic_split: Optional[List[float]] = None,
        status: Optional[str] = None
    ) -> ABTest:
        """Update an A/B test"""
        test = await self.get_ab_test(test_id)
        
        if traffic_split:
            test.traffic_split = traffic_split
            test._validate()  # Validate new traffic split

        if status:
            valid_statuses = ["active", "completed", "cancelled"]
            if status not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
            test.status = status
            if status != "active":
                test.end_time = datetime.utcnow()

        await self.repository.save_ab_test(test)
        return test
