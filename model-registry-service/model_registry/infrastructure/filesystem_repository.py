"""
Repository implementation using filesystem storage.
"""
from typing import Dict, List, Optional, Any
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

from model_registry.domain.repository import ModelRegistryRepository
from model_registry.domain.model import (
    ModelMetadata,
    ModelVersion,
    ModelStage,
    ModelType
)
from model_registry.core.exceptions import StorageError

class FilesystemModelRegistry(ModelRegistryRepository):
    """Implementation of ModelRegistryRepository using filesystem storage"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.versions_path = self.base_path / "versions"
        self.artifacts_path = self.base_path / "artifacts"

        # Create directories if they don't exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

    async def save_model_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to JSON file"""
        try:
            file_path = self.models_path / f"{metadata.model_id}.json"
            with open(file_path, "w") as f:
                json.dump(metadata.dict(), f, indent=2, default=str)
        except Exception as e:
            raise StorageError(f"Failed to save model metadata: {str(e)}")

    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Load model metadata from JSON file"""
        try:
            file_path = self.models_path / f"{model_id}.json"
            if not file_path.exists():
                return None
            with open(file_path, "r") as f:
                data = json.load(f)
                return ModelMetadata(**data)
        except Exception as e:
            raise StorageError(f"Failed to load model metadata: {str(e)}")

    async def list_models(
        self,
        name_filter: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        tags: Optional[Dict[str, str]] = None,
        business_domain: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List models matching criteria"""
        try:
            models = []
            for file_path in self.models_path.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        model = ModelMetadata(**data)

                        # Apply filters
                        if name_filter and name_filter.lower() not in model.name.lower():
                            continue
                        if model_type and model.model_type != model_type:
                            continue
                        if tags and not all(model.tags.get(k) == v for k, v in tags.items()):
                            continue
                        if business_domain and model.business_domain != business_domain:
                            continue

                        models.append(model)
                except Exception as e:
                    # Log error but continue processing other models
                    print(f"Error loading model from {file_path}: {str(e)}")
            return models
        except Exception as e:
            raise StorageError(f"Failed to list models: {str(e)}")

    async def save_model_version(self, version: ModelVersion) -> None:
        """Save model version to JSON file"""
        try:
            file_path = self.versions_path / f"{version.version_id}.json"
            with open(file_path, "w") as f:
                json.dump(version.dict(), f, indent=2, default=str)
        except Exception as e:
            raise StorageError(f"Failed to save model version: {str(e)}")

    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Load model version from JSON file"""
        try:
            file_path = self.versions_path / f"{version_id}.json"
            if not file_path.exists():
                return None
            with open(file_path, "r") as f:
                data = json.load(f)
                return ModelVersion(**data)
        except Exception as e:
            raise StorageError(f"Failed to load model version: {str(e)}")

    async def list_model_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """List versions for a model"""
        try:
            versions = []
            for file_path in self.versions_path.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        version = ModelVersion(**data)
                        if version.model_id == model_id:
                            if stage is None or version.stage == stage:
                                versions.append(version)
                except Exception as e:
                    # Log error but continue processing other versions
                    print(f"Error loading version from {file_path}: {str(e)}")
            return versions
        except Exception as e:
            raise StorageError(f"Failed to list model versions: {str(e)}")

    async def save_model_artifact(
        self,
        version_id: str,
        artifact_data: bytes,
        artifact_name: str
    ) -> str:
        """Save model artifact to file"""
        try:
            # Create version-specific artifact directory
            artifact_dir = self.artifacts_path / version_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Save artifact
            artifact_path = artifact_dir / artifact_name
            with open(artifact_path, "wb") as f:
                f.write(artifact_data)

            # Return URI (relative path from artifacts root)
            return f"{version_id}/{artifact_name}"
        except Exception as e:
            raise StorageError(f"Failed to save model artifact: {str(e)}")

    async def get_model_artifact(
        self,
        version_id: str,
        artifact_name: str
    ) -> Optional[bytes]:
        """Load model artifact from file"""
        try:
            artifact_path = self.artifacts_path / version_id / artifact_name
            if not artifact_path.exists():
                return None
            with open(artifact_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise StorageError(f"Failed to load model artifact: {str(e)}")

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions"""
        try:
            # Get model metadata
            model = await self.get_model_metadata(model_id)
            if not model:
                return False

            # Delete all version files and artifacts
            for version in model.versions:
                await self.delete_model_version(version.version_id)

            # Delete model metadata file
            metadata_path = self.models_path / f"{model_id}.json"
            metadata_path.unlink(missing_ok=True)

            return True
        except Exception as e:
            raise StorageError(f"Failed to delete model: {str(e)}")

    async def delete_model_version(self, version_id: str) -> bool:
        """Delete a specific model version"""
        try:
            # Delete version metadata file
            version_path = self.versions_path / f"{version_id}.json"
            if not version_path.exists():
                return False
            version_path.unlink()

            # Delete version artifacts directory
            artifact_dir = self.artifacts_path / version_id
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)

            return True
        except Exception as e:
            raise StorageError(f"Failed to delete model version: {str(e)}")

    async def save_ab_test(self, test: "ABTest") -> None:
        """Save A/B test metadata to JSON file"""
        try:
            test_dir = self.base_path / "abtests"
            test_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = test_dir / f"{test.test_id}.json"
            with open(file_path, "w") as f:
                json.dump(test.to_dict(), f, indent=2, default=str)
        except Exception as e:
            raise StorageError(f"Failed to save A/B test: {str(e)}")

    async def get_ab_test(self, test_id: str) -> Optional["ABTest"]:
        """Get A/B test metadata from JSON file"""
        try:
            test_dir = self.base_path / "abtests"
            file_path = test_dir / f"{test_id}.json"
            if not file_path.exists():
                return None
                
            with open(file_path, "r") as f:
                data = json.load(f)
                return ABTest.from_dict(data)
        except Exception as e:
            raise StorageError(f"Failed to load A/B test: {str(e)}")

    async def list_ab_tests(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List["ABTest"]:
        """List A/B tests matching criteria"""
        try:
            test_dir = self.base_path / "abtests"
            if not test_dir.exists():
                return []
                
            tests = []
            for file_path in test_dir.glob("*.json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    test = ABTest.from_dict(data)
                    
                    # Apply filters
                    if model_id and test.model_id != model_id:
                        continue
                    if status and test.status != status:
                        continue
                        
                    tests.append(test)
            return tests
        except Exception as e:
            raise StorageError(f"Failed to list A/B tests: {str(e)}")
