# DataManagementServiceInterface

*Generated on 2025-05-13 05:58:22*

## Description

Interface for data-management-service service.

## File

`data_management_service_interface.py`

## Methods

### get_status() -> Dict

Get the status of the service.

Returns:
    Service status information

#### Returns

- Dict

### get_data(dataset_id: str, start_date: Optional, end_date: Optional) -> List

Get data from the service.

Args:
    dataset_id: Dataset identifier
Args:
    start_date: Start date (ISO format)
Args:
    end_date: End date (ISO format)
Returns:
    List of data records

#### Parameters

- **dataset_id** (str)
- **start_date** (Optional)
- **end_date** (Optional)

#### Returns

- List

