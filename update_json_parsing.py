"""
Update JSON Parsing

This script updates all response.json() calls in the feature store client to use optimized JSON parsing.
"""
import re
import os

# Path to the feature store client
client_path = "strategy-execution-engine/strategy_execution_engine/clients/feature_store_client.py"

# Read the file
with open(client_path, "r") as f:
    content = f.read()

# Replace all response.json() calls with parse_json_response(response)
content = re.sub(r"await response\.json\(\)", "await parse_json_response(response)", content)

# Replace all json.dumps calls with dumps
content = re.sub(r"json\.dumps\(", "dumps(", content)

# Replace all json.loads calls with loads
content = re.sub(r"json\.loads\(", "loads(", content)

# Write the updated content back to the file
with open(client_path, "w") as f:
    f.write(content)

print(f"Updated JSON parsing in {client_path}")
