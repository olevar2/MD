import requests
import json

def test_mcp_server():
    """Test if the MCP server is running and accessible."""
    try:
        response = requests.get("http://localhost:3001/mcp/info")
        if response.status_code == 200:
            print("MCP server is running!")
            print("Server info:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        return False

if __name__ == "__main__":
    test_mcp_server()
