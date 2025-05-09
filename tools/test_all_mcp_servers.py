import requests
import json
import sys

def test_mcp_server(url, name):
    """Test if an MCP server is running and accessible."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✅ {name} MCP server is running!")
            print(f"Server info:")
            print(json.dumps(response.json(), indent=2))
            print("\nAvailable tools:")
            tools = response.json().get("tools", [])
            for tool in tools:
                print(f"- {tool['name']}: {tool['description']}")
            print("\n" + "-" * 50 + "\n")
            return True
        else:
            print(f"❌ Error: {name} MCP server returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to {name} MCP server: {e}")
        return False

def main():
    """Test all configured MCP servers."""
    servers = [
        {"url": "http://localhost:3001/mcp/info", "name": "Todo"},
        {"url": "http://localhost:3000/mcp/info", "name": "MS-Todo"}
    ]
    
    success_count = 0
    for server in servers:
        if test_mcp_server(server["url"], server["name"]):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(servers)} MCP servers are running.")
    
    if success_count == len(servers):
        print("All MCP servers are running successfully! 🎉")
        return 0
    else:
        print(f"Some MCP servers are not running. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
