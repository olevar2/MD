"""
Mock service module.

This module provides functionality for...
"""


import http.server
import socketserver
import json
import time
from urllib.parse import urlparse, parse_qs

PORT = 8006

class MockHandler(http.server.BaseHTTPRequestHandler):
    """
    MockHandler class that inherits from http.server.BaseHTTPRequestHandler.
    
    Attributes:
        Add attributes here
    """

    def do_GET(self):
    """
    Do get.
    
    """

        path = urlparse(self.path).path

        # Health check endpoint
        if path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy"}
            self.wfile.write(json.dumps(response).encode())
            return

        # API endpoints
        for endpoint_config in [{'method': 'GET', 'endpoint': '/api/v1/models', 'expected_status': 200}, {'method': 'GET', 'endpoint': '/api/v1/predictions', 'expected_status': 200}]:
            if path == endpoint_config['endpoint']:
                self.send_response(endpoint_config['expected_status'])
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"status": "success", "data": []}
                self.wfile.write(json.dumps(response).encode())
                return

        # Test interaction endpoints
        if path.startswith('/api/v1/test/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "success", "message": "Interaction successful"}
            self.wfile.write(json.dumps(response).encode())
            return

        # Default response for unknown endpoints
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"status": "error", "message": "Endpoint not found"}
        self.wfile.write(json.dumps(response).encode())

print(f"Starting mock service for ml-integration-service on port {PORT}")
httpd = socketserver.TCPServer(("", PORT), MockHandler)
httpd.serve_forever()
