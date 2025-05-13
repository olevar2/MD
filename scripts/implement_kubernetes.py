#!/usr/bin/env python3
"""
Script to implement Kubernetes integration for services.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

def find_service_directories(root_dir: str) -> List[str]:
    """Find all service directories in the given directory."""
    service_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and (item.endswith('-service') or item.endswith('_service')):
            service_dirs.append(item_path)
    
    return service_dirs

def create_kubernetes_deployment_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes deployment."""
    template = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": service_name,
            "labels": {
                "app": service_name
            }
        },
        "spec": {
            "replicas": 2,
            "selector": {
                "matchLabels": {
                    "app": service_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": service_name
                    },
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/path": "/metrics",
                        "prometheus.io/port": "8000"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": service_name,
                            "image": f"forex-trading-platform/{service_name}:latest",
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [
                                {
                                    "containerPort": 8000,
                                    "name": "http"
                                }
                            ],
                            "env": [
                                {
                                    "name": "SERVICE_NAME",
                                    "value": service_name
                                },
                                {
                                    "name": "ENVIRONMENT",
                                    "value": "production"
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "value": "INFO"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "500m",
                                    "memory": "512Mi"
                                },
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "128Mi"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health/liveness",
                                    "port": "http"
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health/readiness",
                                    "port": "http"
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            }
                        }
                    ]
                }
            }
        }
    }
    
    return template

def create_kubernetes_service_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes service."""
    template = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "labels": {
                "app": service_name
            }
        },
        "spec": {
            "selector": {
                "app": service_name
            },
            "ports": [
                {
                    "port": 8000,
                    "targetPort": 8000,
                    "name": "http"
                }
            ],
            "type": "ClusterIP"
        }
    }
    
    return template

def create_kubernetes_configmap_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes configmap."""
    template = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{service_name}-config",
            "labels": {
                "app": service_name
            }
        },
        "data": {
            "config.yaml": """
# Service configuration
service:
  name: {service_name}
  port: 8000
  environment: production

# Logging configuration
logging:
  level: INFO
  format: json

# Database configuration
database:
  host: postgres
  port: 5432
  name: {service_name}
  user: postgres
  # password is stored in a secret

# Metrics configuration
metrics:
  enabled: true
  path: /metrics

# Tracing configuration
tracing:
  enabled: true
  endpoint: jaeger-collector:4317
""".format(service_name=service_name)
        }
    }
    
    return template

def create_kubernetes_secret_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes secret."""
    template = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{service_name}-secrets",
            "labels": {
                "app": service_name
            }
        },
        "type": "Opaque",
        "data": {
            "db-password": "cGFzc3dvcmQ=",  # base64 encoded "password"
            "api-key": "c2VjcmV0LWtleQ=="  # base64 encoded "secret-key"
        }
    }
    
    return template

def create_kubernetes_hpa_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes horizontal pod autoscaler."""
    template = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": service_name,
            "labels": {
                "app": service_name
            }
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": service_name
            },
            "minReplicas": 2,
            "maxReplicas": 10,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 70
                        }
                    }
                },
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 80
                        }
                    }
                }
            ]
        }
    }
    
    return template

def create_kubernetes_networkpolicy_template(service_name: str) -> Dict[str, Any]:
    """Create a template for a Kubernetes network policy."""
    template = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": f"{service_name}-network-policy",
            "labels": {
                "app": service_name
            }
        },
        "spec": {
            "podSelector": {
                "matchLabels": {
                    "app": service_name
                }
            },
            "ingress": [
                {
                    "from": [
                        {
                            "podSelector": {
                                "matchLabels": {
                                    "app": "api-gateway"
                                }
                            }
                        }
                    ],
                    "ports": [
                        {
                            "port": 8000,
                            "protocol": "TCP"
                        }
                    ]
                }
            ],
            "egress": [
                {
                    "to": [
                        {
                            "podSelector": {
                                "matchLabels": {
                                    "app": "postgres"
                                }
                            }
                        }
                    ],
                    "ports": [
                        {
                            "port": 5432,
                            "protocol": "TCP"
                        }
                    ]
                }
            ],
            "policyTypes": [
                "Ingress",
                "Egress"
            ]
        }
    }
    
    return template

def create_helm_chart(service_name: str, kubernetes_dir: str) -> None:
    """Create a Helm chart for a service."""
    # Create chart directory
    chart_dir = os.path.join(kubernetes_dir, service_name)
    os.makedirs(chart_dir, exist_ok=True)
    
    # Create templates directory
    templates_dir = os.path.join(chart_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create Chart.yaml
    chart_yaml = {
        "apiVersion": "v2",
        "name": service_name,
        "description": f"Helm chart for {service_name}",
        "type": "application",
        "version": "0.1.0",
        "appVersion": "0.1.0"
    }
    
    with open(os.path.join(chart_dir, 'Chart.yaml'), 'w') as f:
        yaml.dump(chart_yaml, f, default_flow_style=False)
    
    # Create values.yaml
    values_yaml = {
        "replicaCount": 2,
        "image": {
            "repository": f"forex-trading-platform/{service_name}",
            "tag": "latest",
            "pullPolicy": "IfNotPresent"
        },
        "service": {
            "type": "ClusterIP",
            "port": 8000
        },
        "resources": {
            "limits": {
                "cpu": "500m",
                "memory": "512Mi"
            },
            "requests": {
                "cpu": "100m",
                "memory": "128Mi"
            }
        },
        "autoscaling": {
            "enabled": True,
            "minReplicas": 2,
            "maxReplicas": 10,
            "targetCPUUtilizationPercentage": 70,
            "targetMemoryUtilizationPercentage": 80
        },
        "nodeSelector": {},
        "tolerations": [],
        "affinity": {},
        "env": {
            "SERVICE_NAME": service_name,
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO"
        },
        "config": {
            "service": {
                "name": service_name,
                "port": 8000,
                "environment": "production"
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            },
            "database": {
                "host": "postgres",
                "port": 5432,
                "name": service_name,
                "user": "postgres"
            },
            "metrics": {
                "enabled": True,
                "path": "/metrics"
            },
            "tracing": {
                "enabled": True,
                "endpoint": "jaeger-collector:4317"
            }
        },
        "secrets": {
            "dbPassword": "password",
            "apiKey": "secret-key"
        }
    }
    
    with open(os.path.join(chart_dir, 'values.yaml'), 'w') as f:
        yaml.dump(values_yaml, f, default_flow_style=False)
    
    # Create deployment.yaml
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "chart.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "chart.selectorLabels" . | nindent 8 }}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          env:
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ include "chart.fullname" . }}-secrets
                  key: db-password
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ include "chart.fullname" . }}-secrets
                  key: api-key
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          livenessProbe:
            httpGet:
              path: /health/liveness
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/readiness
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
      volumes:
        - name: config-volume
          configMap:
            name: {{ include "chart.fullname" . }}-config
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
"""
    
    with open(os.path.join(templates_dir, 'deployment.yaml'), 'w') as f:
        f.write(deployment_yaml)
    
    # Create service.yaml
    service_yaml = """
apiVersion: v1
kind: Service
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "chart.selectorLabels" . | nindent 4 }}
"""
    
    with open(os.path.join(templates_dir, 'service.yaml'), 'w') as f:
        f.write(service_yaml)
    
    # Create configmap.yaml
    configmap_yaml = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "chart.fullname" . }}-config
  labels:
    {{- include "chart.labels" . | nindent 4 }}
data:
  config.yaml: |
    # Service configuration
    service:
      name: {{ .Values.config.service.name }}
      port: {{ .Values.config.service.port }}
      environment: {{ .Values.config.service.environment }}

    # Logging configuration
    logging:
      level: {{ .Values.config.logging.level }}
      format: {{ .Values.config.logging.format }}

    # Database configuration
    database:
      host: {{ .Values.config.database.host }}
      port: {{ .Values.config.database.port }}
      name: {{ .Values.config.database.name }}
      user: {{ .Values.config.database.user }}
      # password is stored in a secret

    # Metrics configuration
    metrics:
      enabled: {{ .Values.config.metrics.enabled }}
      path: {{ .Values.config.metrics.path }}

    # Tracing configuration
    tracing:
      enabled: {{ .Values.config.tracing.enabled }}
      endpoint: {{ .Values.config.tracing.endpoint }}
"""
    
    with open(os.path.join(templates_dir, 'configmap.yaml'), 'w') as f:
        f.write(configmap_yaml)
    
    # Create secret.yaml
    secret_yaml = """
apiVersion: v1
kind: Secret
metadata:
  name: {{ include "chart.fullname" . }}-secrets
  labels:
    {{- include "chart.labels" . | nindent 4 }}
type: Opaque
data:
  db-password: {{ .Values.secrets.dbPassword | b64enc }}
  api-key: {{ .Values.secrets.apiKey | b64enc }}
"""
    
    with open(os.path.join(templates_dir, 'secret.yaml'), 'w') as f:
        f.write(secret_yaml)
    
    # Create hpa.yaml
    hpa_yaml = """
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "chart.fullname" . }}
  labels:
    {{- include "chart.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "chart.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
"""
    
    with open(os.path.join(templates_dir, 'hpa.yaml'), 'w') as f:
        f.write(hpa_yaml)
    
    # Create _helpers.tpl
    helpers_tpl = """
{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "chart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "chart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "chart.labels" -}}
helm.sh/chart: {{ include "chart.chart" . }}
{{ include "chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
"""
    
    with open(os.path.join(templates_dir, '_helpers.tpl'), 'w') as f:
        f.write(helpers_tpl)
    
    print(f"Created Helm chart for {service_name}")

def main():
    """Main function to implement Kubernetes integration."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create kubernetes directory
    kubernetes_dir = os.path.join(root_dir, 'kubernetes')
    os.makedirs(kubernetes_dir, exist_ok=True)
    
    # Find service directories
    service_dirs = find_service_directories(root_dir)
    
    # Create Helm charts for each service
    for service_dir in service_dirs:
        service_name = os.path.basename(service_dir).replace('_', '-')
        create_helm_chart(service_name, kubernetes_dir)
    
    print("Kubernetes integration completed.")

if __name__ == '__main__':
    main()
