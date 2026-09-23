"""Auto-manage K8s Secret for model API key authentication."""

import subprocess

from evalhub import ModelConfig
from evalhub.models.api import ModelAuth

MODEL_AUTH_SECRET = "model-api-key"


def ensure_model_auth_secret(api_key: str, namespace: str) -> str | None:
    """Create/update a K8s Secret from MODEL_API_KEY and return the secret name.

    If api_key is empty, returns None (no auth needed).
    The secret is idempotent — safe to call on every notebook run.
    """
    if not api_key:
        return None

    r = subprocess.run(
        [
            "oc", "create", "secret", "generic", MODEL_AUTH_SECRET,
            f"--from-literal=api-key={api_key}",
            "-n", namespace,
            "--dry-run=client", "-o", "yaml",
        ],
        capture_output=True, text=True,
    )
    subprocess.run(
        ["oc", "apply", "-f", "-", "-n", namespace],
        input=r.stdout, capture_output=True, text=True,
    )
    print(f"Model auth secret '{MODEL_AUTH_SECRET}' ready in namespace '{namespace}'")
    return MODEL_AUTH_SECRET


def build_model_config(endpoint: str, name: str, api_key: str, namespace: str) -> ModelConfig:
    """Build a ModelConfig with optional API key auth.

    - If api_key is set: creates K8s Secret and attaches ModelAuth
    - If api_key is empty: returns ModelConfig without auth
    """
    secret_name = ensure_model_auth_secret(api_key, namespace)
    auth = ModelAuth(secret_ref=secret_name) if secret_name else None
    return ModelConfig(url=endpoint, name=name, auth=auth)
