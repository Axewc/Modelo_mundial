"""
gh_commit.py — Commit de un archivo al repositorio vía GitHub Contents API
==========================================================================
Permite que la app de Streamlit persista el dataset (y el track-record)
escribiéndolos de vuelta en GitHub. El token NUNCA va en el repo: se lee de
los Secrets de Streamlit y se pasa a estas funciones.

Módulo puro (sin Streamlit) para poder testearlo: acepta un `session`
inyectable. Usa la API REST de GitHub:
  GET  /repos/{repo}/contents/{path}  -> sha actual (para actualizar)
  PUT  /repos/{repo}/contents/{path}  -> crea/actualiza el archivo
"""
import base64

import requests

API = "https://api.github.com"


def _headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_sha(token, repo, path, branch="main", session=None):
    """SHA actual del archivo en la rama, o None si no existe todavía."""
    s = session or requests
    r = s.get(f"{API}/repos/{repo}/contents/{path}",
              params={"ref": branch}, headers=_headers(token), timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def commit_file(token, repo, path, content_str, message, branch="main", session=None):
    """Crea o actualiza `path` con `content_str` y devuelve la URL del commit.
    Lanza excepción si la API responde con error (token inválido, permisos…)."""
    s = session or requests
    sha = get_sha(token, repo, path, branch, session=s)
    payload = {
        "message": message,
        "branch": branch,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha  # requerido para actualizar un archivo existente
    r = s.put(f"{API}/repos/{repo}/contents/{path}",
              json=payload, headers=_headers(token), timeout=30)
    r.raise_for_status()
    j = r.json()
    return {
        "html_url": (j.get("commit") or {}).get("html_url"),
        "sha": (j.get("content") or {}).get("sha"),
    }
