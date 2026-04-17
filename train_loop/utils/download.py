import requests
import os
import hashlib
import getpass
from urllib.parse import urlparse
import random
import time

_cached_username = None
_cached_password = None


def _load_dotenv_credentials():
    """Search for .env file up to 4 levels up from cwd (and home dir) and return (username, password) or (None, None)."""
    search_dirs = []
    d = os.getcwd()
    for _ in range(5):  # cwd + 4 levels up
        search_dirs.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    home = os.path.expanduser("~")
    if home not in search_dirs:
        search_dirs.append(home)

    for search_dir in search_dirs:
        for env_filename in (".env", "secrets.env"):
            env_path = os.path.join(search_dir, env_filename)
            if os.path.isfile(env_path):
                username = None
                password = None
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip()
                        if key == "download_http_username":
                            username = val
                        elif key == "download_http_password":
                            password = val
                if username is not None and password is not None:
                    return username, password
    return None, None


def _get_credentials():
    global _cached_username, _cached_password
    if _cached_username is None or _cached_password is None:
        # First try .env file
        username, password = _load_dotenv_credentials()
        if username is not None and password is not None:
            _cached_username = username
            _cached_password = password
        else:
            print("Authentication required:")
            _cached_username = input("Username: ")
            _cached_password = getpass.getpass("Password: ")
    return _cached_username, _cached_password


def download_file(url):
    rng = random.Random()
    rng.seed(int(time.time() * 1000000) + int(os.getpid()))

    if not url.startswith("http"):
        return url
    download_dir = os.path.expanduser("~/.downloads")
    os.makedirs(download_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed_url = urlparse(url)
    url_filename = os.path.basename(parsed_url.path)
    if url_filename:
        name, ext = os.path.splitext(url_filename)
        filename = f"{name}_{url_hash}{ext}"
    else:
        filename = url_hash
    file_path = os.path.join(download_dir, filename)
    print(file_path)
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return file_path

    # Try with .env creds upfront if available, otherwise unauthenticated first
    env_username, env_password = _load_dotenv_credentials()
    if env_username is not None:
        response = requests.get(url, auth=(env_username, env_password))
    else:
        response = requests.get(url)

    if response.status_code == 401:
        username, password = _get_credentials()
        response = requests.get(url, auth=(username, password))

    response.raise_for_status()
    file_tmp_path = file_path + str(rng.randint(0, 10000)) + ".tmp"
    with open(file_tmp_path, "wb") as f:
        f.write(response.content)
    os.rename(file_tmp_path, file_path)
    return file_path
