import requests
import os
import hashlib
import getpass
from urllib.parse import urlparse
import random

_cached_username = None
_cached_password = None

def _get_credentials():
    global _cached_username, _cached_password
    if _cached_username is None or _cached_password is None:
        print("Authentication required:")
        _cached_username = input("Username: ")
        _cached_password = getpass.getpass("Password: ")
    return _cached_username, _cached_password

def download_file(url):
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
    response = requests.get(url)
    if response.status_code == 401:
        username, password = _get_credentials()
        response = requests.get(url, auth=(username, password))
    response.raise_for_status()
    file_tmp_path = file_path + str(random.randint(0, 10000)) + ".tmp"
    with open(file_tmp_path, "wb") as f:
        f.write(response.content)
    os.rename(file_tmp_path, file_path)
    return file_path
