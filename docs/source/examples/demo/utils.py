import os
import tarfile
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download(url, output="data"):
    response = requests.get(url, stream=True)
    assert response.status_code == 200, "unable to download data"
    content_type = response.headers["Content-Type"]
    content = response.iter_content(chunk_size=int(1e6))
    bar_format = "Downloaded: {n}MB / {total}MB -{rate_fmt}, "
    bar_format += "Elapsed: {elapsed}, Remaining: {remaining}, Progress: {l_bar}{bar}"
    total = round(int(response.headers.get("content-length", 0)) / 1e6)
    content = tqdm(content, total=total, unit="MB", bar_format=bar_format)
    extract(content, content_type, output)
    response.close()


def extract(content, content_type, output):
    if content_type == "application/zip":
        extract_zip(content, output)
    elif content_type == "application/x-gzip":
        extract_tarball(content, output)
    else:
        raise TypeError(f'"{content_type}" not supported')


def extract_tarball(content, output):
    with open("data.tar.gz", "wb") as file:
        for chunk in content:
            file.write(chunk)

    with tarfile.open("data.tar.gz", "r:gz") as file:
        file.extractall(output)

    os.remove("data.tar.gz")


def extract_zip(content, output):
    with open("data.zip", "wb") as file:
        for chunk in content:
            file.write(chunk)

    with ZipFile("data.zip", "r") as file:
        file.extractall(output)

    os.remove("data.zip")
