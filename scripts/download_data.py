import os
import requests
import zipfile
import fire


def download_and_extract(url: str, target_dir: str) -> None:
    """Downloads a ZIP file from a URL, extracts it, and deletes the ZIP file after extraction.

    Args:
        url (str): The URL to download the ZIP file from.
        target_dir (str): The directory to extract the ZIP file to.
    """
    if not target_dir or not isinstance(target_dir, str):
        raise ValueError("Target directory must be a valid non-empty string.")

    # Normalize and validate target_dir
    target_dir = os.path.normpath(target_dir)
    print(f"Normalized target directory: {target_dir}")

    # Ensure the target directory exists
    print(f"Creating directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    # Path to the ZIP file
    zip_path = os.path.join(target_dir, "dataset.zip")
    print(f"ZIP path: {zip_path}")

    # Download the ZIP file
    print(f"Downloading file from {url} to {zip_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded file to {zip_path}.")

    # Extract the ZIP file
    print(f"Extracting {zip_path} to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extracted files to {target_dir}.")

    # Delete the ZIP file
    print(f"Deleting {zip_path}...")
    os.remove(zip_path)
    print(f"Deleted {zip_path}.")


if __name__ == "__main__":
    fire.Fire(download_and_extract)
