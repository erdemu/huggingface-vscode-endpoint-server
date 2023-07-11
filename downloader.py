import os

import requests
import typer
from tqdm import tqdm

app = typer.Typer()

def download_file(url, path):
    '''
    This function will download a single file from a URL and save it to a specified path.

    Args:
        url (str): The URL of the file to download
        path (str): The path to save the downloaded file

    Returns:
        None
    '''
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kbyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()

def download_model(model_name, destination_folder="models"):
    '''
    This function will download a model into: models/{model_name}/{file}

    Args:
        model_name (str): The name of the model to download
        destination_folder (str): The folder to download the model into (default: models)

    Returns:
        None
    '''
    
    # Define the base URL and headers for the Hugging Face API
    base_url = f"https://huggingface.co/{model_name}/resolve/main"
    headers = {"User-Agent": "Hugging Face Python"}

    # Send a GET request to the Hugging Face API to get a list of all files
    response = requests.get(f"https://huggingface.co/api/models/{model_name}", headers=headers)
    response.raise_for_status()
    
    # Extract the list of files from the response JSON
    files_to_download = [file["rfilename"] for file in response.json()["siblings"]]

    # Ensure the directory exists
    os.makedirs(f"{destination_folder}/{model_name}", exist_ok=True)

    # Download each file
    for file in files_to_download:
        print(f"Downloading {file}...")
        download_file(f"{base_url}/{file}", f"{destination_folder}/{model_name}/{file}")

@app.command()
def download(model_name: str = "TheBloke/WizardCoder-15B-1.0-GPTQ"):
    download_model(model_name)

if __name__ == "__main__":
    app()
