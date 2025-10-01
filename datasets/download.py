#%%
import subprocess
import typer
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%

def main(
    dataset_id: str = typer.Option(..., "--dataset-id", "-id", help="The dataset ID"),
    input_dir: Path = typer.Option(Path("data/input/openneuro"), "--input-dir", "-i", help="The input directory")
):
    input_dir = input_dir / dataset_id
    logging.info(f"Starting download to {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(["aws", "s3", "sync", "--no-sign-request", f"s3://openneuro.org/{dataset_id}", input_dir.absolute()], check=True)
        logging.info(f"Download completed successfully to {input_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Download failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

#%%
if __name__ == "__main__":
    typer.run(main)
