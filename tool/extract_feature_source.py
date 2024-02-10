import os
import zipfile
from pathlib import Path
import sys
import shutil
import numpy as np

if __name__ == "__main__":
    DATABASE = "SCVD"
    CNN_EXTRACTION = "ResNet50"

    current_working_directory = os.getcwd()
    PREFIX = "../" if current_working_directory.endswith("tool") else ""

    FEATURE_DIR = Path(f"{PREFIX}feature/{DATABASE}/{CNN_EXTRACTION}/")

    FEATURE_SOURCE_DIR = Path(f"{PREFIX}source/{DATABASE}/SCVD_feature.zip")
    FEATURE_EXTRACT_DIR = Path(f"{PREFIX}feature/{DATABASE}/")

    if os.path.exists(FEATURE_DIR):
        print(f"{DATABASE} Video feature {CNN_EXTRACTION} exists in {FEATURE_DIR}")
    else:
        with zipfile.ZipFile(FEATURE_SOURCE_DIR, "r") as zip_ref:
            print(
                f"Unzipping {DATABASE} features {CNN_EXTRACTION}: {FEATURE_SOURCE_DIR}"
            )
            zip_ref.extractall(FEATURE_EXTRACT_DIR)
        Path(FEATURE_EXTRACT_DIR / "all").rename(FEATURE_DIR)

        for dir_path, dir_names, file_names in os.walk(FEATURE_DIR):
            for file_name in file_names:
                if file_name.endswith(".npy"):
                    idx = None
                    org_path = os.path.join(dir_path, file_name)
                    new_path = None

                    if file_name.endswith("_score.npy"):
                        idx = file_name.replace("_score.npy", "")
                        new_path = os.path.join(f"{dir_path}/{idx}", "mos.npy")
                    else:
                        idx = file_name.replace(".npy", "")
                        new_path = os.path.join(f"{dir_path}/{idx}", "feature.npy")
                    if not os.path.exists(FEATURE_DIR / idx):
                        Path(FEATURE_DIR / idx).mkdir(parents=True, exist_ok=True)
                    shutil.move(org_path, new_path)
        print(f"{DATABASE} Video feature {CNN_EXTRACTION} extracted in {FEATURE_DIR}")
