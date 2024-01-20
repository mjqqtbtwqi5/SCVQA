import os
import sys
import numpy as np
from pathlib import Path

import torch



if __name__ == '__main__':

    DATABASE = "CSCVQ"
    CNN_MODULE = "ResNet50"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8


    FEATURE_DIR = Path(f"feature/{DATABASE}/{CNN_MODULE}/")
    if not os.path.exists(FEATURE_DIR):
        print(f"No features found in {FEATURE_DIR}")
        sys.exit()

    feature = np.load(f"{FEATURE_DIR}/0/feature.npy")
    feature = torch.from_numpy(feature).to(device=DEVICE)

    mos = np.load(f"{FEATURE_DIR}/0/mos.npy")
    mos = torch.from_numpy(mos).to(device=DEVICE).unsqueeze(dim=0)

    print(feature.shape)
    print(mos[0])