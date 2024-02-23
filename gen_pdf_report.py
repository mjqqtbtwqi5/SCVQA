import os
from pathlib import Path
import pandas as pd

import sys

sys.path.append("./util")
from report_generator import Report, PdfGenerator

if __name__ == "__main__":
    print("=" * 50)
    _CSCVQ = "CSCVQ"
    _SCVD = "SCVD"
    DATABASE = _SCVD

    _LSTM = "LSTM"
    _TRANSFORMER = "Transformer"
    MODEL = _LSTM

    CNN_EXTRACTION = "ResNet50"

    MODEL_DIR = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}")
    MODEL_REPORT_DIR = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}/report")
    MODEL_HISTORY_CSV = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}/history.csv")

    # ==================================================
    # 1. Data preparation
    # ==================================================
    print("=" * 50)
    reports = list()
    if not os.path.exists(MODEL_DIR):
        print(f"Model result not exists in {MODEL_DIR}/")
        sys.exit()
    else:
        directories = list(pd.read_csv(str(MODEL_HISTORY_CSV)).DIR.values)
        result_df = pd.DataFrame()
        for dir in directories:
            model_result_csv = str(MODEL_DIR / dir / "result.csv")
            result_df = pd.concat([result_df, pd.read_csv(model_result_csv)])

        report = Report(
            result_df,
            str(MODEL_REPORT_DIR),
            str(MODEL_REPORT_DIR / "loss.png"),
            str(MODEL_REPORT_DIR / "RMSE.png"),
            str(MODEL_REPORT_DIR / "PLCC.png"),
            str(MODEL_REPORT_DIR / "SROCC.png"),
        )

    PdfGenerator(report).generate()
