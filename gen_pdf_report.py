import os
from pathlib import Path
import pandas as pd

import sys

sys.path.append("./util")
from report_generator import Report, PdfGenerator

if __name__ == "__main__":
    print("=" * 50)

    _LSTM = "LSTM"
    _TRANSFORMER = "Transformer"
    MODEL = _TRANSFORMER

    _CSCVQ = "CSCVQ"
    _SCVD = "SCVD"
    DATABASE = _SCVD

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

        model_prediction_csv = str(MODEL_DIR / directories[-1] / "prediction.csv")
        prediction_df = pd.read_csv(model_prediction_csv)

        report = Report(
            result_df,
            prediction_df,
            str(MODEL_REPORT_DIR),
            str(MODEL_REPORT_DIR / "loss.png"),
            str(MODEL_REPORT_DIR / "RMSE.png"),
            str(MODEL_REPORT_DIR / "PLCC.png"),
            str(MODEL_REPORT_DIR / "SROCC.png"),
            str(MODEL_REPORT_DIR / "prediction.png"),
        )

    PdfGenerator(report).generate()
