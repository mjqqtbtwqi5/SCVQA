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
    _VSFA_GRU = "VSFA_GRU"

    MODELS = [_LSTM, _TRANSFORMER, _VSFA_GRU]

    _MSELoss = "MSELoss"
    _L1Loss = "L1Loss"

    _CSCVQ = "CSCVQ"
    _SCVD = "SCVD"
    DATABASES = [_CSCVQ, _SCVD]

    _ResNet50 = "ResNet50"
    CNN_EXTRACTIONS = [_ResNet50]

    for model in MODELS:
        for database in DATABASES:
            for cnn_extraction in CNN_EXTRACTIONS:

                loss_fn = _L1Loss if model == _VSFA_GRU else _MSELoss

                model_dir = Path(f"model/{model}/{database}/{cnn_extraction}")
                model_report_dir = Path(
                    f"model/{model}/{database}/{cnn_extraction}/report"
                )
                model_history_csv = Path(
                    f"model/{model}/{database}/{cnn_extraction}/history.csv"
                )

                reports = list()
                if not os.path.exists(model_dir):
                    print(f"Model result not exists in {model_dir}/")
                    sys.exit()
                else:
                    directories = list(pd.read_csv(str(model_history_csv)).DIR.values)
                    result_df = pd.DataFrame()
                    for dir in directories:
                        model_result_csv = str(model_dir / dir / "result.csv")
                        result_df = pd.concat(
                            [result_df, pd.read_csv(model_result_csv)]
                        )

                    model_prediction_csv = str(
                        model_dir / directories[-1] / "prediction.csv"
                    )
                    prediction_df = pd.read_csv(model_prediction_csv)

                    report = Report(
                        result_df,
                        prediction_df,
                        str(model_report_dir),
                        str(model_report_dir / "loss.png"),
                        str(model_report_dir / "RMSE.png"),
                        str(model_report_dir / "PLCC.png"),
                        str(model_report_dir / "SROCC.png"),
                        str(model_report_dir / "prediction.png"),
                    )

                PdfGenerator(report, loss_fn).generate()
