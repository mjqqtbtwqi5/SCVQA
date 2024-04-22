import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys

sys.path.append("./util")
from report_generator import Report, PdfGenerator


def plot_time_complexity(
    database,
    cnn_extraction,
    model_list,
    second_list,
):
    save_path = f"model/{database}_{cnn_extraction}_time_complexity.png"

    plt.figure()
    plt.title(f"Time Complexity of {database} ({cnn_extraction})")
    for i in range(len(model_list)):
        # start_time = second_list[i][0]
        # minute_results = (second_list[i] - start_time) / 60
        minute_results = second_list[i] / 60
        epochs = range(len(minute_results))
        plt.plot(epochs, minute_results, label=model_list[i])
    plt.xlabel("Epochs")
    plt.ylabel("Minutes")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


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

    _ResNet18 = "ResNet18"
    _ResNet34 = "ResNet34"
    _ResNet50 = "ResNet50"
    _ResNet101 = "ResNet101"
    CNN_EXTRACTIONS = [
        _ResNet18,
        _ResNet34,
        _ResNet50,
        _ResNet101,
    ]

    for database in DATABASES:
        for cnn_extraction in CNN_EXTRACTIONS:
            time_complexity_compare_database = database
            time_complexity_compare_cnn_extraction = cnn_extraction
            time_complexity_compare_model_list = []
            time_complexity_compare_second_list = []
            for model in MODELS:

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

                    time_complexity_compare_model_list.append(model)
                    time_complexity_compare_second_list.append(result_df["second"])

                PdfGenerator(report, loss_fn).generate()
            # plot_time_complexity(
            #     time_complexity_compare_database,
            #     time_complexity_compare_cnn_extraction,
            #     time_complexity_compare_model_list,
            #     time_complexity_compare_second_list,
            # )
