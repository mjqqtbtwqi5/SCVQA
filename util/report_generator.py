import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class Report:
    def __init__(
        self,
        result_file: str,
        report_dir: str,
        report_pdf_file: str,
        loss_img_file: str,
        RMSE_img_file: str,
        PCC_img_file: str,
        SROCC_img_file: str,
    ) -> None:
        self.result_file = result_file
        self.report_dir = report_dir
        self.report_pdf_file = report_pdf_file
        self.loss_img_file = loss_img_file
        self.RMSE_img_file = RMSE_img_file
        self.PCC_img_file = PCC_img_file
        self.SROCC_img_file = SROCC_img_file


class PdfGenerator:
    def __init__(self, reports: list[Report]) -> None:
        self.reports = reports

    def plot_curves(self, results, title, train_column, test_column, save_path):
        train_results = results[train_column]
        test_results = results[test_column]
        epochs = range(len(results[train_column]))

        plt.figure()

        plt.plot(epochs, train_results, label=train_column)
        plt.plot(epochs, test_results, label=test_column)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(save_path)

    def generate(self) -> None:
        if len(self.reports) > 0:
            for report in self.reports:
                if os.path.exists(report.report_dir):
                    print(f"Report already exist at: {report.report_dir}")
                else:
                    Path(report.report_dir).mkdir(parents=True, exist_ok=True)
                    results = pd.read_csv(report.result_file)
                    self.plot_curves(
                        results,
                        "MSE Loss",
                        "train_MSELoss",
                        "test_MSELoss",
                        report.loss_img_file,
                    )
                    self.plot_curves(
                        results,
                        "RMSE",
                        "train_RMSE",
                        "test_RMSE",
                        report.RMSE_img_file,
                    )
                    self.plot_curves(
                        results,
                        "PCC",
                        "train_PCC",
                        "test_PCC",
                        report.PCC_img_file,
                    )
                    self.plot_curves(
                        results,
                        "SROCC",
                        "train_SROCC",
                        "test_SROCC",
                        report.SROCC_img_file,
                    )

                    print(f"Report created at: {report.report_dir}")
        else:
            print("No report(s) to be generated.")
