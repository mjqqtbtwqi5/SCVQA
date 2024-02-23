import os
from pandas import DataFrame
import matplotlib.pyplot as plt
from pathlib import Path


class Report:
    def __init__(
        self,
        result_df: DataFrame,
        report_dir: str,
        loss_img_file: str,
        RMSE_img_file: str,
        PLCC_img_file: str,
        SROCC_img_file: str,
    ) -> None:
        self.result_df = result_df
        self.report_dir = report_dir
        self.loss_img_file = loss_img_file
        self.RMSE_img_file = RMSE_img_file
        self.PLCC_img_file = PLCC_img_file
        self.SROCC_img_file = SROCC_img_file


class PdfGenerator:
    def __init__(self, report: Report) -> None:
        self.report = report

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
        plt.close()

    def generate(self) -> None:
        if self.report:
            if os.path.exists(self.report.report_dir):
                print(f"Report already exist at: {self.report.report_dir}")
            else:
                Path(self.report.report_dir).mkdir(parents=True, exist_ok=True)
                self.plot_curves(
                    self.report.result_df,
                    "MSE Loss",
                    "train_MSELoss",
                    "test_MSELoss",
                    self.report.loss_img_file,
                )
                self.plot_curves(
                    self.report.result_df,
                    "RMSE",
                    "train_RMSE",
                    "test_RMSE",
                    self.report.RMSE_img_file,
                )
                self.plot_curves(
                    self.report.result_df,
                    "PLCC",
                    "train_PLCC",
                    "test_PLCC",
                    self.report.PLCC_img_file,
                )
                self.plot_curves(
                    self.report.result_df,
                    "SROCC",
                    "train_SROCC",
                    "test_SROCC",
                    self.report.SROCC_img_file,
                )
                print(f"Report created at: {self.report.report_dir}")
        else:
            print("No report to be generated.")
