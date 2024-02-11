import os
from pathlib import Path

import sys

sys.path.append("./util")
from report_generator import Report, PdfGenerator

if __name__ == "__main__":
    print("=" * 50)
    DATABASE = "CSCVQ"
    CNN_EXTRACTION = "ResNet50"

    _LSTM = "LSTM"
    _TRANSFORMER = "Transformer"
    MODEL = _TRANSFORMER

    MODEL_DIR = Path(f"model/{MODEL}/{DATABASE}/{CNN_EXTRACTION}")

    # ==================================================
    # 1. Data preparation
    # ==================================================
    print("=" * 50)
    reports = list()
    if not os.path.exists(MODEL_DIR):
        print(f"Model result not exists in {MODEL_DIR}/")
        sys.exit()
    else:
        model_result_dir_list = [f.path for f in os.scandir(MODEL_DIR) if f.is_dir()]

        for model_result_dir in model_result_dir_list:
            result_file = f"{model_result_dir}/result.csv"

            report_dir = f"{model_result_dir}/report/"
            report_pdf_file = f"{model_result_dir}/report/report.pdf"
            loss_img_file = f"{model_result_dir}/report/loss.png"
            RMSE_img_file = f"{model_result_dir}/report/RMSE.png"
            PLCC_img_file = f"{model_result_dir}/report/PLCC.png"
            SROCC_img_file = f"{model_result_dir}/report/SROCC.png"

            reports.append(
                Report(
                    result_file,
                    report_dir,
                    report_pdf_file,
                    loss_img_file,
                    RMSE_img_file,
                    PLCC_img_file,
                    SROCC_img_file,
                )
            )

    PdfGenerator(reports).generate()
