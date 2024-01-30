import os
from pathlib import Path

import sys

sys.path.append("./util")
from report_generator import Report, PdfGenerator

if __name__ == "__main__":
    print("=" * 50)
    RNN = ["LSTM", "Transformer"]
    MODEL_IMPL = RNN[0]

    MODEL_DIR = Path(f"model/{MODEL_IMPL}/")

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
            info_file = f"{model_result_dir}/info.csv"
            result_file = f"{model_result_dir}/result.csv"

            report_dir = f"{model_result_dir}/report/"
            report_pdf_file = f"{model_result_dir}/report/report.pdf"
            loss_img_file = f"{model_result_dir}/report/loss.png"
            MSE_img_file = f"{model_result_dir}/report/MSE.png"
            PCC_img_file = f"{model_result_dir}/report/PCC.png"
            SROCC_img_file = f"{model_result_dir}/report/SROCC.png"

            reports.append(
                Report(
                    info_file,
                    result_file,
                    report_dir,
                    report_pdf_file,
                    loss_img_file,
                    MSE_img_file,
                    PCC_img_file,
                    SROCC_img_file,
                )
            )

    PdfGenerator(reports).generate()
