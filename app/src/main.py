import os
import glob
import shutil
from src import loop


def main(req=None):

    params_public = {
        "root_path_s3": "",
        "root_path_local": "",
        "aws_access_key": "",
        "aws_secret_key": "",
    }

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/files/Eudora-1a7d26865c6e.json"
    os.environ["FONT_FILE"] = "/app/files/arial.ttf"

    pdf_files = glob.glob("/app/files/pdfs/*.pdf")

    for f in pdf_files:
        print("Extracting: {}".format(f))

        main_path = "/app/files/output/" + f.split("/")[-1].split(".")[0]

        if not os.path.isdir(main_path):
            print("   ... creating main path!")
            os.mkdir(main_path)

        params_private = {
            "path_pdf_full": "{}/03_pdf_201805".format(main_path),
            "path_pdf_clean": "{}/02_pdf_clean".format(main_path),
            "path_pages_full": "{}/11_pages_full".format(main_path),
            "path_pages_clean": "{}/12_pages_clean".format(main_path),
            "path_pages_ann_f": "{}/21_pages_ann_full".format(main_path),
            "path_pages_ann_c": "{}/22_pages_ann_clean".format(main_path),
            "path_pages_ann_g": "{}/23_pages_ann_grouped_text".format(main_path),
            "path_pages_ann_l": "{}/24_pages_ann_links/".format(main_path),
            "path_metadata": "{}/50_metadata".format(main_path),
            "google_credentials_file": os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            "log_path": "{}/60_log".format(main_path),
        }

        # Create paths if they do not exist
        for v in [v for k, v, in params_private.items() if "path" in k]:
            if not os.path.isdir(v):
                os.mkdir(v)

        # Copy file
        shutil.copy(f, params_private["path_pdf_full"])

        loop.loop(**params_private)

        print("... done : {}!".format(f))


if __name__ == "__main__":
    main()