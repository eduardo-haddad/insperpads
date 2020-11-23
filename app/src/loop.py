import glob
from src.functions import *


def loop(**params_private):

    pdf_files = glob.glob(params_private["path_pdf_full"] + "/*.pdf")

    # TODO: extend the program to process multiple pdf files
    # For now, working only with the first of the list
    pdf_file = pdf_files[0].split("/")[-1]

    # Extract text and products from PDF full¶

    # Create Logger
    logger, _ = create_logger("AppGuia_v0.7", log_path=params_private["log_path"])
    logger.info("STARTING: {}".format(pdf_file))

    # Extract text from PDF
    df_txt_original, df_txt, df, page_map = pdf2df(
        params_private["path_pdf_full"], pdf_file, logger=logger
    )

    # Eliminate text from PDF full and create PDF clean
    # Clean PDFs from text
    clean_pdf_files(pdf_file, params_private, logger)

    ## Convert pdf files to images

    # Convert full PDF to PNG
    pdf2image(
        pdf_file,
        params_private["path_pdf_full"],
        params_private["path_pages_full"],
        logger=logger,
    )

    # Convert clean PDF to PNG
    pdf2image(
        pdf_file,
        params_private["path_pdf_clean"],
        params_private["path_pages_clean"],
        logger=logger,
    )

    # Detect objects
    img_files = glob.glob(params_private["path_pages_full"] + "/*.png")
    df_objs = detect_objects(img_files, params_private, logger)

    # Detect Faces

    logger.info("Starting face detection...")

    img_files = sorted(glob.glob(params_private["path_pages_full"] + "/*.png"))

    df_faces = detect_faces(img_files, params_private, logger)

    df_faces.head()

    # Join structured data...
    df_guia = join_objects([df_objs, df_faces], df_txt_original, logger)

    # Annotate full guia
    annotate_images(
        img_files, df_guia, annotated_path=params_private["path_pages_ann_f"], logger=logger
    )

    # Save metadata
    save_metadata(df_guia, params_private, "01_metdata_guia.pkl", logger)

    # Load metadata
    df_guia = load_metadata(params_private, "01_metdata_guia.pkl", logger)

    # Feature engineering
    df_guia = create_obj_id(df_guia, logger)

    ## Find  índice & abre
    try:
        df_guia = find_indice_abre(df_guia, logger)
    except:
        logger.warning("Could not identify: abre.")

    # Cluster text
    df_guia = cluster_text(df_guia, logger, join=False)

    # Annotate images with grouped text
    img_files = sorted(glob.glob(params_private["path_pages_full"] + "/*.png"))

    annotate_images(
        img_files,
        df_guia.loc[(df_guia["type"] == "object") | (df_guia["subtype"] == "text_sku")],
        annotated_path=params_private["path_pages_ann_g"],
        logger=logger,
    )

    ## Connect text and images

    # Create links between txt and images and create Id
    df_guia, df_links = link_txt_fig(df_guia, logger)

    img_files = glob.glob(params_private["path_pages_ann_g"] + "/*.png")

    logger.info("Creating annotated images...")

    annotate_connections(
        img_files, df_guia, df_links, params_private["path_pages_ann_l"], logger=logger
    )

    df_guia = join_guia_links(df_guia, df_links, logger)

    save_metadata(df_guia, params_private, "02_metdata_guia.pkl", logger)

    logger.info("DONE: {}".format(pdf_file))