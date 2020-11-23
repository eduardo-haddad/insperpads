from google.cloud import vision

import logging

def create_logger(name="App Guia", log_path="./log", level=logging.DEBUG):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    import os
    from datetime import datetime

    # Create log_path if it does not exist
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Create unique filename
    log_file = datetime.now().strftime(
        "%Y%m%d_%H%M%S_" + name.replace(" ", "_") + ".log"
    )

    # Create Logger
    logger = logging.getLogger(name)

    # Set level
    logger.setLevel(level)

    # Create handlers
    fh = logging.FileHandler("/".join([log_path, log_file]))
    ch = logging.StreamHandler()

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    #logger.addHandler(ch)

    return logger, "/".join([log_path, log_file])

def save_img_txt(IMG_FILE, df_guia, save_file):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    im = Image.open(IMG_FILE)

    df_guia["H"] = im.height
    df_guia["W"] = im.width

    img = ImageDraw.Draw(im)

    for txt in objs:
        img.rectangle(
            [obj[0][0], obj[0][1], obj[2][0], obj[2][1]], outline="lime", width=4
        )

    for obj in objs:
        img.rectangle(
            [obj[0][0], obj[0][1], obj[2][0], obj[2][1]], outline="lime", width=4
        )

    im.save(save_file)

def save_img_objs(IMG_FILE, objects, save_file, color="lime"):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    from PIL import Image, ImageDraw
    
    im = Image.open(IMG_FILE)

    H = im.height
    W = im.width

    objs = list()

    for object_ in objects:
        bounding_box = list()

        for vertex in object_.bounding_poly.normalized_vertices:
            bounding_box.append((int(W * vertex.x), int(H * vertex.y)))

        objs.append(bounding_box)

    img = ImageDraw.Draw(im)

    for obj in objs:
        img.rectangle(
            [obj[0][0], obj[0][1], obj[2][0], obj[2][1]], outline=color, width=4
        )

    im.save(save_file)

def show_img_objs(IMG_FILE, objects):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    plt.figure(figsize=(14, 20))

    im = Image.open(IMG_FILE)

    imshow(im)

    H = im.height
    W = im.width

    objs = list()

    for object_ in objects:
        bounding_box = list()

        for vertex in object_.bounding_poly.normalized_vertices:
            bounding_box.append((int(W * vertex.x), int(H * vertex.y)))

        objs.append(bounding_box)

    ax = plt.gca()

    img = ImageDraw.Draw(im)

    for obj in objs:
        # Create a Rectangle patch
        w = int(obj[2][0] - obj[0][0])
        h = int(obj[2][1] - obj[0][1])

        x0 = obj[0][0]
        y0 = obj[0][1]

        rect = patches.Rectangle(
            (x0, y0), w, h, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

def gc_detect_faces(IMG_FILE):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    client = vision.ImageAnnotatorClient()

    with open(IMG_FILE, "rb") as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    
    response = client.face_detection(image=image)

    return response

def gc_detect_objects(IMG_FILE):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    client = vision.ImageAnnotatorClient()

    with open(IMG_FILE, "rb") as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    return objects

def gc_detect_labels(IMG_FILE):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    client = vision.ImageAnnotatorClient()

    with open(IMG_FILE, "rb") as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    labels = client.label_detection(image=image)

    return labels

def faces2df(list_faces):
    list_faces_2 = list()

    for c, (f, s) in enumerate(list_faces):
        if f.ListFields() != []:
            W = s[0]
            H = s[1]
            for face in f.face_annotations:
                list_faces_2.append(
                    (c, [(v.x / W, v.y / H) for v in face.bounding_poly.vertices])
                )

    list_faces_3 = [(c, p[0][0], p[0][1], p[2][0], p[2][1]) for (c, p) in list_faces_2]

    df_faces = pd.DataFrame(
        list_faces_3, columns=["page", "pos_x0", "pos_y0", "pos_x1", "pos_y1"]
    )

    df_faces["type"] = "face"

    df_faces["pos_x"] = (df_faces["pos_x1"] - df_faces["pos_x0"]) / 2
    df_faces["pos_y"] = (df_faces["pos_y1"] - df_faces["pos_y0"]) / 2

    return df_faces

def flatten_list(l, output=False):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    if output == False:
        output = list()

    for i in l:
        if type(i) == list:
            flatten_list(i, output)
        else:
            output.append(i)

    return output

def process_obj(obj, k, pg, ignore_spread):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
        
    import pdfminer
    
    fig_list = list()

    pg_size_x = pg.mediabox[-2]
    pg_size_y = pg.mediabox[-1]

    pg_height = pg_size_y

    if ignore_spread or ((pg_size_x / pg_size_y) < 1):
        spread = False
        pg_center = pg_size_x
        pg_width = pg_size_x
    else:
        spread = True
        pg_center = pg_size_x / 2
        pg_width = pg_size_x / 2

    # if it's a textbox, print text and location
    if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
        if spread and (obj.bbox[0] > pg_center):
            pg_side = 1
        else:
            pg_side = 0

        ret = {
            "type": "text",
            "page_orig": k,
            "spread": spread,
            "side": pg_side,
            "pos_x0": (obj.bbox[0] - pg_side * pg_center) / pg_width,
            "pos_y0": (obj.bbox[1]) / pg_height,
            "pos_x1": (obj.bbox[2] - pg_side * pg_center) / pg_width,
            "pos_y1": (obj.bbox[3]) / pg_height,
            "pos_x": ((obj.bbox[0] - pg_side * pg_center) + obj.width / 2) / pg_width,
            "pos_y": (obj.bbox[1] + obj.height / 2) / pg_height,
            "width": (obj.width) / pg_width,
            "height": (obj.height) / pg_height,
            "contents": obj.get_text().replace("\n", " ").strip(),
        }

        return ret

    # if image
    elif isinstance(obj, pdfminer.layout.LTImage):
        if obj.bbox[0] < pg_center:
            pg_side = 0
        else:
            pg_side = 1

        ret = {
            "type": "image",
            "page_orig": k,
            "spread": spread,
            "side": pg_side,
            "name": obj.name,
            "pos_x0": (obj.bbox[0] - pg_side * pg_center) / pg_width,
            "pos_y0": (obj.bbox[1]) / pg_height,
            "pos_x1": (obj.bbox[2] - pg_side * pg_center) / pg_width,
            "pos_y1": (obj.bbox[3]) / pg_height,
            "pos_x": ((obj.bbox[0] - pg_side * pg_center) + obj.width / 2) / pg_width,
            "pos_y": (obj.bbox[1] + obj.height / 2) / pg_height,
            "width": (obj.width) / pg_width,
            "height": (obj.height) / pg_height,
            "contents": obj.name,
        }

        return ret

    # if it's a container, recurse
    elif isinstance(obj, pdfminer.layout.LTFigure):

        for ob in obj._objs:
            fig_list.append(process_obj(ob, k, pg, ignore_spread))

        return fig_list

    else:
        ret = {"type": "other", "page_orig": k, "spread": spread, "contents": obj}
        return ret

import pandas as pd
import numpy as np

def pdf2df(pdf_path, pdf_file_name, logger):

    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfpage import PDFTextExtractionNotAllowed
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.pdfinterp import PDFPageInterpreter
    from pdfminer.pdfdevice import PDFDevice
    from pdfminer.layout import LAParams
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.pdfinterp import resolve1
    
    # Open PDF File
    logger.info('Extracting text objects from file {}...'.format(pdf_file_name))
    pdf_file = open(pdf_path + '/' + pdf_file_name, 'rb')

    # Create a PDF parser object associated with the file object.
    logger.info('Creating pdf objects!')
    parser = PDFParser(pdf_file)

    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    document = PDFDocument(parser)

    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()

    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)

    # Set parameters for analysis.
    laparams = LAParams()

    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # get number of pages
    page_num = resolve1(document.catalog['Pages'])['Count']

    # create pages
    pages = list(PDFPage.create_pages(document))

    list_objs = list()

    logger.info('Processing pdf pages!')
    for k, pg in zip(range(0,len(pages)), pages):
        logger.info('Processing page {}...'.format(k))

        interpreter.process_page(pg)

        layout = device.get_result()

        pg_objs = layout._objs
       
        ignore_spread = (k == 0) or (k == (len(pages) - 1))
        
        list_objs += [process_obj(ob, k, pg, ignore_spread) for ob in pg_objs]

    list_objs2 = flatten_list(list_objs)

    logger.info('Extracting text and features!')
    
    df_txt = pd.DataFrame([obj for obj in list_objs2 if (obj['type'] == 'text') & (obj['contents'] != '')])
    
    pg_map = pd.DataFrame([(o['page_orig'], o['side']) for o in list_objs2 if 'other' not in o['type']], columns = ['page_spread', 'side'])\
                .drop_duplicates()\
                .sort_values(by = ['page_spread', 'side'])\
                .reset_index(drop = True)\
                .reset_index().rename({'index' : 'page'}, axis = 1)
    
    df_txt = df_txt.merge(pg_map, left_on = ['page_orig', 'side'], right_on = ['page_spread', 'side'])
    
    df_txt_original = df_txt.copy()
    
    df_txt = df_txt.groupby('page')['contents'].apply(lambda x: '\n'.join(x)).reset_index()
    
    df_txt['skus'] = df_txt.contents.str.findall(r'\d{5,6}')

    df_txt['skus'] = df_txt['skus'].apply(lambda x: list(set(x)))
    
    df_txt['indice'] = df_txt.contents.str.findall(r'Indice').apply(lambda x: x != [])

    #pg_indice = df_txt.loc[df_txt.indice, 'page'][0].values

    df = df_txt.loc[df_txt['skus'].apply(lambda x: x != []), ['page', 'skus']]

    df['page_density'] = df.skus.str.len()

    df = df.explode(column = 'skus')

    df1 = pd.concat([df.groupby('skus')['page_density'].apply(lambda x: np.array(x)).reset_index(),
            df.groupby('skus')['page'].apply(lambda x: np.array(x)).reset_index(drop = True)], axis = 1)

    df1['sku_density'] = df1['page'].str.len()

    df1['page_norm'] = df1['page'] / df1['page'].apply(lambda x: max(x)).max()

    # TODO: model capa and abre
    # df1['abre'] = df1['page'].apply(lambda x: (x < pg_indice).any())
    # df1['capa_01'] = df1['page'].apply(lambda x: (x == 0).any())
    df1['capa_02'] = df1['page'].apply(lambda x: (x == (df1['page'].apply(lambda x : max(x)).max())).any())

    df1['meio'] = df1.page_norm.apply(lambda x: ((x > 0.45) & (x < 0.55)).any())
    
    # Invert coordinates
    df_txt_original.loc[df_txt_original.type == 'text', 'pos_y0'] = 1 - df_txt_original.loc[df_txt_original.type == 'text', 'pos_y0']
    df_txt_original.loc[df_txt_original.type == 'text', 'pos_y1'] = 1 - df_txt_original.loc[df_txt_original.type == 'text', 'pos_y1']
    df_txt_original.loc[df_txt_original.type == 'text', 'pos_y']  = 1 - df_txt_original.loc[df_txt_original.type == 'text', 'pos_y']
        
    logger.info('Done!\n')
    
    return df_txt_original, df_txt, df1, pg_map

def pdf2image(pdf_file, pdf_path, target_path, logger):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    from pdf2image import convert_from_path, convert_from_bytes
    
    logger.info("Converting file {} to PNG...".format(pdf_path + '/' + pdf_file))
    
    imgs = convert_from_bytes(open(pdf_path + '/' + pdf_file, "rb").read())

    imgs_ar = [im.height / im.width for im in imgs]

    imgs_splitted = list()

    for ar, img in zip(imgs_ar, imgs):
        w = int(img.width)
        h = int(img.height)

        if ar < 1:
            imgs_splitted.append(img.crop((0, 0, int(w / 2), h)))
            imgs_splitted.append(img.crop((int(w / 2), 0, w, h)))
        else:
            imgs_splitted.append(img)

    for c, im in enumerate(imgs_splitted):
        logger.info("{}/page_{:03d}.png saved!".format(target_path, c))
        im.save("{}/page_{:03d}.png".format(target_path, c))
        
    logger.info("Done!")
    
    return 0

def convert_objs(list_objs, img_files):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    from PIL import Image
    
    new_list = list()

    for l_objs, img_file in zip(list_objs, img_files):
        im = Image.open(img_file)

        for obj in l_objs:
            new_list.append(
                (
                    obj.name,
                    obj.score,
                    [((v.x), (v.y)) for v in obj.bounding_poly.normalized_vertices],
                    img_file,
                )
            )

    df1 = pd.DataFrame(new_list, columns=["contents", "score", "bbox", "file"])

    df2 = pd.concat(
        [
            df1,
            pd.DataFrame(df1["bbox"].values.tolist(), columns=["bl", "br", "ur", "ul"]),
        ],
        axis=1,
    )

    df2["pos_x0"] = df2.bl.map(lambda x: x[0])
    df2["pos_y0"] = df2.bl.map(lambda y: y[1])

    df2["pos_x1"] = df2.ur.map(lambda x: x[0])
    df2["pos_y1"] = df2.ur.map(lambda y: y[1])

    df2["pos_x"] = (df2["pos_x0"] + df2["pos_x1"]) / 2
    df2["pos_y"] = (df2["pos_y0"] + df2["pos_y1"]) / 2

    df2["width"] = df2["pos_x1"] - df2["pos_x0"]
    df2["height"] = df2["pos_y1"] - df2["pos_y0"]

    df2["page"] = df2["file"].str.extract(r"_(\d+)\.png$").astype(int)

    df2["type"] = "object"

    return df2[
        [
            "type",
            "contents",
            "page",
            "pos_x0",
            "pos_y0",
            "pos_x1",
            "pos_y1",
            "pos_x",
            "pos_y",
            "width",
            "height",
        ]
    ]

def join_txts_objs(df_objs, df_txt_original):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """

    return pd.concat([df_objs, df_txt_original], axis=0, sort=False)

def join_objects(list_df_objs, df_txt_original, logger):
    logger.info("Joining objects and text...")

    df_objs_faces = pd.concat(list_df_objs, sort=False, axis=0)

    df_guia = join_txts_objs(df_objs_faces, df_txt_original)

    try:
        assert len(df_guia) == len(df_guia.drop_duplicates())
    except:
        logger.warning("Dataframe df_guia has duplicate rows!")

    # Create subtype
    df_guia["subtype"] = df_guia["type"]
    
    df_guia.loc[df_guia["type"] == "object", "subtype"] = df_guia.loc[
        df_guia["type"] == "object", "contents"
    ]

    return df_guia

def clean_pdf_files(pdf_file, params_private, logger):

    ### Assert ghostscript exists

    import os

    try:
        assert os.system("command -v gs") == 0
    except:
        print("Ghostscript (gs) not in server found!")

    ### Convert

    # GS Parameters: -dFILTERVECTOR, -dFILTERTEXT, -dFILTERIMAGE
    cmd = "gs -o {} -sDEVICE=pdfwrite -dFILTERTEXT {}".format(
        params_private["path_pdf_clean"] + "/" + pdf_file,
        params_private["path_pdf_full"] + "/" + pdf_file,
    )

    logger.info("Filtering text from PDF with GhostScript...")

    try:
        assert os.system(cmd) == 0
        logger.info("Done!")
    except:
        msg = "GhostScript command could not be executed!"
        logger.error(msg)
        raise EnvironmentError(msg)

    return 0

def detect_objects(img_files, params_private, logger, annotate=True):
    img_files = sorted(img_files)

    logger.info("Starting object detection...")
    logger.info("There are {} files to be processed!".format(len(img_files)))

    list_objs = list()

    for im in img_files:
        logger.info("Processing {} with Google Cloud".format(im))
        list_objs.append(gc_detect_objects(im))

    logger.info("Done!")

    if annotate:
        for img_objs, img_file in zip(list_objs, img_files):
            logger.info("Annotating {}...".format(img_file))
            save_img_objs(
                img_file,
                img_objs,
                params_private["path_pages_ann_c"] + "/" + img_file.split("/")[-1],
            )

        logger.info("Done!")

    df_objs = convert_objs(list_objs, img_files)

    return df_objs

def detect_faces(img_files, params_private, logger):
    img_files = sorted(img_files)

    logger.info("There are {} pages to be processed.".format(len(img_files)))

    from PIL import Image

    list_faces = list()

    for im in img_files:
        logger.info("Processing {} with Google Cloud.".format(im))
        list_faces.append((gc_detect_faces(im), Image.open(im).size))

    logger.info("Done!")

    logger.info("Structuring faces information...")

    df_faces = faces2df(list_faces)

    logger.info("Done!")

    return df_faces

def annotate_images(img_files, df_guia, annotated_path, logger):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    logger.info("Annotating... {} files to {}.".format(len(img_files), annotated_path))
    
    for page, file in enumerate(sorted(img_files)):

        im = Image.open(file)

        H = im.height
        W = im.width

        img = ImageDraw.Draw(im)

        fnt = ImageFont.truetype(os.environ["FONT_FILE"], size=20)

        df = df_guia.loc[df_guia.page == page].copy()

        for n, row in df.loc[df.type == "object"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="lime",
                width=6,
            )
            # img.text((W * row.pos_x0, H * row.pos_y0), row.contents, font = fnt, fill=(0,255,0,1))

        for n, row in df.loc[df.type == "text"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="red",
                width=4,
            )

        for n, row in df.loc[df.type == "face"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="yellow",
                width=4,
            )

        im.save(annotated_path + "/" + file.split("/")[-1])

        logger.info("Page {} annotated!".format(file.split("/")[-1]))
        
    logger.info("Done!")

# DEPRECATED

def annotate_images_v2(img_files, df_guia, annotated_path):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    import os
    from PIL import ImageDraw, ImageFont
    
    for page, file in enumerate(img_files):

        im = Image.open(file)

        H = im.height
        W = im.width

        img = ImageDraw.Draw(im)

        fnt = ImageFont.truetype(os.environ["FONT_FILE"], size=20)

        df = df_guia.loc[df_guia.page == page].copy()

        for n, row in df.loc[df.type == "object"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="lime",
                width=4,
            )
            # img.text((W * row.pos_x0, H * row.pos_y0), row.contents, font = fnt, fill=(0,255,0,1))

        for n, row in df.loc[df.type == "text"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="red",
                width=4,
            )

            img.line(
                [
                    W * row.pos_x0,
                    H * row.pos_y0,
                    W * row.pos_x0_obj,
                    H * row.pos_y0_obj,
                ],
                fill="yellow",
                width=4,
            )

        for n, row in df.loc[df.type == "face"].iterrows():
            img.rectangle(
                [W * row.pos_x0, H * row.pos_y0, W * row.pos_x1, H * row.pos_y1],
                outline="yellow",
                width=4,
            )

        im.save(annotated_path + file.split("/")[-1])

        print(str(page) + " " + file.split("/")[-1])

def annotate_connections(img_files, df_guia, df_links, annotated_path, logger):
    """The sum of two numbers.
    
    Parameters
    ----------
    IMG_FILE: type
        Description of parameter `x`.
    y:
        Description of parameter `y` (with type not specified)


    Returns
    -------
    err_code : int
        Non-zero value indicates error code, or zero on success.
    err_msg : str or None
        Human readable error message, or None on success.
    """
    from PIL import ImageDraw, ImageFont, Image
    import shutil
    import os

    fnt = ImageFont.truetype(os.environ["FONT_FILE"], size=30)

    for page, file in enumerate(sorted(img_files)):

        df_page = df_links.loc[df_links["page"] == page]

        if df_page.empty:
            shutil.copyfile(file, annotated_path + '/' + file.split('/')[-1])
            continue

        im = Image.open(file)

        H = im.height
        W = im.width

        img = ImageDraw.Draw(im)

        for txt, obj in df_page["links"].values[0]:
            img.line(
                [
                    W * df_guia.loc[df_guia.id == txt, "pos_x"],
                    H * df_guia.loc[df_guia.id == txt, "pos_y"],
                    W * df_guia.loc[df_guia.id == obj, "pos_x"],
                    H * df_guia.loc[df_guia.id == obj, "pos_y"],
                ],
                fill="blue",
                width=4,
            )

            img.text(
                (
                    W * df_guia.loc[df_guia.id == txt, "pos_x"],
                    H * df_guia.loc[df_guia.id == txt, "pos_y"],
                ),
                str(df_guia.loc[df_guia.id == txt, "id"].values[0]),
                font=fnt,
                fill="blue",
            )

            img.text(
                (
                    W * df_guia.loc[df_guia.id == obj, "pos_x"],
                    H * df_guia.loc[df_guia.id == obj, "pos_y"],
                ),
                str(df_guia.loc[df_guia.id == obj, "id"].values[0]),
                font=fnt,
                fill="blue",
            )

        im.save(annotated_path + file.split("/")[-1])

        logger.info("Page {} annotated!".format(file.split("/")[-1]))

def border_distance(p1, p2):

    from sklearn.metrics.pairwise import euclidean_distances

    # order in the vector: x0, x1, y0, y1
    # returns the least distance between borders (the two edges of the border)
    p11 = p1[[0, 2]].reshape(1, -1)
    p12 = p1[[0, 3]].reshape(1, -1)
    p13 = p1[[1, 2]].reshape(1, -1)
    p14 = p1[[1, 3]].reshape(1, -1)

    p1 = np.concatenate([p11, p12, p13, p14])

    p21 = p2[[0, 2]].reshape(1, -1)
    p22 = p2[[0, 3]].reshape(1, -1)
    p23 = p2[[1, 2]].reshape(1, -1)
    p24 = p2[[1, 3]].reshape(1, -1)

    p2 = np.concatenate([p21, p22, p23, p24])

    d1 = euclidean_distances(p1, p2)

    return (np.partition(d1.min(axis=1), 2)[:2] ** 3).sum()

def find_min_distance(mat, idx=[]):

    if mat.shape[0] == 0:
        return idx

    idx = idx + [mat.stack().idxmin()]

    if mat.shape[0] > mat.shape[1]:
        mat = mat.loc[mat.index != idx[-1][0], :]
    else:
        mat = mat.loc[mat.index != idx[-1][0], :]#, mat.columns != idx[-1][1]]

    idx = find_min_distance(mat, idx)
    return idx

# cluster text
def cluster_text_dbscan(df_text):

    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=0.07, min_samples=1)

    # TESTE-EDU
    # df_guia["cluster"] = None
    df_text["cluster"] = None

    for pg_num in sorted(df_text.page.unique()):
        pg = df_text[(df_text["page"] == pg_num)]

        if len(pg) > 0:
            model.fit(pg[["pos_x", "pos_y"]])

            df_text.loc[(df_text["page"] == pg_num), "cluster"] = (
                pg_num * 1000 + model.labels_
            )

    return df_text

# join text clusters
def join_txt_clusters(df_text):
   
    df = df_text.iloc[0].copy()

    df["type"] = "text"
    df["contents"] = ",".join(df_text["contents"])
    
    df["pos_x0"] = df_text['pos_x0'].min()
    df["pos_x1"] = df_text['pos_x1'].max()

    df["pos_y0"] = df_text['pos_y0'].max()
    df['pos_y1'] = df_text['pos_y1'].min()

    df['width'] = df['pos_x1'] - df['pos_x0']
    df['height'] = df['pos_y0'] - df['pos_y1']

    df['pos_x'] = (df['pos_x1'] - df['pos_x0']) / 2 + df['pos_x0']
    df['pos_y'] = (df['pos_y0'] - df['pos_y1']) / 2 + df['pos_y1']
    
    df['skus'] = np.concatenate(df_text["skus"].values).tolist()

    return df

def find_txt_skus(df_text):
    df_text["skus"] = df_text["contents"].str.findall(r"\b\d{5,6}\b")
    df_text["skus"] = df_text["skus"].apply(lambda x: list(set(x)))

    return df_text

def save_metadata(df_guia, params_private, filename, logger):

    logger.info("Saving pickle metadata...")

    df_guia.to_pickle(params_private["path_metadata"] + "/" + filename)

    logger.info("Done!")

    return 0

def load_metadata(params_private, filename, logger):

    logger.info("Reading pickle metadata...")

    df_guia = pd.read_pickle(params_private["path_metadata"] + "/" + filename)

    logger.info("Done!")

    return df_guia

def find_indice_abre(df_guia, logger):

    logger.info("Finding índice & abre...")

    pg_indice = df_guia.loc[(df_guia.loc[:, "contents"].str.find("FSC") == 0)]["page"]

    df_guia["abre"] = False

    df_guia.loc[df_guia["page"] <= pg_indice.values[0], "abre"] = True

    logger.info("Done!")

    return df_guia

def cluster_text(df_guia, logger, join=True):

    # Split text and objects
    df_text = df_guia.loc[df_guia["type"] == "text"].copy()
    df_objs = df_guia.loc[df_guia["type"].isin(["object", "face"])].copy()

    try:
        assert len(df_objs) + len(df_text) == len(df_guia)
    except:
        logger.warning("Len of splitted guia does not match sum of len of parts!")

    # Find sku codes
    df_text_skus = find_txt_skus(df_text)

    # Create subtype for texts with SKU on page
    df_text_skus.loc[
        df_text_skus["skus"].apply(lambda x: x != []), "subtype"
    ] = "text_sku"

    # Define if text should be clustered or not

    # Cluster with DBSCAN
    df_text_cluster = cluster_text_dbscan(df_text_skus)

    if join:
        # Join text clusters
        df_text_grouped = (
            df_text_cluster.groupby("cluster")
            .apply(join_txt_clusters)
            .reset_index(drop=True)
        )
    else:
        df_text_grouped = df_text_skus

    # Reconstruct guia and annotate
    df_guia = pd.concat([df_text_grouped, df_objs], sort=False, axis=0)
    
    return df_guia

def create_obj_id(df_guia, logger):
    logger.info("Creating unique ID for objects...")
    
    if 'id' in df_guia.columns:
        logger.warning("ID column exists! Recreating...")
        df_guia = df_guia.drop(['id'], axis = 1)

    df_guia1 = (
        df_guia.reset_index(drop=True).reset_index().rename({"index": "id"}, axis=1)
    )

    logger.info("Done!")
    
    return df_guia1

def link_txt_fig(df_guia1, logger):

    from sklearn.metrics import pairwise_distances

    # Split text and objects and create unique index
    logger.info("Linking text with figures...")

    # Split text and objects and create unique index
    logger.info("Splitting text and pictures data frame...")

    df_text = df_guia1.loc[df_guia1["subtype"] == "text_sku"].rename(
        {"id": "txt_id"}, axis=1
    )
    df_objs = df_guia1.loc[
        (df_guia1["type"] == "object") & (~df_guia1["subtype"].isin(["Person"]))
    ].rename({"id": "obj_id"}, axis=1)

    ## Calculate distances between text and objects

    logger.info("Calculating distances...")

    coordinates = ["pos_x", "pos_y"]

    dist_list = list()

    # TESTE-EDU
    # for pg in range(max(df_guia.page)):
    for pg in range(max(df_guia1.page)):

        # Split text and objects
        pg_text = df_text.loc[(df_text["page"] == pg), coordinates + ["txt_id"]]
        pg_objs = df_objs.loc[(df_objs["page"] == pg), coordinates + ["obj_id"]]

        if pg_text.empty or pg_objs.empty:
            continue

        dist_matrix = pd.DataFrame(
            pairwise_distances(
                pg_text[coordinates], pg_objs[coordinates], metric="manhattan"
            ),
            index=pg_text["txt_id"],
            columns=pg_objs["obj_id"],
        )

        dist_list.append((pg, find_min_distance(dist_matrix)))

    df_links = pd.DataFrame(dist_list, columns=["page", "links"])

    logger.info("Done!")

    return df_guia1, df_links

def join_guia_links(df_guia1, df_links, logger):

    logger.info("Creating dataframe with links of objects and texts...")

    df_links1 = pd.DataFrame(
        df_links.explode(column="links")["links"].tolist(),
        columns=["id", "id_linked_obj"],
    )

    assert df_links1["id"].nunique() == df_links1.shape[0]

    df_guia2 = df_guia1.merge(df_links1, on="id", how="left")

    assert df_guia2.shape[0] == df_guia1.shape[0]

    return df_guia2