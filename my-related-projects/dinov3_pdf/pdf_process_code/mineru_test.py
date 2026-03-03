import numpy as np
import cv2
import io
import time
import os
import re
from minio_tool import minio_clint,bucket_name_,minio_sava_path_pre
import hashlib
import random
import base64
import requests
from context_process import create_input_image_information,get_imagename
from my_log import log_info,Level
from mysql_tool import MySQLClient

SQL_HOST    ="192.168.103.200"
SQL_PORT    =3306
SQL_USER    ="root"
SQL_PASSWARD="Ryzh!0329"
SQL_DATABASE="scihub_reaction_information"
MINIO_DIR   ='reaction_information_and_image'

Len_INF     = 100

########################################
#             数据库工具函数 
########################################
def make_id_from_file(filepath: str) -> int:
    """根据文件路径和修改时间生成稳定、唯一、BIGINT 兼容的 ID"""
    mtime = os.path.getmtime(filepath)
    # 构造确定性键（绝对路径 + 精确到微秒的时间）
    key = f"{os.path.abspath(filepath)}:{mtime:.6f}"
    # SHA256 哈希，取前 8 字节转为非负 BIGINT
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], 'big') & 0x7FFFFFFFFFFFFFFF

def generate_id():
    EPOCH = 1767225600  # 2026-01-01 00:00:00 UTC
    return (int(time.time()) - EPOCH) * 1000 + random.randint(0, 999)

def insert_reaction_information_and_image(id, file_name, information, img_minio_url,img_markdown_name, commnt):
    """
    向 reaction_text_img 表插入一条数据
    
    参数:
    id: 主键ID
    file_name: 文件名
    information: 反应信息（可为None）
    img_minio_url: 图片MinIO URL
    img_markdown_name: 图片在markdown中的名称
    commnt: 备注（可为None）
    """
    # from mysql_tool import MySQLClient
    db = MySQLClient(
        host=SQL_HOST,
        port=SQL_PORT,
        user=SQL_USER,
        password=SQL_PASSWARD,
        database=SQL_DATABASE,
        autocommit=False
    )
    return db.insert_one(
        """
        INSERT INTO reaction_information_and_image 
        (id, file_name, information, img_minio_url,img_markdown_name, commnt)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (id, file_name, information, img_minio_url,img_markdown_name, commnt)
    )


def insert_discarded_pdf(id, file_name, commnt):
    """
    向 reaction_text_img 表插入一条数据
    
    参数:
    id: 主键ID
    file_name: 文件名
    commnt: 备注1（可为None）
    """
    # from mysql_tool import MySQLClient
    db = MySQLClient(
        host=SQL_HOST,
        port=SQL_PORT,
        user=SQL_USER,
        password=SQL_PASSWARD,
        database=SQL_DATABASE,
        autocommit=False
    )
    return db.insert_one(
        """
        INSERT INTO discarded_pdf 
        (id, file_name, commnt)
        VALUES (%s, %s, %s)
        """,
        (id, file_name, commnt)
    )


def check_sql_pdf_file_exists(file_name):
    """
    检查文件名是否已存在于数据库中
    
    参数:
    file_name: 要检查的文件名
    
    返回:
    如果存在返回 True，否则返回 False
    """
    # from mysql_tool import MySQLClient
    db = MySQLClient(
        host=SQL_HOST,
        port=SQL_PORT,
        user=SQL_USER,
        password=SQL_PASSWARD,
        database=SQL_DATABASE,
        autocommit=False
    )
    results_reaction_text_img = db.fetch_all(
        "SELECT file_name FROM reaction_information_and_image WHERE file_name = %s",
        (file_name,)  # 注意这里需要是元组
    )

    results_discarded_pdf = db.fetch_all(
        "SELECT file_name FROM discarded_pdf WHERE file_name = %s",
        (file_name,)  # 注意这里需要是元组
    )
    return len(results_reaction_text_img) > 0 or len(results_discarded_pdf)>0


#############################################
#       ☆提取反应文本核心操作流程☆
#############################################
def reaction_information_pipline(input_pdf_path):
    """
    mineru pdf test, 给MinerU输入整个PDF
    """
    def base64_to_cv2_image(base64_str: str) -> np.ndarray:
        if base64_str.startswith("data:image/"):
            # 分割字符串，取逗号后面的部分
            base64_str = base64_str.split(',')[1]
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)

    folder_name = os.path.basename(os.path.dirname(input_pdf_path))  # 10.5562
    pdf_name    = os.path.basename(input_pdf_path)                   # cca3528.pdf
    last_two    = f"{folder_name}/{pdf_name}"                           # 10.5562/cca3528.pdf
    pdf_name_no_ext = os.path.splitext(pdf_name)[0]
    last_two_no_ext = os.path.splitext(last_two)[0]

    if check_sql_pdf_file_exists(last_two):
        log_info(Level.VERB, f"file has processed : %s",last_two)
        return

    response = requests.post(
        "http://192.168.103.203:8000/file_parse",
        files={"files": open(input_pdf_path, "rb")},
        data={
            'return_middle_json': 'false',
            'return_model_output': 'false', 
            'return_md': 'true',
            'return_images': 'true',
            'end_page_id': '99999',
            'parse_method': 'auto',
            'start_page_id': '0',
            'lang_list': 'ch',
            'output_dir': '',
            'server_url': 'string',
            'return_content_list': 'false',
            'backend': 'pipeline',
            'table_enable': 'false',
            'response_format_zip': 'false',
            'formula_enable': 'false',

        },
        headers={"accept": "application/json"}
    )

    num = 1
    flag = False

    md_images,passage_text = create_input_image_information(response.json()['results'][pdf_name_no_ext]['md_content'])
    img_minio_url_text = ""
    images_text = ""
    if len(passage_text) > Len_INF:
        for md_image in md_images:
            cut_image_name = get_imagename(md_image)
            cut_image_byte = response.json()['results'][pdf_name_no_ext]['images'][cut_image_name]
            img = base64_to_cv2_image(cut_image_byte)
            _ , buffer = cv2.imencode('.jpg', img)
            img_bytes = buffer.tobytes()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            resp = requests.post(
                "http://192.168.60.205:8282/reaction_class/",
                json={"image": img_b64},
                timeout=60
            )
            if resp.json()["result"] == 1 and float(resp.json()["confidence"]) > 0.85:
                # reaction_text = unstreamfunc("qwen3:14b",one_data[0],one_data[1])
                try:
                    object_name = f"{MINIO_DIR}/{last_two_no_ext}_{num:02d}.jpg"
                    minio_clint.put_object(object_name, img, bucket_name_)
                    img_url = f"{minio_sava_path_pre}/{object_name}"
                    img_minio_url_text = img_minio_url_text + img_url + '\n\n'
                    images_text = images_text + cut_image_name + '\n\n'
                    flag = True
                    num+=1
                except Exception as e:
                    log_info(Level.ERROR, "minio upload failed: %s", e)

    if flag:
        id = generate_id()
        file_name = last_two
        information = passage_text
        img_minio_url = img_minio_url_text
        img_markdown_name = images_text
        commnt = ""
        try:
            insert_reaction_information_and_image(id, file_name, information, img_minio_url,img_markdown_name, commnt)
            log_info(Level.INFO, "insert sql data success: %s | %s", file_name, num)
        except Exception as e:
            log_info(Level.ERROR, "insert_reaction_information_and_image failed: %s", e)
    else:
        log_info(Level.VERB, "pass: %s", last_two)
        id = generate_id()
        file_name = last_two
        commnt = ""
        try:
            insert_discarded_pdf(id,file_name,commnt)
            log_info(Level.VERB, "insert sql data success: %s", file_name)
        except Exception as e:
            log_info(Level.ERROR, "insert_discarded_pdf failed: %s", e)


def postprocess():
    base_path = '/home/备份文件/有机合成/原始数据/scihub_data/04_reaction_extract_True'
    # root: 当前正在访问的文件夹路径
    # dirs: 当前文件夹下的子文件夹名列表
    # files: 当前文件夹下的文件名列表
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                log_info(Level.VERB, "post process file: %s", full_path)
                reaction_information_pipline(full_path)

if __name__ == "__main__":
    postprocess()
    # while True:
    #     postprocess()
    #     time.sleep(30)
    #     break
    # pass


    # SET NAMES utf8mb4;
    # SET FOREIGN_KEY_CHECKS = 0;

    # -- ----------------------------
    # -- Table structure for reaction_text_img
    # -- ----------------------------
    # DROP TABLE IF EXISTS `reaction_information_and_image`;
    # CREATE TABLE `reaction_information_and_image` (
    # `id` bigint NOT NULL,
    # `file_name` varchar(1000) DEFAULT NULL,
    # `information` text DEFAULT NULL,
    # `img_minio_url` text DEFAULT NULL,
    #     `img_markdown_name` text DEFAULT NULL,
    # `commnt` varchar(1000) DEFAULT NULL
    # ) ENGINE=InnoDB 
    # DEFAULT CHARSET=utf8mb4 
    # COLLATE=utf8mb4_unicode_ci;

    # SET FOREIGN_KEY_CHECKS = 1;



    # SET NAMES utf8mb4;
    # SET FOREIGN_KEY_CHECKS = 0;

    # -- ----------------------------
    # -- Table structure for discarded_pdf
    # -- ----------------------------
    # DROP TABLE IF EXISTS `discarded_pdf`;
    # CREATE TABLE `discarded_pdf` (
    # `id` bigint NOT NULL,
    # `file_name` varchar(300) DEFAULT NULL,
    # `commnt` varchar(300) DEFAULT NULL
    # ) ENGINE=InnoDB 
    # DEFAULT CHARSET=utf8mb4 
    # COLLATE=utf8mb4_unicode_ci;

    # SET FOREIGN_KEY_CHECKS = 1;


