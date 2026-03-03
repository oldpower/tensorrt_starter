import base64
import configparser
import io
from io import BytesIO

import cv2
from minio import Minio
from datetime import timedelta
from xmlrpc.client import ResponseError

from tqdm import tqdm

# 读取配置文件

minio_host = "192.168.60.206"
minio_port = "9000"
minio_user = "admin"
minio_password = "yhcminio123"
bucket_name_ = "scihub"

minio_sava_path_pre = "http://192.168.60.165:9008/scihub"

class MyMinio:
    def __init__(self):
        self.minio_client = Minio(f"{minio_host}:{minio_port}",
                                  access_key=minio_user,
                                  secret_key=minio_password,
                                  secure=False)

    # 文件上传
    def put_file(self, object_name, file_path, bucket_name=bucket_name_):
        """
        上传文件到minio
        @param object_name: minio文件名
        @param file_path: 本地文件地址
        @param bucket_name: 桶名
        @return:
        """
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        try:
            self.minio_client.fput_object(bucket_name, object_name, file_path)
        except ResponseError as err:
            print(err)

    def put_base(self, object_name, file_base, bucket_name=bucket_name_):
        """
        上传bae64到minio
        @param object_name: minio文件名
        @param file_base: 本地base64
        @param bucket_name: 桶名
        @return:
        """
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        image_data = base64.b64decode(file_base)
        bytes_io = BytesIO(image_data)
        self.minio_client.put_object(
            bucket_name,
            object_name,
            data=bytes_io,
            length=len(image_data),  # 内容长度
            content_type='image/svg+xml'  # 或者 'image/jpeg'，根据实际内容类型设置
        )
        # return f"http://{get_key('Minio', 'HOST')}:{get_key('Minio', 'PORT')}/{bucket_name}/{object_name}"
        return object_name

    def put_object(self, object_name, frame, bucket_name):
        found = self.minio_client.bucket_exists(bucket_name)
        if not found:
            self.minio_client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)

        params = [cv2.IMWRITE_JPEG_QUALITY, 50]  # ratio:0~100
        # msg = cv2.imencode(".jpg", img, params)[1]
        _, buffer = cv2.imencode('.jpg', frame, params)
        image_bytes = io.BytesIO(buffer)
        self.minio_client.put_object(
            bucket_name,
            object_name,
            image_bytes,
            image_bytes.getbuffer().nbytes  # 获取字节流的大小
        )
        return object_name


        # 文件下载
    def get_file(self, object_name, file_path, bucket_name=bucket_name_):
        """
        下载minio文件到本地
        @param object_name: minio文件名
        @param file_path: 本地路径
        @param bucket_name: 桶名
        @return:
        """
        try:
            self.minio_client.fget_object(bucket_name, object_name, file_path)
        except ResponseError as err:
            print(err)

    # 获取文件地址
    def get_url(self, method, object_name, bucket_name=bucket_name_):
        """
        生成minio文件地址为url
        @param method: 方法
        @param object_name: minio文件名
        @param bucket_name: 桶名
        @return:
        """
        if method == '':
            url = f"http://{minio_host}:{minio_port}/{bucket_name}/{object_name}"
        else:
            try:
                url = self.minio_client.get_presigned_url(method, bucket_name, object_name, expires=timedelta(days=7))
            except ResponseError as err:
                url = f"http://{minio_host}:{minio_port}/{bucket_name}/{object_name}"
        return url

    # 删除存储桶中的一个对象
    def del_file(self, object_name, bucket_name=bucket_name_):
        """
        删除minio文件
        @param object_name: minio文件名
        @param bucket_name: 桶名
        @return:
        """
        try:
            self.minio_client.remove_object(bucket_name, object_name)
            print("Sussess")
        except ResponseError as err:
            print(err)

    # 删除存储桶中的多个对象
    def del_files(self, delete_list, bucket_name=bucket_name_):
        """
        批量删除minio文件
        @param delete_list: minio文件列表
        @param bucket_name: 桶名
        @return:
        """
        try:
            for del_err in self.minio_client.remove_objects(bucket_name, delete_list):
                print("Deletion Error: {}".format(del_err))
            print("Sussess")
        except ResponseError as err:
            print(err)

    def del_bucket_files(self, bucket_name):
        objects = self.minio_client.list_objects(bucket_name, recursive=True)
        for obj in tqdm(objects):
            # 删除每个对象
            self.minio_client.remove_object(bucket_name, obj.object_name)
        print(f"Bucket {bucket_name} is now empty.")


minio_clint = MyMinio()

if __name__ == "__main__":
    m = MyMinio()
    # m.del_bucket_files('reference-data')
    import cv2
    img = cv2.imread("./assets/3class-img/0/10.1046_j.1526-4610.2002.02126.x.pdf_01.jpg")
    m.put_object("reaction/test/test.jpg", img, bucket_name_)
