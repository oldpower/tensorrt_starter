# mysql_client.py
from mysql.connector import connect, Error, pooling
from contextlib import contextmanager
import logging

# 可选：配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MySQLClient:
    def __init__(self, host, port, user, password, database, autocommit=False, pool_name="mypool", pool_size=5):
        """
        初始化 MySQL 客户端（使用连接池）
        """
        try:
            self.config = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database,
                "autocommit": autocommit,
                "charset": "utf8mb4",
                "use_unicode": True,
                "pool_name": pool_name,
                "pool_size": pool_size,
                "pool_reset_session": True
            }
            # 创建连接池
            self.pool = pooling.MySQLConnectionPool(**self.config)
            logger.info("MySQL 连接池初始化成功")
        except Error as e:
            logger.error(f"MySQL 连接池初始化失败: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        获取连接的上下文管理器，确保连接被正确释放
        """
        conn = None
        try:
            conn = self.pool.get_connection()
            yield conn
        except Error as e:
            if conn:
                conn.rollback()
            logger.error(f"数据库操作出错: {e}")
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()  # 归还到连接池

    def insert_one(self, sql, params=None):
        """
        插入一条记录
        :param sql: INSERT SQL 语句（带 %s 占位符）
        :param params: 参数元组或列表
        :return: 新插入记录的 ID（若表有自增主键）
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                last_id = cursor.lastrowid
                if not conn.autocommit:
                    conn.commit()
                cursor.close()
                return last_id
            except Exception as e:
                conn.rollback()
                cursor.close()
                raise e

    def insert_many(self, sql, params_list):
        """
        批量插入
        :param sql: INSERT SQL 语句
        :param params_list: 参数列表，如 [(val1, val2), (val1, val2), ...]
        :return: 受影响行数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params_list)
                rowcount = cursor.rowcount
                if not conn.autocommit:
                    conn.commit()
                cursor.close()
                return rowcount
            except Exception as e:
                conn.rollback()
                cursor.close()
                raise e

    def execute(self, sql, params=None):
        """
        执行任意非查询 SQL（UPDATE, DELETE 等）
        :return: 受影响行数
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                rowcount = cursor.rowcount
                if not conn.autocommit:
                    conn.commit()
                cursor.close()
                return rowcount
            except Exception as e:
                conn.rollback()
                cursor.close()
                raise e

    def fetch_one(self, sql, params=None):
        """
        查询单条记录
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            result = cursor.fetchone()
            cursor.close()
            return result

    def fetch_all(self, sql, params=None):
        """
        查询多条记录
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            result = cursor.fetchall()
            cursor.close()
            return result

if __name__ == "__main__":
    """CREATE TABLE `temp`.`reaction_text_img` (
        `id` bigint NOT NULL,
        `file_name` varchar(300),
        `num` int,
        `context` varchar(3000),
        `reaction_text` varchar(3000),
        `img_minio_url` varchar(300),
        `commnt_1` varchar(1000),
        `commnt_2` varchar(1000)
        ) ENGINE=OLAP
        PRIMARY KEY(`id`)
        DISTRIBUTED BY HASH(`id`) BUCKETS 10
        PROPERTIES("replication_num" = "1");


        CREATE TABLE `temp`.`discarded_pdf` (
        `id` bigint NOT NULL,
        `file_name` varchar(300),
        `commnt` varchar(300)
        ) ENGINE=OLAP
        PRIMARY KEY(`id`)
        DISTRIBUTED BY HASH(`id`) BUCKETS 10
        PROPERTIES("replication_num" = "1");
    """
    pass