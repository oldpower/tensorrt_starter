"""
get_rtsp_tool 的 Docstring

Error Log
===
❌ 未找到 url 字段，返回内容： {'code': '0x02401019', 'msg': 'no access_token and header parameter X-Ca-Key(appKey) found! apiName : 获取监控点预览取流URLv2'}
❌ 未找到 url 字段，返回内容： {'code': '0x02401003', 'msg': 'api AK/SK signature authentication failed,Invalid Signature! and StringToSign: POST\n*/*\napplication/json\n/artemis/api/video/v2/cameras/previewURLs, apiName : 获取监控点预览取流URLv2, appKey : 23008979'}

指的是AK\SK认证, 其中AK\SK是通过API网关发放的。原文:
1. 签名字符串组成及顺序
    HTTP METHOD + “\n” +
    ​Accept +“\n” + //建议显示设置 AcceptHeader, 部分 Http 客户端当 Accept 为空时会给 Accept 设置默认值：/，导致签名校验失败。
    ​Content-MD5 + “\n” +
    ​Content-Type +“\n” +
    ​Date +“\n” +
    ​Headers +
    ​Url
    HTTPMethod为全大写, 如 “POST”。如果请求headers中不存在Accept、Content-MD5、Content-Type、Date 则不需要添加换行符”\n”。

Demo:
    HTTP METHOD  -> POST
    ​Accept       -> */*
    ​Content-Type -> application/json
    ​Date         -> Mon, 25 Dec 2025 09:24:00 GMT
    ​Url          -> /artemis/api/video/v2/cameras/previewURLs

2. 关于签名说明原文：签名字符串由Http Method、headers、Url(指path+query+bodyForm)组成。以appSecret为密钥，使用HmacSHA256算法对签名字符串生成消息摘要，对消息摘要使用BASE64算法生成签名（签名过程中的编码方式全为UTF-8）。
"""


def get_rtsp_url():
    import hmac
    import hashlib
    import base64
    import requests
    from datetime import datetime, timezone

    # === 配置你的 AK/SK 和服务器 ===
    ACCESS_KEY = "23008979"          # appKey
    SECRET_KEY = "xHSqwGbYufdex5XCpmi3"  
    HOST = "192.168.60.178"          # 设备 IP
    PORT = "443"                     # HTTPS 端口（Artemis 默认 443）

    # === 构造请求 ===
    url_path = "/artemis/api/video/v1/cameras/previewURLs"
    full_url = f"https://{HOST}:{PORT}{url_path}"

    payload = {
        "cameraIndexCode": "9339391a2f584277bf79b1ccab78ecff",
        "streamType": 0,
        "protocol": "rtsp",
        "transmode": 1,
        "expand": "transcode=0",
        "streamform": "ps"
    }

    # === 生成 GMT 时间（必须！）===
    gmt_time = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')

    # === 构造 StringToSign（5行！）===
    string_to_sign = f"POST\n*/*\napplication/json\n{gmt_time}\n{url_path}"

    # === 生成签名 ===
    signature = base64.b64encode(
        hmac.new(
            SECRET_KEY.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')

    # === 设置 Headers ===
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Date": gmt_time,                    # 必须与签名中的时间一致
        "X-Ca-Key": ACCESS_KEY,              # 即 appKey
        "X-Ca-Signature": signature
    }

    # === 发送请求（跳过 SSL 验证，因自签名证书）===
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    response = requests.post(
        full_url,
        json=payload,
        headers=headers,
        verify=False  # 因为是自签名证书
    )

    # === 处理响应 ===
    if response.status_code != 200:
        return False
    else:
        print("Status Code:", response.status_code)
        print("Response:", response.json())
        return response.json()['data']['url']

def get_vedio():
    import cv2
    import time
    # rtsp_url = "rtsp://192.168.60.178:554/openUrl/vsigZWQwITC08b85dcdd6984a589979a"
    rtsp_url = get_rtsp_url()
    print(rtsp_url)
    if rtsp_url is False:
        print("rtsp取流链接获取异常")
        return
    # 打开视频流
    # cap = cv2.VideoCapture(rtsp_url)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_HW_DEVICE, 0)

    # 可选：设置缓冲（降低延迟）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ 无法打开 RTSP 流，请检查 URL、网络或权限")
        exit()

    print("✅ 成功打开 RTSP 流，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 无法读取帧（可能流已过期或断开）")
            break

        # 显示画面
        frame = cv2.resize(frame,(300,200))
        cv2.imshow('RTSP Stream', frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()



def save_video():
    import cv2
    import time
    import os
    import shutil

    output_file = "./assets/save_rtsp_vedio/recording.mp4"
    segment_duration = 3600  # 每段 1 小时（秒）

    while True:
        # 获取 RTSP 地址
        rtsp_url = get_rtsp_url()
        if rtsp_url is False:
            print("❌ RTSP 链接获取失败，5 秒后重试...")
            time.sleep(5)
            continue

        # 打开视频流
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 启用硬件加速（见第二部分解释）
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)

        if not cap.isOpened():
            print("❌ 无法打开流，5 秒后重试...")
            cap.release()
            time.sleep(5)
            continue

        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ 开始录制: {width}x{height} @ {fps} FPS")

        # 创建临时文件（避免覆盖未完成的文件）
        # temp_file = "../assets/save_rtsp_vedio/recording_temp.mp4"
        temp_file = f"./assets/save_rtsp_vedio/recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

        if not out.isOpened():
            print("❌ 视频写入器初始化失败")
            cap.release()
            time.sleep(5)
            continue

        start_time = time.time()
        frame_count = 0

        try:
            while (time.time() - start_time) < segment_duration:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 帧读取失败，重新连接...")
                    break
                out.write(frame)
                frame_count += 1
                time.sleep(0.01)  # 减少 CPU 占用

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            out.release()
            cap.release()
            if os.path.exists(temp_file):
                shutil.move(temp_file, output_file)
            return

        finally:
            out.release()
            cap.release()

        # 完整录完 1 小时，安全覆盖主文件
        # if os.path.exists(temp_file):
        #     shutil.move(temp_file, output_file)
        #     print(f"✅ 已更新 {output_file}（{frame_count} 帧，{(time.time() - start_time):.1f} 秒）")

        # 短暂等待后开始下一轮（自动刷新 RTSP URL）
        time.sleep(1)

def save_video_hardvideocapture():
    import cv2
    import time
    import os
    import shutil
    from HardVideoCapture import HardVideoCapture

    output_file = "./assets/save_rtsp_vedio/recording.mp4"
    segment_duration = 3600  # 每段 1 小时（秒）

    while True:
        # 获取 RTSP 地址
        rtsp_url = get_rtsp_url()
        if rtsp_url is False:
            print("❌ RTSP 链接获取失败，5 秒后重试...")
            time.sleep(5)
            continue

        # 先用软解快速探测分辨率和编码格式（可选，也可手动指定）
        # temp_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # if not temp_cap.isOpened():
        #     print(f"无法打开RTSP流探测: {rtsp_url}")
        #     return
        # width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = temp_cap.get(cv2.CAP_PROP_FPS)
        # temp_cap.release()

        width = 2560
        height = 1920
        fps = 25
        # 假设是 H.264（海康多数是 H.264），如需 H.265 改为 codec='hevc'
        cap = HardVideoCapture(rtsp_url, gpu_id=0, width=width, height=height, codec='h264')
        if not cap.isOpened():
            print(f"无法打开RTSP流 (硬解): {rtsp_url}")
            cap.release()
            time.sleep(5)
            continue

        # 获取视频参数
        print(f"✅ 开始录制: {width}x{height} @ {fps} FPS")

        # 创建临时文件（避免覆盖未完成的文件）
        # temp_file = "../assets/save_rtsp_vedio/recording_temp.mp4"
        temp_file = f"./assets/save_rtsp_vedio/recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

        if not out.isOpened():
            print("❌ 视频写入器初始化失败")
            cap.release()
            time.sleep(5)
            continue

        start_time = time.time()
        frame_count = 0

        try:
            while (time.time() - start_time) < segment_duration:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 帧读取失败，重新连接...")
                    break
                out.write(frame)
                frame_count += 1
                time.sleep(0.01)  # 减少 CPU 占用

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            out.release()
            cap.release()
            if os.path.exists(temp_file):
                shutil.move(temp_file, output_file)
            return

        finally:
            out.release()
            cap.release()

        # 完整录完 1 小时，安全覆盖主文件
        # if os.path.exists(temp_file):
        #     shutil.move(temp_file, output_file)
        #     print(f"✅ 已更新 {output_file}（{frame_count} 帧，{(time.time() - start_time):.1f} 秒）")

        # 短暂等待后开始下一轮（自动刷新 RTSP URL）
        time.sleep(1)

if __name__ == "__main__":
    # get_rtsp()
    # get_vedio()
    # save_video()
    save_video_hardvideocapture()