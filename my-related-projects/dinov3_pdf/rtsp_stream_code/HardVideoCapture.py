import cv2
import subprocess
import numpy as np

class HardVideoCapture:
    def __init__(self, rtsp_url, gpu_id=0, width=1920, height=1080, codec='h264'):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.gpu_id = gpu_id
        self.codec = codec
        self.process = None
        self._frame_buffer = b''
        self._opened = False

        # 强制 TCP 避免丢包
        if "?" not in rtsp_url:
            rtsp_url += "?tcp"
        else:
            rtsp_url += "&tcp"

        decoder = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"

        ffmpeg_cmd = [
            "ffmpeg",
            "-v", "quiet",
            "-c:v", decoder,
            "-i", rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an", "-sn",
            "-"
        ]

        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            self._opened = True
        except Exception as e:
            print(f"Failed to start FFmpeg hard decode process: {e}")
            self._opened = False

    def isOpened(self):
        return self._opened and self.process.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        self._opened = False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == cv2.CAP_PROP_FPS:
            # 注意：硬解模式下 FPS 需要预设或探测，这里返回默认值
            return 25.0
        return 0.0