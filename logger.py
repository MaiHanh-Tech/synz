import logging
import os
from datetime import datetime

class AppLogger:
    def __init__(self):
        self.logger = logging.getLogger("CognitiveWeaver")
        self.logger.setLevel(logging.INFO)
        
        # Tự động tạo thư mục logs nếu chưa có
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Tạo file log theo ngày
        fh = logging.FileHandler(f"logs/app_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def log_api_call(self, model, tokens, latency, success):
        """Ghi lại lịch sử gọi AI"""
        self.logger.info(f"API_CALL | Model={model} | Tokens={tokens} | Latency={latency:.2f}s | Success={success}")
    
    def log_error(self, module, error, traceback):
        """Ghi lại lỗi"""
        self.logger.error(f"ERROR | Module={module} | Error={error}\n{traceback}")
