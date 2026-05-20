import sys
import os
import datetime
import time
from pathlib import Path


class Logger:
    def __init__(self, args, log_dir):
        self.log_dir = log_dir
        self.log_folder_name = time.strftime("%Y%m%d_%H%M%S")
        self.log_folder_path = os.path.join(log_dir, self.log_folder_name)
        os.makedirs(self.log_folder_path, exist_ok=True)
        
 
        log_file_path = os.path.join(self.log_folder_path, f"{self.log_folder_name}.txt")
        self.log = open(log_file_path, 'w')
        
 
        self.terminal = sys.stdout
        sys.stdout = self  
        

    def write(self, message):
        self.terminal.write(message)  
        self.log.write(message)       
        self.log.flush()               

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        if not self.log.closed:
            self.log.close()
    def __del__(self):
        sys.stdout = self.terminal    
        if not self.log.closed:
            self.log.close()



