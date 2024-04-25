from uuid import uuid4
import os

class DownloadHandler:
    def __init__(self, path):
        self.path = path
        self.uid_map = {}
    
    def full_path(self, name):
        return os.path.join(self.path, name)
    
    def new_download(self, file_name, extention=''):
        uid = str(uuid4()) + extention
        self.uid_map[file_name] = uid
        return self.full_path(uid)
    
    def complete_download(self, file_name):
        if file_name in self.uid_map:
            try:
                file_path = self.full_path(file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self.uid_map[file_name]
            except Exception as e:
                print(f"Error deleting file {file_name}: {e}")
