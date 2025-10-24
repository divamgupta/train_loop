import zipfile
import tarfile
import cv2
import numpy as np

class ZipImageReader:
    """
    Helper class to read images from a zip file with deep search.
    """
    def __init__(self, zip_path, img_ext=".jpg"):
        self.zip_path = zip_path
        self.img_ext = img_ext.lower()
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        # Deep search for image files, ignore __MACOSX paths
        self.image_names = [name for name in self.zip_file.namelist()
                            if name.lower().endswith(self.img_ext) and "__MACOSX" not in name]
        if len(self.image_names) == 0:
            raise ValueError(f"No images found in zip {zip_path} with extension {img_ext}")

    def __len__(self):
        return len(self.image_names)

    def get_image(self, idx):
        img_name = self.image_names[idx]
        img_data = self.zip_file.read(img_name)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image {img_name} from zip {self.zip_path}")
        return img, img_name

    def close(self):
        self.zip_file.close()

class TarGzImageReader:
    """
    Helper class to read images from a tar.gz file with deep search.
    """
    def __init__(self, tar_path, img_ext=".jpg"):
        self.tar_path = tar_path
        self.img_ext = img_ext.lower()
        self.tar_file = tarfile.open(tar_path, 'r:gz')
        # Deep search for image files, ignore __MACOSX paths
        self.image_members = [m for m in self.tar_file.getmembers()
                              if m.name.lower().endswith(self.img_ext) and m.isfile() and "__MACOSX" not in m.name]
        if len(self.image_members) == 0:
            raise ValueError(f"No images found in tar.gz {tar_path} with extension {img_ext}")

    def __len__(self):
        return len(self.image_members)

    def get_image(self, idx):
        member = self.image_members[idx]
        img_data = self.tar_file.extractfile(member).read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image {member.name} from tar.gz {self.tar_path}")
        return img, member.name

    def close(self):
        self.tar_file.close()