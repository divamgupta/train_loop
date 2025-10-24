import glob
import cv2
import torch
from torch.utils.data import Dataset
from ..utils.data_utils import ZipImageReader, TarGzImageReader

class ImageDataset(Dataset):
    """
    Dataset class for YOLO model distillation.
    Loads images from a directory or zip/tar.gz file and prepares them for the distillation process.
    """
    
    def __init__(self, img_dir="imgs", img_ext=".jpg", transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_dir: Directory containing the images or zip/tar.gz file
            img_ext: Image file extension
            transform: Optional additional transformations
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_ext = img_ext

        if img_dir.lower().endswith('.zip'):
            # Read from zip file
            self.zip_reader = ZipImageReader(img_dir, img_ext)
            self.image_paths = self.zip_reader.image_names
            self._from_zip = True
            self._from_tar = False
            print(f"Found {len(self.image_paths)} images in zip file {img_dir}")
        elif img_dir.lower().endswith('.tar.gz'):
            # Read from tar.gz file
            self.tar_reader = TarGzImageReader(img_dir, img_ext)
            self.image_paths = [m.name for m in self.tar_reader.image_members]
            self._from_zip = False
            self._from_tar = True
            print(f"Found {len(self.image_paths)} images in tar.gz file {img_dir}")
        else:
            # Read from directory
            self.image_paths = sorted(glob.glob(f"{img_dir}/**/*{img_ext}", recursive=True))
            self._from_zip = False
            self._from_tar = False
            print(f"Found {len(self.image_paths)} images in {img_dir} folder")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {img_dir} with extension {img_ext}")
            
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset by index.
        
        Returns:
            dict: {'image': image_tensor, 'path': image_path}
        """
        if getattr(self, '_from_zip', False):
            img, img_path = self.zip_reader.get_image(idx)
            img_path = f"{self.img_dir}:{img_path}"  # Indicate zip source
        elif getattr(self, '_from_tar', False):
            img, img_path = self.tar_reader.get_image(idx)
            img_path = f"{self.img_dir}:{img_path}"  # Indicate tar.gz source
        else:
            img_path = self.image_paths[idx]
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load {img_path}")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()

        
        # Apply additional transforms if specified
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return {'image': img_tensor, 'path': img_path}
    
    def __del__(self):
        # Clean up archive file handle if needed
        if hasattr(self, 'zip_reader'):
            self.zip_reader.close()
        if hasattr(self, 'tar_reader'):
            self.tar_reader.close()


    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of dictionaries {'image': image_tensor, 'path': image_path}
            
        Returns:
            dict: {'images': batch_tensor, 'paths': image_paths}
        """
        images = [item['image'] for item in batch]
        paths = [item['path'] for item in batch]
        
        # Stack images into a single batch tensor
        batch_tensor = torch.stack(images, dim=0)

        return {'image': batch_tensor, 'path': paths}