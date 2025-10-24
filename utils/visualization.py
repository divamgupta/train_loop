import numpy as np
from PIL import Image
from io import BytesIO

def show_tensor(tensor, title=None, save_to=None):
    """
    Visualize numpy or pytorch tensors as images.
    """
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    img = np.array(tensor)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3:
        if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = img.squeeze(2)
    if img.dtype == np.uint8:
        pass
    elif img.max() <= 1.0 and img.min() >= 0.0:
        img = (img * 255).astype(np.uint8)
    elif img.max() <= 1.0 and img.min() >= -1.0:
        img = ((img + 1) * 127.5).astype(np.uint8)
    else:
        img = img - img.min()
        img = (img / (img.max() + 1e-8) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    if save_to is not None:
        pil_img.save(save_to)
        print(f"Image saved to {save_to}")
        return
    try:
        get_ipython()
        in_jupyter = True
    except NameError:
        in_jupyter = False
    if in_jupyter:
        from IPython.display import display
        if title:
            print(title)
        display(pil_img)
    else:
        from imgcat import imgcat
        if title:
            print(f"\n{title}")
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        imgcat(buf.getvalue())
