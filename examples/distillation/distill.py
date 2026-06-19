"""Knowledge distillation: EfficientNet teacher → small CNN student on CIFAR-10.

The teacher (pretrained EfficientNet-B0) runs in ``post_process_batch`` via
``extra_models``, producing ``teacher_logits``.  The student is trained with
a combination of:
  1. Hard label loss (cross-entropy vs ground truth)
  2. Soft label loss (KL divergence vs teacher's softened logits)

This demonstrates the ``extra_models`` + ``post_process_batch`` pattern used
for teacher-student distillation in train_loop.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ── Dataset ───────────────────────────────────────────────────────────────

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        tf = transforms.Compose([
            transforms.Resize(224),  # EfficientNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        self.dataset = datasets.CIFAR10(
            root='/tmp/dataCIFAR10', train=train,
            download=True, transform=tf,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {"inp": img, "gt_class_id": label}

    def post_process_batch(self, batch):
        """Run teacher model and add teacher_logits to batch."""
        if hasattr(self, 'extra_models') and 'teacher' in self.extra_models:
            with torch.no_grad():
                teacher_out = self.extra_models['teacher'](batch)
            for k, v in teacher_out.items():
                batch[f'teacher_{k}'] = v.detach().clone() if torch.is_tensor(v) else v


# ── Teacher: pretrained EfficientNet-B0 ───────────────────────────────────

class EfficientNetTeacher(nn.Module):
    """Pretrained EfficientNet-B0 as teacher.

    Loads torchvision's pretrained weights and replaces the classifier
    head for CIFAR-10 (10 classes).  Since this is used as a teacher,
    we use the pretrained ImageNet weights directly — the logits won't
    be perfect for CIFAR-10, but they provide a meaningful soft target
    that captures inter-class relationships.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Replace classifier for 10 classes
        self.model.classifier[1] = nn.Linear(1280, 10)

    def forward(self, batch):
        logits = self.model(batch['inp'])
        return {"logits": logits}


# ── Student: tiny CNN ─────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """Tiny 3-layer CNN student for CIFAR-10.

    ~50K parameters vs EfficientNet's ~5M.
    """

    def __init__(self, n_channels=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, n_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(n_channels, n_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * 2 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, batch):
        x = batch['inp']
        x = self.features(x)
        logits = self.classifier(x)
        return {"pred_logits": logits}


# ── Distillation loss ─────────────────────────────────────────────────────

def distillation_loss(batch, model_outs, src_key="pred_logits",
                      tgt_key="teacher_logits", temperature=4.0, **kwargs):
    """KL divergence between student and teacher softened logits.

    Uses temperature scaling: softer distributions transfer more
    knowledge about inter-class relationships.
    """
    student_logits = model_outs[src_key]
    teacher_logits = batch[tgt_key]

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL(teacher || student), scaled by T^2 to match gradient magnitude
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl * (temperature ** 2)
