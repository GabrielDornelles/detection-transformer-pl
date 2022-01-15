
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100, VOCDetection
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from torchmetrics.functional import accuracy
from torchvision.models import resnet50
from detr_criterion import SetCriterion, HungarianMatcher
from rich.console import Console

console = Console()

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

def preprocess_batch(y, device = torch.device("cuda")):
    num_of_objects = len(y["annotation"]["object"])
    y_boxes   = [y["annotation"]["object"][i]["bndbox"] for i in range(num_of_objects)]
    y_boxes   = [ [ int(x["xmin"][0]), int(x["xmax"][0]), int(x["ymin"][0]), int(x["ymax"][0]) ] for x in y_boxes]
    y_targets = [object_categories.index(y["annotation"]["object"][i]["name"][0]) for i in range(num_of_objects)]
    y_boxes = torch.tensor(y_boxes, dtype=torch.float)
    y_targets = torch.tensor([y_targets], dtype=torch.long) # hardcode a dimension. also tensors used as indices must be long, byte or bool tensors
    y_boxes = y_boxes[None,:]
    targets = []
    for target,box in zip(y_targets,y_boxes):
        targets.append({"labels": target.to(device), "boxes":box.to(device)})
    return targets

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DetectionTransformer(pl.LightningModule):
    """
    https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.matcher = HungarianMatcher()
        self.weight_dict = {"loss_ce": 1, "loss_bbox": 5}
        self.losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes=20, matcher=self.matcher, weight_dict=self.weight_dict, eos_coef=0.1, losses=self.losses)

        # ResNet backbone
        self.backbone = resnet50()
        del self.backbone.fc
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4) # use 3-layer MLP later

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                                self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), # (100,21) tensor
                'pred_boxes': self.linear_bbox(h).sigmoid()} # (100,4) tensor

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
	
    def evaluate(self, batch, stage = None):
        x, y = batch
     

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        targets = preprocess_batch(y)
        outputs = self(x)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.log('train_loss', losses)
        return losses 
	
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        # there's no test but if you want simple call evaluate like that
        self.evaluate(batch, "test")

# data
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset = VOCDetection(root="./", download=True, transform=data_transforms)
train_size = int(len(dataset) * 0.9)
val_size = int(len(dataset) - train_size)
cifar_train, cifar_val = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(cifar_train, batch_size=1, num_workers=4)
val_loader = DataLoader(cifar_val, batch_size=1, num_workers=4)

# model
model = DetectionTransformer(num_classes=20)

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, limit_train_batches=1.0, callbacks=RichProgressBar())

try:
    trainer.fit(model, train_loader, val_loader)
except Exception:
    console.print_exception()
