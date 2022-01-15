# detection-transformer-pl
PyTorch Lightning implementation of Facebook DETR for object detection on Pascal VOC 2012 Dataset

------------------------------

# TODO:

- Well Pascal VOC is a really huge dataset, so I didn't really train it, but it will start training when you execute detr.py. 

- I've implemented the pipeline with DETR own loss function (proposed in the same paper) that is present in the original repository. Everything is running properly (it will train!), but if it optimizes and works, I'm not sure yet. 

- Only works with batch of 1. Pascal VOC dataset available at torchvision has images with different shapes. You can either train it with 1, or try to rescale it to a fixed shape.