# import packages
from qgis.core import QgsTask, QgsApplication, QgsProject, QgsVectorLayer, QgsField, QgsWkbTypes
from qgis.PyQt.QtCore import QVariant
import torch
import numpy as np

# helper function to initialize model 
class ImgClassifier(torch.nn.Module):
    def __init__(self, model_arch, n_class, n_channels=3, pretrained=False):
        super().__init__()
        import timm
        self.model = timm.create_model(model_arch, in_chans=n_channels, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_class)
    def forward(self, x):
        return self.model(x)

# QGIS task to extract and classify chips
class ClassifyChipTask(QgsTask):
    def __init__(self, raster_layer_id, vector_layer_id, model_config):
        super().__init__("Extract & Classify Chips", QgsTask.CanCancel)
        print(f'raster_layer_id: {raster_layer_id}')
        print(f'vector_layer_id: {vector_layer_id}')
        print(f'model_config: {model_config}')

        self.ras = QgsProject.instance().mapLayer(raster_layer_id)
        self.vec = QgsProject.instance().mapLayer(vector_layer_id)

        self.MODEL_NAME = model_config['model_name']
        self.CKPT = model_config['weights_ckpt']
        self.CLASS_NAMES = model_config['class_names']
        self.MEAN = model_config['normalization_mean']
        self.STD = model_config['normalization_std']
        self.MODEL_ARCH = model_config['model_arch']
        self.N_CHANNELS = model_config['n_channels']
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        self.IMG_SIZE = model_config['image_size']
        
        if not self.MODEL_NAME:
            self.MODEL_NAME = self.CKPT.replace('\\','/').split('/')[-1].split('.')[0]

        self.ATTR_LABEL_FIELD = "label"        # string label
        self.ATTR_PROB_PREFIX = "p_"           # p_deadtree, p_topdownthreat, p_tree
        self.OUT_LAYER_NAME = f"{self.MODEL_NAME}_classification"
        self.results = {}   # key: feature id; val: (label, probs)

    def _prep_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImgClassifier(self.MODEL_ARCH, self.NUM_CLASSES, n_channels=self.N_CHANNELS, pretrained=False)
        model.load_state_dict(torch.load(self.CKPT, map_location='cpu'))
        model.to(device).eval()
        return model, device

    def _read_chip_np(self, src_ds, xoff, yoff, xsize, ysize, device):
        from albumentations import Compose, Resize, Normalize
        from albumentations.pytorch import ToTensorV2
        arr = src_ds.ReadAsArray(xoff, yoff, xsize, ysize)  # shape: (C, H, W)
        arr = np.transpose(arr, (1, 2, 0)) # shape: (H, W, C)
        max_pixel_value = 255.0 if arr.max() > 1.5 else 1.0
        tfm = Compose([
            Resize(self.IMG_SIZE, self.IMG_SIZE), 
            Normalize(mean=self.MEAN, std=self.STD, max_pixel_value=max_pixel_value), 
            ToTensorV2()
            ])
        x = tfm(image=arr)['image']
        return x.unsqueeze(0).float().to(device)

    def run(self):
        import time 
        start_time = time.time()
        # 1) reproject vector to match raster CRS 
        if self.vec.crs() != self.ras.crs():
            import processing
            self.vec = processing.run("native:reprojectlayer", {
                "INPUT": self.vec, "TARGET_CRS": self.ras.crs(), "OUTPUT": "memory:"
            })["OUTPUT"]

        # 2) load Raster grid extent and source
        extent = self.ras.extent()
        px_w = extent.width()  / self.ras.width()
        px_h = extent.height() / self.ras.height()
        origin_x = extent.xMinimum()
        origin_y = extent.yMaximum()
        
        from osgeo import gdal
        src_ds = gdal.Open(self.ras.source(), gdal.GA_ReadOnly)

        # 3) extract and classify chips 
        model, device = self._prep_model()
        feats = list(self.vec.getFeatures())
        total = len(feats)

        with torch.no_grad():
            for i, f in enumerate(feats):
                if self.isCanceled():
                    break
                
                # extract bounding box 
                e = f.geometry().boundingBox()
                xoff  = int((e.xMinimum() - origin_x) / px_w)
                yoff  = int((origin_y - e.yMaximum()) / px_h)
                xsize = int(e.width()  / px_w)
                ysize = int(e.height() / px_h)
                if xsize <= 0 or ysize <= 0:
                    continue
                
                # read chip 
                chip_np = self._read_chip_np(src_ds, xoff, yoff, xsize, ysize, device)

                # classify chip 
                with torch.no_grad():
                    logits = model(chip_np)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                label = self.CLASS_NAMES[int(probs.argmax())]

                # save results 
                self.results[f.id()] = (label, probs.tolist())
                self.setProgress(100.0 * (i + 1) / max(1, total))

        src_ds = None
        end_time = time.time()
        print(f'Processing completed; time taken {round((end_time-start_time)/60,2)} minutes')
        return True

    def finished(self, ok):
        pass