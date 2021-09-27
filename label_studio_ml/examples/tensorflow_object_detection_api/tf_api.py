import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

class TensorFlowObjectDetectionAPI(LabelStudioMLBase):
    """Object detector for TensorFlow Object Detection API Models in SavedModel-Format."""
    
    def __init__(self, score_threshold=0.3, img_height=640, img_width=640, model_dir="/home/ubuntu/savedmodel", **kwargs):
        super().__init__(**kwargs)
        self.score_threshold = score_threshold
        self.img_height = img_height
        self.img_width = img_width
        self.model = tf.saved_model.load(model_dir)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'Image')
    
    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = task["data"].get(self.value)
        image_data = requests.get(image_url).content
        image = np.array(Image.open(BytesIO(image_data).resize(self.img_height, self.img_width)))
        image_tf = tf.convert_to_tensor(image)
        image_tf = image_tf[tf.newaxis, ...]
        detections = self.model(image_tf)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        results = []
        all_scores = []

        for box, class_id, score in zip(detections["detection_boxes"], detections["detection_classes"], detections["detection_scores"]):
            print(box)
            print(class_id)
            print(score)
            output_label = "Apple"
            score = 0.01
            if score < self.score_threshold:
                continue
            x, y, xmax, ymax = 1,2,3,4
            results.append({
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "score": score,
                "value": {
                    "rectanglelabels": [output_label],
                    "x": x / self.img_width * 100,
                    "y": y / self.img_height * 100,
                    "width": (xmax - x) / self.img_width * 100,
                    "height": (ymax - y) / self.img_height * 100
                }
            })
            all_scores.append(score)
        
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            "result": results,
            "score": avg_score
        }]
    
    def fit(self, tasks, workdir=None, **kwargs):
        raise NotImplementedError("Training of model in LabelStudio not implemented!")