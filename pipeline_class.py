"""
Shared class definition for the self-contained chicken-disease pipeline.

IMPORTANT: this file must be imported (not redefined inline) both:
  - in the notebook that creates chicken_disease_pipeline_selfcontained.pkl
  - in app.py that loads it

Pickle stores a reference to the class as "<module>.<ClassName>". If the
class is defined inline in a notebook, that module is "__main__" there,
which won't match "__main__" when Streamlit imports app.py as a module
instead of running it as a script. Keeping the class in this dedicated
pipeline_class.py file gives it a stable, matching module path
("pipeline_class.ChickenDiseasePipelineSelfContained") everywhere.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf


class ChickenDiseasePipelineSelfContained:
    def __init__(self, model_bytes, class_indices, img_size, model_format="keras"):
        self.model_bytes = model_bytes      # raw bytes of the model file
        self.model_format = model_format    # "keras" or "h5"
        self.class_indices = class_indices
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        self.img_size = img_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            suffix = ".keras" if self.model_format == "keras" else ".h5"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(self.model_bytes)
                tmp_path = tmp.name
            try:
                self._model = tf.keras.models.load_model(tmp_path)
            finally:
                os.remove(tmp_path)
        return self._model

    def predict(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preds = self._load_model().predict(img_array)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        return self.idx_to_class[pred_idx], float(np.max(preds))
