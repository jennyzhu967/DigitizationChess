import cv2
import os
import typing
import numpy as np
import matplotlib.pyplot as plt

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/chessChild/model.onnx")

    accum_cer = []
    accum_wer = [] 

    directory = "C:/Users/ericz/OneDrive/Desktop/IAM Model/chessTest"
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)

        image = cv2.imread(image_path)
        label = filename.replace('.png', '')
        prediction_text = model.predict(image)
        
        try:
            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            wer = get_wer(prediction_text, label)
            print(f"WER: {wer}")

            accum_cer.append(cer)
            accum_wer.append(wer)

        except: 
            print()
        
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
