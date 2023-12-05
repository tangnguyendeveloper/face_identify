from .utils import LoadModelTFlite, TFlitePredict, cropBox
from .model_path import model_path
from .mtcnn_tflite import MTCNN
import numpy as np
import cv2

'''
MODEL_PATH = "./facenet_tflite_model/embedding_20180402-114759.tflite"
'''


class MTCNNFaceNetTFlite:
    '''
     Base on keras_facenet model version 20180402-114759
    '''
    
    def __init__(self):
        self.model = LoadModelTFlite(model_path)
        self.mtcnn = MTCNN()
        self.image_size = 160

    def _normalize(self, image):
        return (np.float32(image) - 127.5) / 127.5
    
    def crop(self, filepath_or_image, threshold=0.95):
        """Get face crops from images.

        Args:
            filepath_or_image: The input image (see extract)
            threshold: The threshold to use for face detection
        """
        if isinstance(filepath_or_image, str):
            image = cv2.imread(filepath_or_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = filepath_or_image
        detections = [detection for detection in self.mtcnn.detect_faces(image) if detection['confidence'] > threshold]
        if not detections:
            return [], []
        margin = int(0.1*self.image_size)
        crops = [cropBox(image, detection=d, margin=margin) for d in detections]
        return detections, crops
    
    def extract(self, filepath_or_image, threshold=0.95):
        """Extract faces and compute embeddings in one go. Requires
        mtcnn to be installed.

        Args:
            filepath_or_image: Path to image (or an image as RGB array)
            threshold: The threshold for a face to be considered
        Returns:
            Same output as `mtcnn.MTCNN.detect_faces()` but enriched
            with an "embedding" vector.
        """
        detections, crops = self.crop(filepath_or_image, threshold=threshold)
        if not detections:
            return []
        return [{**d, 'embedding': e} for d, e in zip(detections, self.embeddings(images=crops))]

    def embeddings(self, images):
        """Compute embeddings for a set of images.

        Args:
            images: A list of images (cropped faces)

        Returns:
            Embeddings of shape (N, K) where N is the
            number of cropepd faces and K is the dimensionality
            of the selected model.
        """
        s = self.image_size
        images = [cv2.resize(image, (s, s)) for image in images]
        X = np.float32([self._normalize(image) for image in images])
        embeddings = TFlitePredict(self.model, X)
        return embeddings

    def compute_cosine_distance(self, embedding1, embedding2):
        """Compute the distance between two embeddings.

        Args:
            embedding1: The first embedding
            embedding2: The second embedding

        Returns:
            The distance between the two embeddings.
        """
        
        a = np.asarray(embedding1).flatten()
        b = np.asarray(embedding2).flatten()

        S_c = np.sum(a*b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))

        return 1 - S_c

