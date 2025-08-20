import joblib
import importlib_resources
import cv2
import numpy as np
import skimage.measure

with importlib_resources.path("streetview_gens", "data/model.pkl") as p:
    model = joblib.load(p)

def analyze_histogram(im):
    hists = []
    for chan in range(3):
        hist = cv2.calcHist([im], [chan], None, [256], [0,256])
        hist = cv2.normalize(hist,hist)
        hists.extend(hist.flatten())
    return hists


def analyze_entropy(im):
    return skimage.measure.shannon_entropy(im)
def analyze_blur(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def analyze_noise(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.var(gray)


def analyze_edge_density(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels


def embed(im):
    features = []
    features.extend(analyze_histogram(im))
    features.append(analyze_entropy(im))
    features.append(analyze_blur(im))
    features.append(analyze_noise(im))
    features.append(analyze_edge_density(im))
    return np.array(features).reshape(1,-1)

def predict(im):
    return model.predict_proba(embed(im)) 
