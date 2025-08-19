import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import glob
import os
import tqdm
import concurrent.futures
import pickle
import joblib

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

def extract_features(filename):
    im = cv2.imread(filename)
    features = []
    features.extend(analyze_histogram(im))
    #features.append(analyze_entropy(im))
    features.append(analyze_blur(im))
    features.append(analyze_noise(im))
    features.append(analyze_edge_density(im))
    return np.array(features)


def make_dataset(base_dir, max_workers=28):
    X, y = [], []
    label_map = {"gen2": 0, "gen3": 1, "shitcam": 2}
    
    files_with_labels = []
    for label_name, label_id in label_map.items():
        for file in glob.glob(os.path.join(base_dir, label_name, "samples/*/*.jpg")):
            files_with_labels.append((file, label_id))
    
    def process_file(file_label):
        file, label_id = file_label
        feats = extract_features(file)
        return feats, label_id
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_file, files_with_labels)
    
        for feats, label_id in tqdm.tqdm(results, total=len(files_with_labels)):
            X.append(feats)
            y.append(label_id)
    
    return np.array(X), np.array(y), label_map

def load_dataset(base_dir):
    try:
        with open("streetview_gens/data/dataset.pickle",'rb') as f:
            return pickle.load(f)
    except:
        result = make_dataset(base_dir)
        with open("streetview_gens/data/dataset.pickle",'wb') as f:
            pickle.dump(result,f)
        return result

def train_classifier(base_dir):
    X, y, label_map = load_dataset(base_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))

    joblib.dump(clf, "streetview_gens/data/model.pkl")
    return clf, label_map

if __name__=="__main__":
    clf, label_map = train_classifier("samples")

