import numpy as np
import skimage.feature as skfeat
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def compute_1d_color_hist(img, bins_per_hist = 32):
    """
    Compute a 1d color histogram of the image.
  
    - img: Color image (Numpy array)
    - bins_per_hist: Number of bins per histogram

    RETURN: 
    - A numpy array of shape (bins_per_hist * 3,)
    """
    histogramR = cv2.calcHist([img], [0], None, [bins_per_hist], [0, 256])
    histogramG = cv2.calcHist([img], [1], None, [bins_per_hist], [0, 256])
    histogramB = cv2.calcHist([img], [2], None, [bins_per_hist], [0, 256])

    cv2.normalize(histogramR, histogramR, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histogramG, histogramG, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histogramB, histogramB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return np.concatenate((histogramR, histogramG, histogramB)).flatten()


def compute_2d_color_hist(img, bins_per_hist = 16):
    """
    Compute a 2d color histogram of the image.
    
    The final descriptor will be the concatenation of 3 normalized 2D histograms: B/G, B/R and G/R.
  
    - img: Color image (Numpy array)
    - bins_per_hist: Number of bins per histogram

    RETURN:
    - A numpy array of shape (bins_per_hist * bins_per_hist * 3,)
    """
    histogramBG = cv2.calcHist([img], [2,1], None, [bins_per_hist, bins_per_hist], [0, 256, 0, 256])
    histogramBR = cv2.calcHist([img], [2,0], None, [bins_per_hist, bins_per_hist], [0, 256, 0, 256])
    histogramGR = cv2.calcHist([img], [1,0], None, [bins_per_hist, bins_per_hist], [0, 256, 0, 256])

    cv2.normalize(histogramBG, histogramBG, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histogramBR, histogramBR, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(histogramGR, histogramGR, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return np.concatenate((histogramBG, histogramBR, histogramGR)).flatten()

def compute_lbp_descriptor(img, p = 8, r = 1):
    """
    Compute a rotation invariant and uniform LBP histogram as image descriptor.
  
    - img: Input image (Numpy array)
    - p: Neighbors to check in radius r
    - r: Radius in pixels

    RETURN: 
    - A numpy array of shape (p + 2,)
    """    
    
    # YOUR CODE HERE
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    lbp = skfeat.local_binary_pattern(gray_img, p, r, method="uniform")
    lbp = lbp.astype(np.float32)

    hist_lbp = cv2.calcHist([lbp], [0], None, [p+2], [0, p+2])
    cv2.normalize(hist_lbp, hist_lbp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


    return hist_lbp.flatten()
    # -----

def compute_global_lbp_descriptor(img, p=8, r=1, grid_x=4, grid_y=4):
    """
    Compute a spatial LBP histogram by dividing the image into a grid.

    - img: Input image (Numpy array)
    - p: Neighbors to check in radius r
    - r: Radius in pixels
    - grid_x: width grid 
    - grid_y: height grid

    RETURN: 
    - Concatenated list with grid histograms
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    lbp = skfeat.local_binary_pattern(gray_img, p, r, method="uniform")
    lbp = lbp.astype(np.float32)

    h, w = lbp.shape
    cell_h = h // grid_y
    cell_w = w // grid_x
    
    histograms = []

    for i in range(grid_y):
        for j in range(grid_x):

            cell = lbp[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
            
            curr_hist = cv2.calcHist([cell], [0], None, [p + 2], [0, p + 2])
            
            cv2.normalize(curr_hist, curr_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            histograms.append(curr_hist.flatten())

    # Concat
    global_descriptor = np.concatenate(histograms)
    
    return global_descriptor


class CBIR:
    """
    Class to encapsulate the basic functionalities of a CBIR system.
    """
    
    def __init__(self, desc_func, **params):
        """
        Class constructor.
        
        - desc_func: The function to be used for describing the images
        - params: A variable number of parameters required to call desc_func.
            This is a dictionary that can be unpacked within the function.
            See more info here: https://realpython.com/python-kwargs-and-args/
        """
        self.desc_func = desc_func
        self.params = params
        self.image_descriptors = {}
        
    def build_image_db(self, images):
        """
        Create the CBIR system database.
        
        - images: A dictionary of images (Numpy arrays)
        
        This function should describe each image using desc_func and save the 
        resulting descriptors in a dictionary of Numpy arrays (global descriptors) called image_descriptors.
        The names should be the same than the ones in images.
        """        

        # YOUR CODE HERE
        raise NotImplementedError()
        # -----
            
    def search_image(self, query_descriptor):
        """
        Search an image in the system.
        
        - query_descriptor: Global descriptor of the query image (NumPy array)
        
        RETURNS:
        - An sorted list of tuples, each one with the format (database image name, L2 distance)
        
        This method is responsible for searching for the most similar images in the database 
        based on a query descriptor. It compares the query descriptor with all the descriptors in 
        the image database using the L2 (Euclidean) distance and returns a sorted list of results.
        """

        # List to store the results (image name and L2 distance)
        results = []

        # YOUR CODE HERE
        raise NotImplementedError()
        # -----

        return results


def extract_interest_points(img, feat_type = 'SIFT', nfeats = 500, thresh = 50):
    """
    Compute keypoints and their corresponding descriptors from an image.
  
    Parameters:
    - img: Input image (Numpy array).
    - feat_type: Detection / description method ('SIFT', 'FAST_BRIEF', 'ORB').
    - nfeats: Maximum number of features. It can be directly used to configure SIFT and ORB.
    - thresh: Detection threshold. Useful for FAST and ORB.
  
    Returns:
    - kp: A list of detected keypoints (cv2.KeyPoint).
    - des: A numpy array of shape (number_of_kps, descriptor_size) of type:
        - 'np.float32' for SIFT.
        - 'np.uint8' for BRIEF and ORB.
    """
    kp = []
    des = []


    # YOUR CODE HERE

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    match feat_type:
        case 'SIFT':
            sift = cv2.SIFT_create(nfeatures=nfeats)
            kp, des = sift.detectAndCompute(gray_img,None)
        case 'FAST_BRIEF':
            fast = cv2.FastFeatureDetector_create(threshold=thresh)
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp = fast.detect(gray_img,None)
            kp, des = brief.compute(gray_img, kp)
        case 'ORB':
            orb = cv2.ORB_create(nfeatures=nfeats, fastThreshold=thresh)
            kp = orb.detect(gray_img,None)
            kp, des = orb.compute(gray_img, kp)

    return kp, des
    
    
def find_matches(query_desc, database_desc, k=2):
    """
    Match two sets of descriptors. For each query descriptor, this method searches
    for the k closest descriptors in the database set.
  
    Parameters:
    - query_desc (np.ndarray): A NumPy array of shape (num_query_kps, descriptor_size),
      containing descriptors from the query image.
    - database_desc (np.ndarray): A NumPy array of shape (num_database_kps, descriptor_size),
      containing descriptors from the database image.
    - k (int): Number of nearest neighbors to retrieve for each descriptor (default: 2).
  
    Returns:
    - matches (list of list of cv2.DMatch): A list where each element contains k matches,
      sorted by distance.
    """
    matches = []

    # FLANN-based matching
    # SIFT
    if query_desc.dtype == np.float32:
        flann_matcher = cv2.FlannBasedMatcher()
        matches = flann_matcher.knnMatch(query_desc, database_desc, k=k)

    # Brute-Force matching
    # ORB, BRIEF
    elif query_desc.dtype == np.uint8:
        bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf_matcher.knnMatch(query_desc, database_desc, k=k)
    
    matches = sorted(matches, key=lambda x: x[0].distance)
    return matches
    # ------

def filter_matches(matches, ratio=0.75):
    """
    Filters matches using the Nearest Neighbor Distance Ratio (NNDR) criterion.

    Parameters:
    - matches (list of list of cv2.DMatch): A list where each element contains k matches,
      sorted by distance (output from 'find_matches').
    - ratio (float): The threshold for the ratio test. A match is kept if the 
      distance of the best match is less than `ratio *` the distance of the second-best match.
    
    Returns:
    - filtered_matches (list of cv2.DMatch): A list of matches that passed the ratio test.
    """

    # YOUR CODE HERE

    filtered_matches = []
    
    for match in matches:
        if len(match) >= 2:
            m, n = match[0], match[1]
            if m.distance < ratio * n.distance:
                filtered_matches.append(m)
            
    return filtered_matches
    # -----


def evaluate(dataset, method='SIFT', nfeats=3000, thresh=25, ratio=0.75):
    """
    Evaluate the image retrieval performance using local features (SIFT, ORB, etc.).
    This function computes the mean Average Precision (mAP) for the dataset.
    
    Args:
        dataset (HolidaysDatasetHandler): The dataset handler object.
        method (str): The feature extraction method to use ('SIFT', 'SURF', 'ORB').
        nfeats (int): Maximum number of features to extract.
        thresh (int): Threshold for feature detection.
        ratio (float): Nearest Neighbor Distance Ratio for filtering matches.
    
    Returns:
        float: The mean Average Precision (mAP) score for the retrieval system.
    """

    # YOUR CODE HERE

    query_images = dataset.get_query_images()
    database_images = dataset.get_database_images()

    db_features = {}
    for db_image in database_images:
        db_img_array = dataset.get_image(db_image)
        _, db_des = extract_interest_points(db_img_array, method, nfeats, thresh)
        db_features[db_image] = db_des

    mAP_dict = {}

    for query_image in query_images:
        query_img_array = dataset.get_image(query_image)
        query_kp, query_des = extract_interest_points(query_img_array, method, nfeats, thresh)
        
        current_scores = []
        
        for database_image in database_images:
            database_des = db_features[database_image]
            
            if query_des is not None and database_des is not None and len(query_des) > 0 and len(database_des) > 0:
                raw_matches = find_matches(query_des, database_des)
                matches = filter_matches(raw_matches, ratio)
                num_matches = len(matches)
                current_scores.append((database_image, num_matches))
            else:
                current_scores.append((database_image, 0))
                
        current_scores.sort(key=lambda x: x[1], reverse=True)
        mAP_dict[query_image] = [img for img, _ in current_scores]

    return dataset.compute_mAP(mAP_dict)
    # -----


class CNNFeatureExtractor:
    def __init__(self):
        """
        Initializes the ResNet-50 model pre-trained on ImageNet.
        We remove the last fully connected layer to use it as a feature extractor.
        """
        # 1. Load the model with ImageNet weights
        # We use the modern 'weights' parameter instead of 'pretrained=True'
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # 2. Remove the last layer (fc) to get the 2048-dim embedding
        # ResNet architecture: [conv1, bn1, relu, maxpool, layer1...layer4, avgpool, fc]
        # We take everything except the last one (fc)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        # 3. Set to evaluation mode (inference only)
        self.model.eval()
        
        # 4. Standard transformation pipeline for ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_feature(self, img):
        """
        Extracts the deep feature vector for a single image.
        
        Args:
            img (np.ndarray):  Input image (Numpy array) in BGR format.
            
        Returns:
            np.ndarray: A 1D numpy array of shape (2048,) containing the deep embedding.
                        The vector MUST BE L2-normalized.
        """
        # Convert OpenCV image (BGR) to PIL Image (RGB)
        # Torchvision transforms usually expect PIL Image or Tensor
        if img is None:
            print("Error: Image is None")
            return np.zeros(2048)

        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img_rgb)
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros(2048)

        # YOUR CODE HERE
        raise NotImplementedError()
        # -----


class ViTFeatureExtractor:
    def __init__(self):
        """
        Initializes the ViT-B/16 model pre-trained on ImageNet.
        We replace the classification head to use it as a feature extractor.
        """
        # 1. Load the model with ImageNet weights
        weights = models.ViT_B_16_Weights.DEFAULT
        self.model = models.vit_b_16(weights=weights)
        
        # 2. The classification head in torchvision ViT is named 'heads'.
        # We replace it with Identity to get the embedding (CLS token output).
        # Output dimension for ViT-B/16 is 768.
        self.model.heads = torch.nn.Identity()
        
        # 3. Set to evaluation mode
        self.model.eval()
        
        # 4. Use the specific transforms for this model (ViT might have specific interpolation)
        self.preprocess = weights.transforms()

    def extract_feature(self, img):
        """
        Extracts the deep feature vector for a single image using ViT.
        
        Args:
            img (np.ndarray): Input image (Numpy array) in BGR format.
            
        Returns:
            np.ndarray: A 1D numpy array of shape (768,) containing the deep embedding.
                        The vector MUST be L2-normalized.
        """
        if img is None:
            print("Error: Image is None")
            return np.zeros(768)

        try:
            # Convert OpenCV BGR to PIL RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img_rgb)
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros(768)
        
        # YOUR CODE HERE
        raise NotImplementedError()
        # -----