import h5py
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import numpy as np
# import docx
import sys
from main import run_net

import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import gc
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision.models import vgg16, vgg19, vgg11
from tabulate import tabulate
from config import cfg
from models.CC import CrowdCounter
from dataset.random import RandomDataset
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import sys
import h5py
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image as im
import numpy
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import MeanShift
import hdbscan
from scipy.cluster.hierarchy import centroid
from sklearn.neighbors import NearestCentroid
import numpy as np

# run_net("/content/drive/MyDrive/MobileCount/CrowdCounting-on-VisDrone2020-main/src/images/", ["save_callback"])

"""# **TRACKING CENTROIDS**
# CARICA LE IMMAGINI DA ANALIZZARE
"""



'''for dirpath, dirs, files in os.walk('C:\\Users\\seven\\Desktop\\Tesi Github\\MobileCount\\CrowdCounting-on-VisDrone2020-main\\dataset\\test\\images\\'):
    img_list_paths = []
    for filename in files:
        fname = os.path.join(dirpath, filename)
        print(fname)

        if fname.endswith('.jpg'):
            img_list_paths.append(fname)
    print(img_list_paths)
    if len(img_list_paths) !=0:
        run_net(img_list_paths, ["track_callback"])
# img_list_paths = img_list_paths[:len(img_list_paths) - 10]
print(str(len(img_list_paths)))'''

"""# DATASET CLASS"""


class CustomDataset(Dataset):
    def __init__(self, path, shape):
        self.imgs_path = path
        self.data = []
        self.shape = shape
        for img_path in glob.glob(self.imgs_path + "*.jpg"):
            self.data.append([img_path])

        '''print(path)
        for dirpath, dirs, files in os.walk(path):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith('.jpg'):
                    self.data.append(fname)'''

        self.data.sort()

        self.img_dim = (1920, 1080)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        '''img = Image.open(img_path[0])
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        img_tensor = trans1(img)'''
        return img_path

    def shape(self):
        return self.shape


def measure_forward(model, y, bg_tresh, clust_alg):
    """
    Measure the time for executing the the given tensor

    @param model: The model for measuring
    @param dataset: The tensor for measuring
    @return: Time elapsed for the execution
    """
    # synchronize gpu time and measure fp

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        run_net(y, ["track_callback"])
        cl = []
        scl = []
        if clust_alg == "hcluster":
            cl, scl = hCluster_tracking(y, bg_tresh)
        elif clust_alg == "meanshift":
            cl, scl = meanShift_tracking(y, bg_tresh)
        else:
            cl, scl = hDBScan_tracking(y, bg_tresh)

        print(cl)
        print(scl)
    torch.cuda.synchronize()
    elapsed_fp = time.perf_counter() - t0

    return elapsed_fp


def measure_fps(model, dataset, bg_tresh, clust_alg):
    """
    Measure the time for executing the whole dataset

    @param model: The model for measuring
    @param dataset: The dataset for measuring
    @return: Time elapsed for the execution
    """
    dataset_first_and_last = []
    # synchronize gpu time and measure fp
    dataset_first_and_last.append(dataset[0][0::len(dataset[0]) - 1])
    # print(dataset_first_and_last)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for x in dataset_first_and_last:
            run_net(x, ["track_callback"])
            cl = []
            scl = []
            if clust_alg == "hcluster":
                cl, scl = hCluster_tracking(x, bg_tresh)
            elif clust_alg == "meanshift":
                cl, scl = meanShift_tracking(x, bg_tresh)
            else:
                cl, scl = hDBScan_tracking(x, bg_tresh)
            print(cl)
            print(scl)

    torch.cuda.synchronize()
    elapsed_fp = time.perf_counter() - t0

    return elapsed_fp


def benchmark_tracking(model, dataset, num_runs, num_warmup_runs, bg_tresh, clust_alg):
    """
    Bench the fps of the given model on the given dataset

    @param model: the model to bench
    @param dataset: DataLoader object
    @param num_runs: number of runs to test
    @param num_warmup_runs: number of warmup forwards before every test
    @return: list of time for executing a test
    """
    print('\nStarting warmup')
    # DRY RUNS
    x = None
    y = []
    # dataset as list of elements
    dataset_reformed = []

    for it in dataset:
        x = it
        break
    # Primi 2 elementi
    y.append(x[0][0])
    y.append(x[0][1])

    for i in range(len(dataset)):
        dataset_reformed.append([])
    print("Len dataset: ", len(dataset_reformed))
    i = 0
    for d in dataset:
        for dr in d:
            for j in range(len(dr)):
                # print(dr[i])
                dataset_reformed[i].append(dr[j])
        i += 1

    for i in tqdm(range(num_warmup_runs)):
        _ = measure_forward(model, y, bg_tresh, clust_alg)
    print(dataset_reformed)
    print('\nDone, now benchmarking')

    # START BENCHMARKING
    t_run = []
    for i in tqdm(range(num_runs)):
        for x in dataset_reformed:
            print("\nDATASET BATCH: ", i)
            # print(x)
            print('-----------\n')
        t_fp = measure_fps(model, dataset_reformed, bg_tresh, clust_alg)
        t_run.append(t_fp)

    # free memory
    del model

    return t_run


class Benchmarker:

    def __init__(self, model, dataset, batch_sizes, out_file, n_workers, seq_name, clust_alg):
        """
        Initialize a Benchmarker object

        @param model: the Pytorch model to test
        @param dataset: Dataset object where to execute the test
        @param batch_sizes: Batch sizes list for executing the test
        @param out_file: output file for saving the results
        @param n_workers: number of workers for loading the dataset examples
        """
        torch.manual_seed(1234)
        cudnn.benchmark = True

        # transfer the model on GPU
        self.model = model.cuda().eval()
        self.dataset = dataset
        self.out_file = out_file
        self.batch_sizes = batch_sizes
        self.n_workers = n_workers
        self.seq_name = seq_name
        self.clust_alg = clust_alg

    def bench_tracking(self, num_runs, num_warmup_runs, bg_tresh, clust_alg):
        """
        Bench the fps count of the model averaging through num_runs, output the results on the stdout
        and eventually on file

        @param num_runs: the number of runs to execute
        @param num_warmup_runs: warmup forward before a run
        """
        mean_fps = []
        nolist = []
        std_fps = []
        tmp_list = []
        for i, bs in enumerate(self.batch_sizes):
            print('\nBatch size is: ' + str(bs))
            print('--------------------------')
            data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=bs, shuffle=False, num_workers=self.n_workers
            )
            tmp = benchmark_tracking(model, data_loader, num_runs, num_warmup_runs, bg_tresh, clust_alg)
            print("Tempo di esecuzione con batch size " + str(bs) + ": " + str(np.asarray(tmp)))
            print("BG TRESH: ", str(bg_tresh))
            mean_fps.append((1 / (np.asarray(tmp) / len(self.dataset))).mean())
            std_fps.append((1 / (np.asarray(tmp) / len(self.dataset))).std())
            tmp_list.append(np.asarray(tmp))

        self.out_results({'mean (fps)': mean_fps, 'std (fps)': std_fps, 'tmp': tmp_list})

        # force garbage collection
        gc.collect()

    def out_results(self, dictionary):
        """
        Prints results on the stdout and on file if defined

        @param dictionary: dictionary of results
        """
        df = pd.DataFrame(dictionary, index=self.batch_sizes)
        # size = 'Input size: ' + str(self.dataset.shape())
        table = tabulate(df, headers='keys', tablefmt='psql')
        device = 'Device: ' + torch.cuda.get_device_name(torch.cuda.current_device())
        # result = "%s\n%s\n%s" % (device, size, table)
        result = "%s\n%s\n%s\n%s\n" % (
            "Results for " + self.seq_name, "Clustering algorithm: " + self.clust_alg, device, table)

        print(result)
        if self.out_file != 'none':
            with open(self.out_file, 'a') as f:
                f.write(result)

"""# **TRACKING WITH SCIPY.CLUSTER.HIERARCHY**"""

class CentroidTracker():

    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.video_IDs = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def initialize_videoIDs_list(self, nIDs):
        id = 0
        while (id < nIDs):
            coordinate_list = []
            self.video_IDs[id] = coordinate_list
            id += 1

    def calculateCentroids(self, img_path, bg_tresh, clust_alg):  # ,my_doc
        inputCentroids = []
        with h5py.File(img_path, 'r') as hf:
            img = hf.get('density')[:]
        # ONLY FOR GT
        # img = (img - img.min())/(img.max()-img.min())
        x = 0
        non_zero_coordinates = []
        while x < len(img):
            y = 0
            while y < len(img[x]):
                if img[x][y] != 0.0 and img[x][y] > bg_tresh:
                    # print(img[x][y])
                    non_zero_coordinates.append([y, x])
                y += 1
            x += 1

        print("LEN nzc: ", len(non_zero_coordinates))

        ndData = numpy.array([numpy.array(xi) for xi in non_zero_coordinates])

        if(clust_alg == 'hcluster'):
            # HCLUSTER CLUSTERING
            thresh = 80
            clusters = hcluster.fclusterdata(ndData, thresh, criterion="distance", metric="euclidean")

            try:
                clf = NearestCentroid()
                clf.fit(ndData, clusters)
                # print(clf.centroids_)
                for x in clf.centroids_:
                    inputCentroids.append((int(x[0]), int(x[1])))
            except:
                print("ERRORE NEL TRACKER")
                # my_doc.add_paragraph("Hcluster cannot be performed on " + img_path.split('/')[11] + " with bg_tresh: " + str(round(bg_tresh, 2)))
        elif(clust_alg == 'meanshift'):
            # MEANSHIFT CLUSTERING
            thresh = 70
            clusters = MeanShift(bandwidth=thresh, bin_seeding=True, cluster_all=False).fit(ndData)

            try:
                for x in clusters.cluster_centers_:
                    inputCentroids.append((int(x[0]), int(x[1])))
            except:
                print("ERRORE NEL TRACKER")
        elif(clust_alg == 'hdbscan'):
            # HDSCAN CLUSTERING
            thresh = 70
            clusters = hdbscan.HDBSCAN(metric='euclidean', cluster_selection_epsilon=thresh).fit(ndData)

            try:
                clf = NearestCentroid()
                clf.fit(ndData, clusters.labels_)
                # print(clf.centroids_)
                for x in clf.centroids_:
                    inputCentroids.append((int(x[0]), int(x[1])))
            except:
                print("ERRORE NEL TRACKER")

        return ndData, clusters, inputCentroids  # ,ss,db,ch,my_doc

    def calculateCentroidsShift(self, centroids):

        shift = OrderedDict()
        centroid_shifting_list = []
        for id in centroids.keys():
            if len(self.video_IDs[id]) == 2:
                del self.video_IDs[id][0]
                self.video_IDs[id].append(centroids[id])
            else:
                self.video_IDs[id].append(centroids[id])

        if len(self.video_IDs[id]) == 2:
            print("Video ids shift: ")
            print()

        for id in centroids.keys():
            shift["east"] = False
            shift["west"] = False
            shift["north"] = False
            shift["south"] = False

            if (len(self.video_IDs[id]) == 2):

                if (self.video_IDs[id][0][1] < self.video_IDs[id][1][1]):
                    shift["south"] = True
                elif (self.video_IDs[id][0][1] > self.video_IDs[id][1][1]):
                    shift["north"] = True
                if (self.video_IDs[id][0][0] < self.video_IDs[id][1][0]):
                    shift["east"] = True
                elif (self.video_IDs[id][0][0] > self.video_IDs[id][1][0]):
                    shift["west"] = True

                directions = []
                for s in shift:
                    if (shift[s]):
                        directions.append(s)
                directions.reverse()

                if (len(directions) == 0):
                    centroid_shifting_list.append("ID " + str(id) + " does not shift")
                elif (len(directions) == 1):
                    centroid_shifting_list.append("ID " + str(id) + " shift to " + directions[0])
                else:
                    centroid_shifting_list.append("ID " + str(id) + " shift to " + directions[0] + "-" + directions[1])
        return centroid_shifting_list

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(inputCentroids) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


"""## Run Hcluster for Computation time calculation"""


def hCluster_tracking(img_list_paths, bg_tresh):
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)
    ct.initialize_videoIDs_list(20)
    img_seq = 0
    centroid_shifting_list = []
    for img in img_list_paths:
        img_seq += 1
        print("-----------NEXT IMAGE--------------")
        print("Current image: " + img)

        # TO GENERATE GROUND_TRUTH
        # h5 = img.replace('.jpg', '.h5')
        h5 = img.replace('.jpg', '.h5').replace('images', 'predictions')
        ndData, clusters, inputCentroids = ct.calculateCentroids(h5, bg_tresh,'hcluster')

        objects = ct.update(inputCentroids)

        centroid_list = []
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            centroid_list.append(str(text) + ": " + str(centroid))

        try:
            centroid_shifting_list.append(ct.calculateCentroidsShift(ct.objects))
        except:
            print("ERROREEE")

    # return metrics
    return centroid_list, centroid_shifting_list


"""## Run Mean Shift for Computation time calculation"""


def meanShift_tracking(img_list_paths, bg_tresh):
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)
    ct.initialize_videoIDs_list(20)
    img_seq = 0
    centroid_shifting_list = []
    for img in img_list_paths:
        img_seq += 1
        print("-----------NEXT IMAGE--------------")
        print("Current image: " + img)

        # TO GENERATE GROUND_TRUTH
        # h5 = img.replace('.jpg', '.h5')
        h5 = img.replace('.jpg', '.h5').replace('images', 'predictions')
        ndData, clusters, inputCentroids = ct.calculateCentroids(h5, bg_tresh,'meanshift')  # , ss, db, ch, my_doc
        objects = ct.update(inputCentroids)

        centroid_list = []
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            # print(str(text) + ": " + str(centroid))
            centroid_list.append(str(text) + ": " + str(centroid))

        try:
            centroid_shifting_list.append(ct.calculateCentroidsShift(ct.objects))
        except:
            print("ERROREEE")

    # return metrics
    return centroid_list, centroid_shifting_list


"""## Run HDBScan for Computation time calculation"""


def hDBScan_tracking(img_list_paths, bg_tresh):
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)
    ct.initialize_videoIDs_list(20)
    img_seq = 0
    centroid_shifting_list = []

    for img in img_list_paths:
        img_seq += 1
        print("-----------NEXT IMAGE--------------")
        print("Current image: " + img)

        # TO GENERATE GROUND_TRUTH
        # h5 = img.replace('.jpg', '.h5')
        h5 = img.replace('.jpg', '.h5').replace('images', 'predictions')
        ndData, clusters, inputCentroids = ct.calculateCentroids(h5, bg_tresh,'hdbscan')  # , ss, db, ch, my_doc
        objects = ct.update(inputCentroids)

        labels = clusters.labels_
        centroid_list = []
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            # print(str(text) + ": " + str(centroid))
            centroid_list.append(str(text) + ": " + str(centroid))

        try:
            centroid_shifting_list.append(ct.calculateCentroidsShift(ct.objects))
        except:
            print("ERROREEE")

    # return metrics
    return centroid_list, centroid_shifting_list

"""# Esecuzione"""



def load_CC():
    cc = CrowdCounter([0], 'MobileCount')

    cc.load(
        'C:/Users/user/Downloads/uav_crowd_flow/uav-crowd-flow-detection/CrowdCounting-on-VisDrone2020-main/exp/09-02_17-32_VisDrone_MobileCount_0.001__1080x1920_CROWD_COUNTING_BS4/all_ep_58_mae_23.9_rmse_30.0.pth')
    return cc

models = {'CC': load_CC}

if __name__ == '__main__':
    clustering_alg_list = ['hcluster', 'hdbscan', 'meanshift']
    bg_tresh_list = [0.25, 0.50, 0.75]
    model = models['CC']()
    bs = literal_eval('[12]')
    in_size = literal_eval('(3, 1080, 1920)')

    path = '../dataset/test/images/'

    for sequence in os.listdir(path):
        sequence_path = path + sequence + "/"
        #clust_alg = "hcluster"
        for clust_alg in clustering_alg_list:
            for bg_tresh in bg_tresh_list:
                dataset = CustomDataset(sequence_path, in_size)

                with open('C:/Users/user/Downloads/uav_crowd_flow/uav-crowd-flow-detection/CrowdCounting-on-VisDrone2020-main/dataset/tracking_results.txt', 'a') as f:
                    f.write("\nTest con sequenza: " + sequence + " algoritmo: " + clust_alg + " e bg_treshold: " + str(
                        bg_tresh) + "\n")

                print("Test con sequenza: " + sequence)
                print('Model is loaded, start forwarding.')
                benchmarker = Benchmarker(model, dataset, bs, 'C:/Users/user/Downloads/uav_crowd_flow/uav-crowd-flow-detection/CrowdCounting-on-VisDrone2020-main/dataset/tracking_results.txt', 2,
                                          sequence, clust_alg)
                benchmarker.bench_tracking(1, 0, bg_tresh, clust_alg)
