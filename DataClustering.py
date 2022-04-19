#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sami

Student ID: 201348278
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#Function to parse cluster data into list of tuples (fname, centroidInd, object)
def readData(fname):
    with open(fname) as file:
        data = []
        for line in file:
            currentObject = line.strip().split(' ')
            currentObject.pop(0)
            currentObject = np.asarray(currentObject, dtype=np.float64, order='C')
            data.append((fname,0,currentObject))
            
    return data

#Returns the squared euclidean distance between X and Y
#For the k-means algorithm
def sqrdEuclidDistance(X,Y):
    #Return the Euclidean distance between X and Y
    return np.linalg.norm(X-Y)**2

#Returns the l1/Manhattan distance between objects X and Y
#For the k-medians algorithm
def l1distance(X,Y):
    
    return np.linalg.norm(X-Y, ord=1)

#Given dataset and indices of centroids, the function updates the clusters of the objects in the dataset
#based on the algorithm passed as a parameter
def assign(algorithm, centroids, dataset):
    numOfObjects = len(dataset)  
    k = len(centroids)
    
    #Go through every object in the dataset
    for i in range(numOfObjects):
        #Current object
        X = dataset[i][2]
        centroidIndOfX = -1
        distanceToClosestCentroid = np.Inf
        #Find the closest centroid for the current object
        #by going through all the centroids and finding
        #the centroid that gives the smallest distance
        for j in range(k):
            currentCentroid = centroids[j]
            #Computed the squared euclidean distance if the algorithm
            #passed is the k-means algorithm
            if(algorithm == "k-means"):
                dist = sqrdEuclidDistance(X, currentCentroid)
            #Computed the squared l1 distance if the algorithm
            #passed is the k-medians algorithm
            elif(algorithm == "k-medians"):
                dist = l1distance(X, currentCentroid)
                
            #Check if the distance is smaller
            if dist < distanceToClosestCentroid:
                #Found closer centroid. Store information about it
                distanceToClosestCentroid = dist
                centroidIndOfX = j
                    
        #assign to object its closest centroid
        dataset[i][1] = centroidIndOfX

#Function to filter the dataset given a centroid        
def filterByCentroid(dataset, currCentroid):
    filteredData = []
    for x in dataset:
        if(currCentroid == x[1]):
            filteredData.append(x)
            
    return filteredData
        

#Implementation of both the clustering algorithms: k-means and k-medians
#dependant on the parameter: algorithm
def clusteringAlgorithm(algorithm, k, dataset, maxIter=10):        
    #Generate indices for initial centroids
    seed = random.randint(0, len(dataset))
    ### Enter own seed for testing! ###
    np.random.seed(10)
    centroidInds = np.random.choice(len(dataset), k, replace=False)
    centroidInds = np.asarray(centroidInds, dtype=np.float64, order='C')
    
    #Use generated indices to retrieve the objects that correspond to them in the dataset
    centroids = []
    for k in range(len(centroidInds)):
        for i in range(len(dataset)):
            if(int(centroidInds[k]) == i):
                centroids.append(dataset[i][2])
                
    #Make initial assignment of objects to the clusters
    assign(algorithm, centroids, dataset)
    
    #Repeat algorithm for maxIter iterations
    for i in range(maxIter):
        #Go through every cluster and compute new centroid
        #for the cluster based on the objects in the cluster
        for y in range(len(centroids)):
            currObjsInCentroid = filterByCentroid(dataset, y)   
            #Retrieve just the features from the objects
            currObjsInCentroid = [features for (fname, centroid, features) in currObjsInCentroid]
            
            newCentroid = 0
            #Compute new centroid based on algorithm
            if(algorithm == "k-means"):
                newCentroid = np.mean(currObjsInCentroid)
            elif(algorithm == "k-medians"):
                newCentroid = np.median(currObjsInCentroid)
            centroids[y] = newCentroid
        #Update assignment of objects to clusters
        assign(algorithm, centroids, dataset)
        
    
    return (dataset, centroids)
            
#Function that returns how many objects
#with the label passed as a parameter are
#in the dataset            
def sameLabel(label, dataset):
    count = 0
    for (fname, centroidInd, features) in dataset:
        if(label == fname):
            count+=1
    return count


#Function that runs a clustering algorithm with k from 1 to 9
#for questions 3 to 6
def runClustering(algorithm, isl2Norm, dataset, maxIter=10):
    if(isl2Norm):       
        normalisation = "l2 normalised"
    else :
        normalisation = "unnormalised"
        
    print("Running: ", algorithm, "with k from 1 to 9 ", normalisation)
    
    dataForPlot = []
    for k in range(1,10):
        print("k: ", k)
        #Get centroids and resulting clustered dataset
        (clusteredDataset, centroids) = clusteringAlgorithm(algorithm, k, dataset, maxIter)
        #Compute B-CUBED precision, recall and fscore for every object in the clustered dataset
        precisions = []
        recalls = []
        fscores = []
        #Go through every cluster
        for i in range(len(centroids)):
            currentCluster = filterByCentroid(clusteredDataset, i)
            for (fname, centroidInd, data) in currentCluster:
                precision = sameLabel(fname, currentCluster)/len(currentCluster)
                precisions.append(precision)
                recall = sameLabel(fname, currentCluster)/len(clusteredDataset)
                recalls.append(recall)
                fscore = (2 * precision * recall)/ (precision + recall)
                fscores.append(fscore)
        b_cubed_precision = np.average(precisions)
        b_cubed_recall = np.average(recalls)
        b_cubed_fscore = np.average(fscores)
        dataForPlot.append((k,b_cubed_precision,b_cubed_recall,b_cubed_fscore))

    
        print("B_CUBED Precision: ", b_cubed_precision)
        print("B_CUBED Recall: ", b_cubed_recall)
        print("B_CUBED F-Score: ", b_cubed_fscore)
        
    print()
    
    #Get data in a format so we can plot
    kValues = [k for (k, bprecision, brecall, fscore) in dataForPlot]
    plotPrecisions = [bprecision for (k, bprecision, brecall, fscore) in dataForPlot]
    plotRecalls = [brecall for (k, bprecision, brecall, fscore) in dataForPlot]
    plotFscores = [fscore for (k, bprecision, brecall, fscore) in dataForPlot]
    
    subPlots = [131,132,133]   
    yLabels = ["B-CUBED Precision", "B-CUBED Recall", "B-CUBED F-Scores"] 
    xData = [plotPrecisions, plotRecalls, plotFscores]   
    plotData = zip(subPlots, yLabels, xData)   
    x_ticks = np.arange(1,10,1)       
    plt.figure()   
    title = algorithm + ", with k from 1 to 9 " + normalisation
    plt.suptitle(title)

    for (subPlot, yLabel, x) in plotData:        
        plt.subplot(subPlot)
        plt.tight_layout(pad=1.0)
        plt.xlabel("k")
        plt.ylabel(yLabel)
        plt.xticks(x_ticks)
        plt.scatter(kValues, x)

""" Read data from text files """

fruitsData = readData("fruits")

countriesData = readData("countries")

animalsData = readData("animals")

veggiesData = readData("veggies")

theDataset = np.concatenate((fruitsData, countriesData, animalsData, veggiesData))

#Dataset where each object is normalised to unit l2 length.
l2NormDataset = np.asarray([(fname, centroid, np.asarray([(x/np.linalg.norm(features)) for x in features])) for (fname, centroid, features) in theDataset], dtype=object)

""" Run program to get the results  """

print("Question 3 \n")
runClustering("k-means", False, theDataset)

print("Question 4 \n")
runClustering("k-means", True, l2NormDataset)

print("Question 5 \n")
runClustering("k-medians", False, theDataset)

print("Question 6 \n")
runClustering("k-medians", True, l2NormDataset)