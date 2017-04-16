import random
import pickle
import os
import numpy as np
from statistics import mean
from math import sqrt
from dataloader import cut_qts_to_dictOnAuthor

#Origine Methodes
def pearson(v1, v2):
    #sum
    sum1 = sum(v1)
    sum2 = sum(v2)

    #sum of pow 2
    sum1Sq = sum([pow(v, 2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])

    #Product
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

    #Compute r
    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1,2)/len(v1)) * (sum2Sq - pow(sum2,2)/len(v2)))
    if den == 0: return 0
    return 1.0 - num/den
def euclid(v1, v2):
    np_v1 = np.array(v1)
    np_v2 = np.array(v2)
    euclid_distance = np.sqrt(np.sum((np_v1 - np_v2)**2))
    return euclid_distance/(euclid_distance + 1)

def prepareDataForKmean(author_word_counter):
    poetRowMapping, wordColMapping, processedPoetWordsList = [],[],[]
    wordPoetDict = transformPrefs(author_word_counter)
    for poet in author_word_counter:
        poetRowMapping.append(poet)
    for word in wordPoetDict:
        wordColMapping.append(word)
    for iRow in range(len(poetRowMapping)):
        processedPoetWordsList.append([])
        for iCol in range(len(wordColMapping)):
            if wordColMapping[iCol] in author_word_counter[poetRowMapping[iRow]]:
                processedPoetWordsList[iRow].append(author_word_counter[poetRowMapping[iRow]][wordColMapping[iCol]])
            else:
                processedPoetWordsList[iRow].append(0)
        if iRow % 100 == 0: print(str(iRow)+"/"+str(len(poetRowMapping)) + " processed.")
    return poetRowMapping, wordColMapping, processedPoetWordsList

def kcluster(rows, k=4, distance = pearson):
    #Calculate range for each word
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
              for i in range(len(rows[0]))]
    #Random creat k clusters
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))]
                for j in range(k)]

    lastMatches = None
    for t in range(150):
        print("Iteration %d" %t)
        bestMatches = [[] for i in range(k)]
        #Find the closet cluster for each line
        for iRow in range(len(rows)):
            row = rows[iRow]
            bestMatch, bestDistance = 0, 1
            for i in range(k):
                d = distance(clusters[i], row)
                if d < bestDistance :
                    bestDistance = d
                    bestMatch = i
            bestMatches[bestMatch].append(iRow)
        #If convergence
        if bestMatches == lastMatches:
            print("Converge break")
            break
        lastMatches = bestMatches

        #Move clusters
        for i in range(k):
            if len(bestMatches[i])>0:
                cList = []
                for idxRow in bestMatches[i]:
                    cList.append(rows[idxRow])
                clusters[i] = np.average(cList, axis=0).tolist()
        J = computeCostFunction(clusters, bestMatches, rows, distance)
    return bestMatches, J, clusters
def computeCostFunction(clusters, bestMatches, rows, distance):
    k = len(clusters)
    J, sumJ = 0, 0
    for idxCluster in range(k):
        J = 0
        cluster = clusters[idxCluster]
        for idxRow in bestMatches[idxCluster]:
            row = rows[idxRow]
            J += distance(cluster, row)
        if len(bestMatches[idxCluster]) != 0:
            J /= len(bestMatches[idxCluster])
        else:
            J = 0
        sumJ += J
    return sumJ/k
def cleanData(prefs, poet_list_to_keep, deleteSingleChar = True, minValue=0.01, maxValue=0.3):
    print("Cleaning Data...")
    wordcount = wordCount(prefs)
    fraclist = []
    i = 0
    for poet in prefs.copy():
        if poet not in poet_list_to_keep:
            del prefs[poet]
    for user in prefs:
        i += 1
        for word in prefs[user].copy():
            if deleteSingleChar and len(word) < 2:
                del prefs[user][word]
                continue
            frac = wordcount[word] / len(prefs)
            fraclist.append(frac)
            if frac < minValue or frac > maxValue:
                del prefs[user][word]
        if i % 10 == 0: print("%d/%d Done !" % (i, len(prefs)))
    print("Cleaning Data Done!")
    print("Max frac:" + str(max(fraclist)))
    print("Min frac:" + str(min(fraclist)))
    print("Avg frac:" + str(mean(fraclist)))
def wordCount(prefs):
    result = {}
    for person in prefs:
        for word in prefs[person]:
            result.setdefault(word, 0)
            result[word] += 1
    return result
def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result

def main():
    #Old Scripts func test
    author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                      "./data/processdWords.txt")

    poetRowMapping, wordColMapping, processedPoetWordsList = prepareDataForKmean(author_word_counter)

    bestMatches, J, clusters = kcluster(processedPoetWordsList)