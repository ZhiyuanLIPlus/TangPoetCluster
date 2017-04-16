import random
import pickle
import os
import numpy as np
from statistics import mean
from dataloader import cut_qts_to_dictOnAuthor,load_all_poet
from math import sqrt


class KMeansCluster(object):
    def __init__(self):
        self.poetRowMapping, self.wordColMapping, self.processedPoetWordsList = [],[],[]
    #Tool Functions
    def pearson(v1, v2):
        # sum
        sum1 = sum(v1)
        sum2 = sum(v2)

        # sum of pow 2
        sum1Sq = sum([pow(v, 2) for v in v1])
        sum2Sq = sum([pow(v, 2) for v in v2])

        # Product
        pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

        # Compute r
        num = pSum - (sum1 * sum2 / len(v1))
        den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v2)))
        if den == 0: return 0
        return 1.0 - num / den
    def euclid(v1, v2):
        np_v1 = np.array(v1)
        np_v2 = np.array(v2)
        euclid_distance = np.sqrt(np.sum((np_v1 - np_v2) ** 2))
        return euclid_distance / (euclid_distance + 1)
    def cleanData(self, prefs, poet_list_to_keep, deleteSingleChar = False, minValue=0.1, maxValue=0.5):
        print("Cleaning Data...")
        wordcount = self.wordCount(prefs)
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
    def wordCount(self, prefs):
        result = {}
        for person in prefs:
            for word in prefs[person]:
                result.setdefault(word, 0)
                result[word] += 1
        return result
    def transformPrefs(self, prefs):
        result = {}
        for person in prefs:
            for item in prefs[person]:
                result.setdefault(item, {})
                result[item][person] = prefs[person][item]
        return result
    def prepareDataForKmean(self, author_word_counter):
        poetRowMapping, wordColMapping, processedPoetWordsList = [],[],[]
        wordPoetDict = self.transformPrefs(author_word_counter)
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
            if iRow % 10 == 0: print(str(iRow)+"/"+str(len(poetRowMapping)) + " processed.")
        self.poetRowMapping = poetRowMapping
        self.wordColMapping = wordColMapping
        self.processedPoetWordsList = processedPoetWordsList
    def computeCostFunction(self, clusters, bestMatches, distance):
        rows = self.processedPoetWordsList
        k = len(clusters)
        J = 0
        for idxCluster in range(k):
            cluster = clusters[idxCluster]
            for idxRow in bestMatches[idxCluster]:
                row = rows[idxRow]
                J += distance(cluster, row)
        return J
    def doKcluster(self, k=4, distance = pearson):
        rows = self.processedPoetWordsList
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
            J = self.computeCostFunction(clusters, bestMatches, distance)
        return bestMatches, J, clusters
    def printClusterResult(self, clusterResult, all_poet_list):
        i = 0
        for poetList in clusterResult:
            i += 1
            print("******** Cluster No.%d ********" %i)
            for poetIdx in poetList:
                print(self.poetRowMapping[poetIdx] + ":"+ str(all_poet_list[self.poetRowMapping[poetIdx]]))
            print("********     End      ******** ")

def main():
    #New wrapped class test
    author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                      "./data/processdWords.txt")
    all_poet_list = load_all_poet("./data/rawdata/early_tang_poets.txt",
                                  "./data/rawdata/high_tang_poets.txt",
                                  "./data/rawdata/middle_tang_poets.txt",
                                  "./data/rawdata/late_tang_poets.txt")

    KMinstance = KMeansCluster()
    KMinstance.cleanData(author_word_counter, all_poet_list, deleteSingleChar = False)
    KMinstance.prepareDataForKmean(author_word_counter)
    min_J = 0
    min_i = -1
    dict_i_result = {}
    for i in range(10):
        bestMatches, J, clusters = KMinstance.doKcluster(k = 30)
        dict_i_result[i] = bestMatches
        if J > min_J:
            min_J = J
            min_i = i
    print("------- Lowest J in 10 tries: %f --------" %J)
    KMinstance.printClusterResult(dict_i_result[min_i], all_poet_list)
if __name__ == '__main__':
    main()