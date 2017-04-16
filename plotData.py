# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import pickle
from func import *
from dataloader import cut_qts_to_dictOnAuthor, load_all_poet

def print_counter(counter, l = 0, n = 20):
    i = 0
    for k, v in counter:
        if i == n: break
        if len(k)>l:
            print(k, v)
            i += 1

def computeKJRanges():
    datapath = "./data/KJRanges.pkl"
    if os.path.exists(datapath):
        print('find preprocessed KJRanges, loading directly...')
        with open(datapath, 'rb') as f:
            kRanges, jRanges = pickle.load(f)
    else:
        author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                          "./data/processdWords.txt")
        poetRowMapping, wordColMapping, processedPoetWordsList = prepareDataForKmean(author_word_counter)

        #K ranges
        kRanges = [i for i in range(2,120,4)]
        jRanges = []
        for k in kRanges:
            print("Compute for K:" + str(k))
            bestMatches, J, clusters = kcluster(processedPoetWordsList, k=k)
            jRanges.append(J)
            print ("K:J -> " + str(k) + ":" + str(J))
        dumped_data = [kRanges, jRanges]
        with open(datapath, 'wb') as f:
            pickle.dump(dumped_data, f)
    return kRanges, jRanges

def drawKJRanges():
    #Compute J K Plot
    kRanges, jRanges = computeKJRanges()
    #Plot figure
    plt.plot(kRanges, jRanges, '--ro')
    plt.show()

def drawWordsDistribution():
    author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                      "./data/processdWords.txt")
    all_poet_list = load_all_poet("./data/rawdata/early_tang_poets.txt",
                                  "./data/rawdata/high_tang_poets.txt",
                                  "./data/rawdata/middle_tang_poets.txt",
                                  "./data/rawdata/late_tang_poets.txt")
    cleanData(author_word_counter, all_poet_list, deleteSingleChar=False, minValue=0.1, maxValue=0.5)
    wordcount = wordCount(author_word_counter)
    wordFreCount = {}
    for word in wordcount:
        wordFreCount.setdefault(wordcount[word], 0)
        wordFreCount[wordcount[word]] += 1
    XasWordCounts = [i for i in range(len(author_word_counter))]
    YasWordFrequency = []
    for i in XasWordCounts:
        if i in wordFreCount:
            YasWordFrequency.append(wordFreCount[i])
        else:
            YasWordFrequency.append(0)
    plt.plot(XasWordCounts, YasWordFrequency)
    plt.ylabel('Frequency')
    plt.xlabel('Numbers of poet who used the words')
    plt.show()

    for i in range(15):
        YasWordFrequency[i] = 0
    plt.plot(XasWordCounts, YasWordFrequency)
    plt.ylabel('Frequency')
    plt.xlabel('Numbers of poet who used the words')
    plt.show()

def checkKJPlot():
    author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                      "./data/processdWords.txt")
    all_poet_list = load_all_poet("./data/rawdata/early_tang_poets.txt",
                                  "./data/rawdata/high_tang_poets.txt",
                                  "./data/rawdata/middle_tang_poets.txt",
                                  "./data/rawdata/late_tang_poets.txt")
    cleanData(author_word_counter, all_poet_list, deleteSingleChar=False, minValue=0.1, maxValue=0.4)
    poetRowMapping, wordColMapping, processedPoetWordsList = prepareDataForKmean(author_word_counter)
    # K ranges
    kRanges = [i for i in range(2, 40, 2)]
    kRanges.extend([50,60,75,90,120])
    jRanges = []
    for k in kRanges:
        print("Compute for K:" + str(k))
        bestMatches, J, clusters = kcluster(processedPoetWordsList, k=k)
        jRanges.append(J)
        print("K:J -> " + str(k) + ":" + str(J))
    plt.plot(kRanges, jRanges, '--ro')
    plt.xlabel('K')
    plt.ylabel('Cost Function J')
    plt.show()

def cleanDataPrez():
    author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt",
                                                                                      "./data/processdWords.txt")
    print("--- Before Clean Data ---")
    print_counter(author_word_counter[u'岑参'].most_common(1000))
    cleanData(author_word_counter,True)
    print("--- After Clean Data ---")
    print_counter(author_word_counter[u'岑参'].most_common(1000))

if __name__ == '__main__':
    #drawWordsDistribution()
    #cleanDataPrez()
    #drawKJRanges()
    checkKJPlot()