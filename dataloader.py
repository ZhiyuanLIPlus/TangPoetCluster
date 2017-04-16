# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
import thulac
import pickle
import os
import io
def print_counter(counter, l = 1, n = 20):
    i = 0
    for k, v in counter:
        if i == n: break
        if len(k)>l:
            print(k, v)
            i += 1

def cut_qts_to_dictOnAuthor(qts_file, saved_words_file):
    save_dir = os.path.dirname((saved_words_file)) #已经处理过的唐诗分词文件
    dumped_file = os.path.join(save_dir, 'qts_words_stat_result.pkl')

    if os.path.exists(dumped_file) and os.path.exists(saved_words_file):
        print('find preprocessed data, loading directly...')
        with open(dumped_file, 'rb') as f:
            char_counter, author_counter, vocab, author_word_counter, author_genre_counter = pickle.load(f)
    else:
        char_counter = Counter()  # 字频统计
        author_counter = Counter()  # 每个作者的写诗篇数
        vocab = set()  # 词汇库
        #word_counter = Counter()  # 词频统计
        author_word_counter = defaultdict(Counter)  # 针对每个作者用词频度的Counter
        author_genre_counter = defaultdict(Counter) # 针对每个作者词性频度的Counter

        fid_save = open(saved_words_file, 'w')
        lex_analyzer = thulac.thulac()  # 分词器
        line_cnt = 0
        with io.open(qts_file,encoding="utf-8") as f:
            for line in f:
                text_segs = line.split()
                author = text_segs[2]
                author_counter[author] += 1

                poem = text_segs[-1]
                # 去除非汉字字符
                #valid_char_list = [c for c in poem if u"\u4E00" <= c <= u"\u9fff" or c == '，' or c == '。']
                valid_char_list = [c for c in poem if u'\u4e00' <= c <= u'\u9fff' or c == u'，' or c == u'。']
                for char in valid_char_list:
                  char_counter[char] += 1

                regularized_poem = ''.join(valid_char_list)
                word_genre_pairs = lex_analyzer.cut(regularized_poem)

                word_list = []
                for word, genre in word_genre_pairs:
                    word_list.append(word)
                    vocab.add(word)
                    #word_counter[word] += 1
                    #genre_counter[genre][word] += 1
                    author_word_counter[author][word] += 1
                    author_genre_counter[author][genre] += 1
                save_line = ' '.join(word_list)
                fid_save.write(save_line + '\n')

                if line_cnt % 10 == 0:
                  print('%d poets processed.' % line_cnt)
                line_cnt += 1

        fid_save.close()
        # 存储下来
        dumped_data = [char_counter, author_counter, vocab, author_word_counter, author_genre_counter]
        with open(dumped_file, 'wb') as f:
            pickle.dump(dumped_data, f)
    #return char_counter, author_counter, genre_counter
    print("Calculate/Load Data Done!")
    return author_word_counter, author_genre_counter, char_counter

def load_poet_list(list_path, type):
    poet_list = {}
    with io.open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            poet_list[line] = type
    return poet_list
def load_all_poet(early_path, high_path, middle_path, late_path):
    all_poet_list = load_poet_list(early_path, 1)
    all_poet_list.update(load_poet_list(high_path, 2))
    all_poet_list.update(load_poet_list(middle_path, 3))
    all_poet_list.update(load_poet_list(late_path, 4))
    return all_poet_list
def main():
    #author_word_counter, author_genre_counter, char_counter = cut_qts_to_dictOnAuthor("./data/qts_zhs.txt", "./data/processdWords.txt")
    #print(len(author_word_counter[u'岑参']))
    #keys_a = set(author_word_counter[u'岑参'].keys())
    #keys_b = set(author_word_counter[u'王维'].keys())
    #intersection = keys_a & keys_b
    #print (len(intersection))
    #print_counter(author_word_counter[u'李贺'].most_common(1000),1)
    #print_counter(char_counter.most_common(200),0, 200)
    #print_counter(author_word_counter[u'王昌龄'].most_common(1000),1)
    all_poet_list = load_poet_list("./data/rawdata/early_tang_poets.txt", 1)
    all_poet_list.update(load_poet_list("./data/rawdata/high_tang_poets.txt", 2))
    all_poet_list.update(load_poet_list("./data/rawdata/middle_tang_poets.txt", 3))
    all_poet_list.update(load_poet_list("./data/rawdata/late_tang_poets.txt", 4))

if __name__ == '__main__':
    main()

