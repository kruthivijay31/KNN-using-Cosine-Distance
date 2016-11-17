from __future__ import division
from collections import defaultdict, OrderedDict
import argparse
import time
import numpy
from functools import reduce
import math
from os.path import abspath, dirname, join

def read_docs(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
    return lines

def get_cosine_distance_for_docs(docs, eps, N):
    result= {}
    idf = defaultdict(list)
    df = defaultdict()
    nrows = 0
    for line in docs:
        nrows += 1
        line = list(map(int, line.lstrip(' ').rstrip('\n').split()))
        df[nrows] = OrderedDict(zip(line[::2],line[1::2]))
    for docId, d_split in df.items():
        #normalize the values by finding the square root of sum of squares
        norm = math.sqrt(sum(numpy.array(list(d_split.values()))**2))
        
        # divide each value by its normalized value and save it in same key
        for k, v in d_split.items():
            d_split[k] = v/ norm
    # Inverse document frequency
    for i in range(1, nrows+1):
        for k, v in df[i].items():
            idf[k].append((i,v))

    #for each value in the prefix pruned dictionary find the dot product
    cosine_dist = []
    for di, featfreq in df.items():
        dResult = defaultdict(lambda:0)
        
        #for each document id in inverted index find the dot prouct with every other document in original normalized list
        for feature in featfreq.keys():
            x = []
            for docId, freq in idf[feature]:
                x.append((docId, featfreq[feature]*freq))
            #Summation of dot product to calculate the cosine distance for normalized documents
            for j in x:
                if j and j[0]!=di:
                    dResult[j[0]] += j[1]
        #Get only values whose value is great than given threshold value i.e. eps
        cosine_dist.append(dict((k, v) for k, v in dResult.items() if v >= eps))
        for cosDist in cosine_dist:
            result[di] = dict(sorted(cosDist.items(), key=lambda kv: kv[1], reverse=True)[:N])
    return result


def write_output(result, outfile, doc_len):
    convert_result = []
    for i in range(doc_len):
        convert_result.append([])
    for di, resultDoc in result.items():
        for k, v in resultDoc.items():
            convert_result[di].append("{} {}".format(k, v))
    with open(outfile, 'w') as output:
        for i in range(doc_len):
            if i == 0:
                continue
            if not convert_result[i]:
                output.write("\n")
            else:
                output.write(" ".join(convert_result[i]))
                output.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('This script finds k similar documents.\n'
                     'usage: python findsim.py -eps <eps> -k <max_num> -infile <input-file> '
                     '-outfile <output_file>\n'))
    parser.add_argument('-infile', type=str, required=True)
    parser.add_argument('-outfile', type=str, required=True)
    parser.add_argument('-eps', type=str, required=True)
    parser.add_argument('-k', type=str, required=True)
    args = parser.parse_args()

    base_data_dir = dirname(abspath(__file__))
    input_file = join(base_data_dir, args.infile)
    output_file = join(base_data_dir, args.outfile)

    
    docs = read_docs(filename=input_file)
    start_time = time.time()
    result = get_cosine_distance_for_docs(docs=docs, eps=float(args.eps), N=int(args.k))
    print("--- %s seconds Total ---" % (time.time() - start_time))
    write_output(result=result, outfile=output_file, doc_len=len(docs) + 1)
    
