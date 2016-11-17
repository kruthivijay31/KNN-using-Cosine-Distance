from __future__ import division
import argparse
import time
from functools import reduce
import math
from os.path import abspath, dirname, join
import numpy
from collections import OrderedDict, defaultdict
from itertools import dropwhile, islice
from operator import add, itemgetter
import heapq

# iterates the list of normalized documents and returns the first n documents
def take(n, iterable):
    return list(islice(iterable, n))

#Accumulator - this generator is used to find the prefix values that need to be pruned based on some threshold
def gen(x):
    iter1 = iter(x)
    curr = iter1.__next__()
    curr = curr**2
    for x in iter1:
        curr = curr + x**2
        yield math.sqrt(curr)

#Open file and read lines
def read_docs(filename):
    with open(filename, 'r', encoding="utf8") as f:
         lines = f.readlines()
    return lines

def get_cosine_distance_for_docs(docs, eps, N):
    result= defaultdict(list)
    fd = defaultdict(list)
    pfd = defaultdict(list)
    start = time.time()
    l_feature_freq = []
    i=1
    prefix_freq = []
    docNorm = defaultdict(lambda:0.0)
    l = 0;
    for line in docs:
        line = list(map(int, line.lstrip(' ').rstrip('\n').split()))
        d_split = OrderedDict(zip(line[::2],line[1::2]))
        
        #normalize the values by finding the square root of sum of squares
        norm = math.sqrt(sum(numpy.array(list(d_split.values()))**2))
        docNorm[l] = norm
        l += 1
        # divide each value by its normalized value and save it in same key
        for k, v in d_split.items():
            d_split[k] = v/ norm
            
        #Sort the dictionry in descending order such that high frequency terms are at the starting of dict
        d = OrderedDict(sorted(d_split.items(), key=lambda kv: kv[1], reverse=True))
        #get the length of dict
        dLen = len(d)
        
        #get the length of suffix
        dSimTLen = len(list(dropwhile(lambda x: x<eps, gen(list(d.values()))))) 
        
        #Remove the prefix values from the dictionary
        pd = defaultdict(lambda:0.0)
        for k in take(dLen-dSimTLen, d):
            pd[k] = d[k]
            del d[k]
        prefix_freq.append(pd)
        #create a inverted index
        for k, v in d.items():
            fd[k].append((i,v))
        
        for k,v in pd.items():
            pfd[k].append((i,v))
            
        #increment the counter which is the document id
        i+=1
        
        #append the prefix pruned values to feature frequency list
        l_feature_freq.append(d)
    
    finaldf = defaultdict(defaultdict)
    #for each value in the prefix pruned dictionary find the dot product
    dResult = []
    for di, val in enumerate(l_feature_freq):
    
        #default key value will be Zero
        dResultInner = defaultdict(lambda:0.0)
        
        #for each document id in inverted index find the dot prouct with every other document in original normalized list
        for i in val.keys():
            x=[]
            x= list(map(lambda kv : (kv[0], val[i]*kv[1]), fd[i]))
            for j in x:
                if j and j[0]!=di+1:
                    dResultInner[j[0]] += j[1]
        dResult.append(dResultInner)
    
    FinalResult = defaultdict(list)
    for i, SufSim in enumerate(dResult):
        prefixNeighbors = defaultdict(lambda:False)
        for k, v in prefix_freq[i].items():
            for k, v in pfd[k]:
                if k is not i + 1:
                    prefixNeighbors[k] = True
        prefixDocNorm = math.sqrt(sum(numpy.array(list(prefix_freq[i].values()))**2))
        for doc, suffixCos in SufSim.items():
            prefixNeighDocNorm = math.sqrt(sum(numpy.array(list(prefix_freq[doc-1].values()))**2))
            if (doc - 1) in prefixNeighbors:
                del prefixNeighbors[doc - 1]
            cosineResult = suffixCos + prefixDocNorm * prefixNeighDocNorm
            if cosineResult >= eps:
                FinalResult[i + 1].append((doc, cosineResult))
        
        for k in prefixNeighbors.keys():
            prefixNeighDocNorm = math.sqrt(sum(numpy.array(list(prefix_freq[k - 1].values()))**2))
            cosineResult = prefixDocNorm * prefixNeighDocNorm
            if cosineResult >= eps:
                FinalResult[i + 1].append((k, cosineResult))
        for doc, fresult in FinalResult.items():
            result[doc] = dict(sorted(fresult, key=lambda kv: kv[1], reverse=True)[:N])
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
    
