import graph
import random

#Кластеризация графов
#Markov clustering
#http://micans.org/mcl/lit/svdthesis.pdf.gz
#http://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
#http://programmers.stackexchange.com/questions/130840/efficient-graph-clustering-algorithm

#PHD Thesis 

#http://www.stat.berkeley.edu/~aldous/RWG/book.html
#Clustring area

'''Приминение кластеризации - Markov clustering algorithm - (application TRIBEMCL)
an approach for detecting protein families within large networks of biological data
'''

''' protein complex prediction
problem  стр 37 Graph clustering Restricted Neighbourhood Search'''

#Image segmentation

#Type of algorithm - LOcal search clustering 

'''RNSC algorithm stages
1. Either read or randomly generate an initial clustering
2. Apply the naive cost function to the clustering and data structures. Attempt to
minimize the naive cost by modifying the clustering one move at a time, reaching
a best naive clustering
3. Do the same for the scaled cost: Starting with the naive clustering Cn , apply the
scaled cost function to the clustering and data structures and attempt to minimize
the scaled cost by making one move at a time. The best scaled clustering, the
output of the experiment, is denoted Cs 

link to www.cs.utoronto.ca/~juris/data/rnsc
'''

# стр 42
class RNSC:
    def __init__(self, sgraph):
        self.exper = 1
        self.sgraph = sgraph
        self.clusters={}
        #Вычисление лучшей кластеризации

    def _generateClusters(self):
        #Initial clusters
        BestCost = 99999
        def initial():
            if len(BestCost) > 0:
                #Make a non-tabunear-optimal move
                #Next, new cluster
                cluster = new_cluster(4)
                if cluster < BestCost:
                    pass
                else: _generateClusters()

        def new_cluster(ident):
            self.clusters[ident] = sgraph[part]

    def _readCluster(self, c0):
        return self.clusters[c0] if c0 in self.clusters else None

    #Run naive scheme
    #Run naive clustering 
    def run_sceheme(self):
        num_moves=0
        DivCount = 0
        BestCost = 999999


    def naive_clustering(self):
        #Run scaled scheme
        #Scaled clustring
        #Is Cs is best clustering
        self.exper += 1


class Clustering:
    def __init__(self, sgraph):
        self.sgraph = sgraph

    def fit(self, nums=1, iters=1000, **kwargs):
        pass

    def markov(self, pgraph):
        pass


#http://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
class MarkovClustering:
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.steps={}

    def _markov_chains(self):
        n = numpy.matrix(self.sgraph.size())
        for nodes in len(self.sgraph):
            prob = 100/len(nodes.connected)
            for write_prob in self.graph:
                if write_prob in nodes.connected:
                    {write_prob: prob}
                else:
                    {write_prob: 0}


#just a fun
def inner_python():
    def inner1():
        return inner2()

    def inner2():
        return 'Foo'
    return inner1()


def inner_python2(s):
    def inner1(f):
        return 'Run inner1 ', f
    def inner2(f):
        return 'Run inner2 ', f
    def inner3(f):
        return 'Run inner3 ', f
    return list(map(lambda x:x(s), [inner1, inner2, inner3]))



#print (inner_python())
print(inner_python2(4))