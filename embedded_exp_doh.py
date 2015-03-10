from __future__ import division
from numpy import *
from graph import *
from math import *
from copy import deepcopy
import pycallgraph
from graphviz import *
from random import randint
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import matplotlib.pyplot as plt



def create_Random_graph(p,n):
    g = Graph_()
    text_file = open("random.txt","r")
    lines = text_file.readlines()
    t =  float(lines[0][2:])
    #n = int(lines[1][2:])
    #print "yahs " + str(n)

    for i in range (0,n):
        g.addVertex(i)

    #dot = Graph(comment='create')
    #for i in range(0,n):
    #     dot.node(str(i), str(i))

    Matrix2 = [[0 for x in range(n)] for x in range(n)]

    for i in range(0,n):
        for j in range(0,n):
            chance = random.uniform(0, 1)
            #print "chance = "+str(chance)
            #print "p = "+str(p)
            if chance<=p:
                if Matrix2[j][i]==0 and i!=j:
                    Matrix2[i][j]=1
                    #dot.edge(str(i),str(j))
                    g.addEdge(i,j)



    text_file.close()
    #print(dot.source)
    #dot.render('Mini-Project-output/Random.gv', view=False)
    #'Mini-Project/round-table.gv.pdf'

    return g,Matrix2



def create_graph():

    g = Graph_()
    text_file = open("random.txt","r")
    lines = text_file.readlines()
    t =  float(lines[0][2:])
    n = int(lines[1][2:])
    #print "yahs " + str(n)

    for i in range (0,n):
        g.addVertex(i)

    dot = Graph(comment='create')

    for i in range(0,n):
         dot.node(str(i), str(i))

    Matrix2 = [[0 for x in range(n)] for x in range(n)]
    ## create edges

    # we randomly pick a number between 0 to n, for how many neighbors
    for vertex in range (0,n):

        num_of_neighbors =randint(0,n-1)     ## Here we can play with the degree of each node!!
        s = set()

        while len(s) < num_of_neighbors:
                # we randomly pick a number between 0 to n, for who are the neighbors for each vertex
                neighbor = randint(0,n-1)
                if vertex!= neighbor:
                    s.add(neighbor)

        for neighbor in s:

                    #build adjacency matrix for the random graph
                    if Matrix2[neighbor][vertex]==0:
                             Matrix2[vertex][neighbor]=1

    #build G from the adjacency Matrix
    for i in range(0,n):
        for j in range(0,n):
            if Matrix2[i][j]==1:
                dot.edge(str(i),str(j))
                g.addEdge(i,j)


    text_file.close()
    print(dot.source)
    dot.render('Mini-Project-output/Random.gv', view=False)
    'Mini-Project/round-table.gv.pdf'

    return g,Matrix2





def show_graphs(g,t_spanner,Matrix2,n):

    #### Original Graph ####
    removed =0
    text_file = open("random.txt","r")
    lines = text_file.readlines()
    t =  float(lines[0][2:])
    #n = int(lines[1][2:])

    #dot = Graph(comment='original')
    #for i in range(0,n):
    #     dot.node(str(i), str(i))

    for line in lines[2:]:
        l,r = line.split(",")
        i =int(l)
        j= int(r)
    #        dot.edge(str(i),str(j))

    text_file.close()
    #print(dot.source)
    #dot.render('Mini-Project-output/Data-Graph.gv', view=False)
    #'Mini-Project/round-table.gv.pdf'

    #### T-Spanner ####

    #dot = Graph(comment='T-Spanner')
    #for i in range(0,n):
    #    dot.node(str(i), str(i))

    Matrix = [[0 for x in range(n)] for x in range(n)]

    for key,value in t_spanner.vertList.iteritems():
        for neighbor in value.connectedTo.values():
            if Matrix[neighbor][key]==0:
                     Matrix[key][neighbor]=1

    for i in range(0,n):
        for j in range(0,n):
            if Matrix[i][j]==1:
                pass
                #dot.edge(str(i),str(j))


    #print(dot.source)
    #dot.render('Mini-Project-output/T-Spanner-Graph.gv', view=False)
    #'Mini-Project/round-table.gv.pdf'

    #### Comparison-Graph ####

    #dot = Graph(comment='Comparison')
    #for i in range(0,n):
    #     dot.node(str(i), str(i))

    Matrix = [[0 for x in range(n)] for x in range(n)]

    for key,value in t_spanner.vertList.iteritems():
        for neighbor in value.connectedTo.values():
            if Matrix[neighbor][key]==0:
                     Matrix[key][neighbor]=1

    for i in range(0,n):
        for j in range(0,n):
            if Matrix[i][j]==1:
                pass
                #dot.edge(str(i),str(j))




            else:
                if Matrix[i][j]==0 and Matrix[j][i]==0:
                    if (Matrix2[i][j] ==1):
                        removed += 1
                        #dot.edge(str(i),str(j),color = "red",style="dashed")




    #print(dot.source)
    #dot.render('Mini-Project-output/Comparison-Graph.gv', view=False)
    #'Mini-Project/round-table.gv.pdf'
    '''
    text_file = open("data.txt","r")
    lines = text_file.readlines()
    t =  int(lines[0][2:])
    n = int(lines[1][2:])
    dot = Graph(comment='Comparison-Graph')
    for i in range(0,n):
         dot.node(str(i), str(i))

    for key,value in t_spanner.vertList.iteritems():
        for neighbor in value.connectedTo.values():
            if Matrix[neighbor][key]==0:
                     Matrix[key][neighbor]=1

    text_file.close()
    print(dot.source)
    dot.render('Mini-Project-output/Comparison-Graph.gv', view=True)
    'Mini-Project/round-table.gv.pdf'
    '''
    return removed

def bfs(graph, start):

    start.parent= start.getId()
    visited, queue = set(), [start]
    #start.parent=start
    graph.vertList[start.getId()].parent = start

    while queue:
            vertex = queue.pop(0)


            if vertex not in visited:

                 visited.add(vertex)
                 list_visited = list(visited)
                 # connectionts_to_send = graph.vertList[vertex.getId()].getConnections()      # vertex.getConnections()

                 connectionts_to_send=[]
                 for k in graph.vertList[vertex.getId()].getConnections():
                     connectionts_to_send.append(graph.vertList[k.getId()])

                 for visitor in list_visited:
                     if visitor in connectionts_to_send:
                         connectionts_to_send.remove(visitor)


                 for t in connectionts_to_send:
                     if t.parent.getId() == t.getId():
                         t.parent=vertex
                         #graph.vertList[t.getId()].parent=t


                 queue.extend(connectionts_to_send)

    return visited




def compute_distances_to_v(graph,v):

    for key,value in graph.vertList.iteritems():

        dist=1
        temp = value.parent

        if temp==value:
                 dist = -1
        else:
                while ( temp != v ):
                    #print value.parent
                    dist += 1
                    temp=temp.parent

        value.distance_from_vertex =dist
        v.distance_from_vertex=0






global_covered = []
edges_between_clusters = [[0 for x in range(5)] for x in range(5)]


def find_the_next_vertex(optional_vertexes):
    if not optional_vertexes:
        return -1
    v_i = optional_vertexes[0]

    return  v_i




def create_clusters(graph,clusters,index,t,n,new_cluster_flag):

    global global_covered

    # no more uncovered vertices left
    if (len(global_covered)==n):
        print("End here Pal!")
        return clusters

    if new_cluster_flag==1:

             #  -- PICK A NEW UNCOVERED VERTEX TO START A NEW CLUSTER --

            optional_vertexes = []
            for key,value in graph.vertList.iteritems():
                if value not in global_covered:
                    optional_vertexes.append(value)

            ## the chosen is Vertex
            v_i = find_the_next_vertex(optional_vertexes)

            if v_i==-1:
                return clusters

            temp_list = [[v_i],[v_i],1]
            clusters.append(temp_list)
            global_covered.append(v_i)

    ## ----------------------------------
    ## find the UnCovered neighbors of V_i:
    ## ----------------------------------
    uncovered_neighbors1=[]
    uncovered_neighbors2=[]
    found_on_this_level = []

    for vertex in clusters[index][1]:

        neighbors = vertex.getConnections()

        #remove the current level covered vertices from the uncovered_neighbors
        for neighbor in neighbors:
                if neighbor not in found_on_this_level:
                    uncovered_neighbors1.append(neighbor)

        #print("neightbors of the vertex without This_level" + str(uncovered_neighbors1))

        #remove the Global covered vertices from the uncovered_neighbors
        for neighbor in uncovered_neighbors1:
                if neighbor not in global_covered:
                    uncovered_neighbors2.append(neighbor)

        #print("neightbors of the vertex without global_Covered" + str(uncovered_neighbors1))


        found_on_this_level.extend(uncovered_neighbors2)
        uncovered_neighbors1=[]
        uncovered_neighbors2=[]
                        ##############################
                        #                            #
                        #       The Condition        #
                        #                            #
                        ##############################
    if  len(found_on_this_level) >= pow(n,1/t) * ((clusters[index][2])):

            clusters[index][0].extend(found_on_this_level)
            clusters[index][1] = found_on_this_level
            clusters[index][2] += len(found_on_this_level)
            global_covered.extend(found_on_this_level)
            return create_clusters(graph,clusters,index,t,n,0)

    else:
        index +=1
        return create_clusters(graph,clusters,index,t,n,1)


#RETURN ALL EDGES BETWEEN TWO CLUSTERS
def find_conntions_between_A_and_B(graph,cluster_A,cluster_B):
    global egdes_to_remove
    kontact_between_A_and_B = []
    verts_of_A = cluster_A[0]
    verts_of_B = cluster_B[0]
    for a in verts_of_A:
        for neighbor in a.getConnections():
            if neighbor in verts_of_B:
                kontact_between_A_and_B.append([neighbor,a])
                no_dups = remove_duplicates(kontact_between_A_and_B)
                break

    for a in verts_of_B:
        for neighbor in a.getConnections():
            if neighbor in verts_of_A:
                kontact_between_A_and_B.append([neighbor,a])
                no_dups = remove_duplicates(kontact_between_A_and_B)
                break


    return kontact_between_A_and_B





def find_edge_between_clusters(g,clusters):


    edges_between_clusters = [[0 for x in range(len(clusters))] for x in range(len(clusters))]
    print(edges_between_clusters)
    #print ("yag" + str(edges_between_clusters[0][1]))

    # List of all edges connect between cluster A and cluster B

    all_connections_between_all_clusters =[]
    for cluster_A in clusters:
         for cluster_B in clusters:
                if (cluster_A!=cluster_B):
                        temp = find_conntions_between_A_and_B(g,cluster_A,cluster_B)
                        if temp:
                            #CHECK in matrix:
                            if edges_between_clusters[clusters.index(cluster_A)][clusters.index(cluster_B)]==0 and edges_between_clusters[clusters.index(cluster_B)][clusters.index(cluster_A)]==0 :
                                    all_connections_between_all_clusters.extend(temp)
                                    edges_between_clusters[clusters.index(cluster_A)][clusters.index(cluster_B)]=temp[0]
                                    edges_between_clusters[clusters.index(cluster_B)][clusters.index(cluster_A)]=temp[0]

    listAnswer = Matrix_To_list(edges_between_clusters)
    return listAnswer




def Matrix_To_list(matt):
    ans = []
    for list in matt:
        for tuple in list:
            if tuple!=0:
                ans.append(tuple)
    return ans




def remove_duplicates(tupleZZ):
      tuples=(tupleZZ)
      i=0
      n=len(tuples)
      while  i< n:
                pivot = tuples[i]
                index = i
                next_index=index+1
                end = len(tuples)
                while next_index<end :
                          if comp(tuples[next_index],pivot)==True:
                             del tuples[next_index]
                             end=end-1
                             n=n-1
                          else:
                             next_index=next_index+1
                i=i+1
      return tuples


def remove_duplicates_from_Matrix(tupleZZ):
      tuples=(tupleZZ)
      i=0
      n=len(tuples)
      while  i< n:
                pivot = tuples[i]
                index = i
                next_index=index+1
                end = len(tuples)
                while next_index<end :
                          if comp2(tuples[next_index],pivot)==True:
                             del tuples[next_index]
                             end=end-1
                             n=n-1
                          else:
                             next_index=next_index+1
                i=i+1
      return tuples




def comp(list1, list2):
        return list1[0].getId()==list2[1].getId() and list1[1].getId()==list2[0].getId()

def comp2(list1, list2):
        return list1[0].getId()==list2[0].getId()  and list1[1].getId()==list2[1].getId()



def build_Connections_in_each_cluster(clusters):

    t_cluster= Graph_()
    for cluster in clusters:
                for ver in cluster[0]:
                        id=ver.getId()
                        t_cluster.addVertex(id)
                        for neighbor in ver.getConnections():

                            if neighbor in cluster[0]:
                                        if neighbor.getId() not in t_cluster.vertList.keys():
                                                t_cluster.addVertex(neighbor.getId())
                                        if neighbor.getId() not in t_cluster.vertList[id].connectedTo.values():
                                                ####
                                                new_neighbor = Vertex(neighbor.getId())
                                                ###
                                                t_cluster.vertList[id].addNeighbor(new_neighbor)
                                                #t_cluster.vertList[neighbor.getId()].addNeighbor(ver)

    return t_cluster




def add_edges_between_the_clusters(t_cluster,edges_to_add):

    t_spanner = deepcopy(t_cluster)
    for edge in edges_to_add:
        From = edge[0]
        To = edge[1]
        t_spanner.addEdge(From.getId(),To.getId())

    return t_spanner

##########################################
####      Tests starts Here          #####
########################################


def Tests(g,t_spanner,n,t,edes_between_clusters,clusters):
    print "\n\n#################################"
    print "             TESTS:"
    print "\n"

    test_epsilon_ceiling(edes_between_clusters,n,t)
    ans = (test_Radius(g,clusters,t_spanner))
    if ans:
        print "Radius Test Passed!"
    else:
        print "Radius Test Failed!"

    ans = compare_graphs(g,t_spanner,t)
    if ans:
        print "Comparing Test Passed!"
    else:
        print "Comparing Test Failed!"
    print " \n\n#################################\n\n"

def test_epsilon_ceiling(edges_between_clusters,n,t):
    epsilon = len(edges_between_clusters)

    if  epsilon <= pow(n,1+(1/t)):
          print "epsilon Test Passed!"
    else:
          print "epsilon Test Failed!"


def test_Radius(g,clusters,t_spanner):
    index_of_cluster = 0
    ans = True
    centers = []
    for cluster in clusters:
        centers.append(cluster[0][0])

    for center in centers:

        v_i = t_spanner.vertList[center.getId()]
        if v_i.getConnections():

                bfs(t_spanner,v_i)
                compute_distances_to_v(t_spanner,v_i)
                #print "------"
                #print v_i
                #print" ------"
                for key,vert in t_spanner.vertList.iteritems():
                         #print vert.distance_from_vertex

                    ## it wont be here because the vertex is vertex of G (All vertices in cluster[index_of_cluster][0]!!!!!

                    #print "******>>"
                    #print vert
                    #print "<<******"
                    #print "---->>"
                    #for a in  clusters[index_of_cluster][0]:
                    #    print a
                    #print "<<----"

                    #if vert in clusters[index_of_cluster][0]:

                    for vert_of_g in clusters[index_of_cluster][0]:
                             if vert_of_g.getId()==vert.getId():


                                     #print "\nnkaka\n"
                                     #print str(vert.getId())+" His distance is: " +str(vert.distance_from_vertex)
                                     if vert.distance_from_vertex > t-1:

                                                        ans = False
        initial(t_spanner)
        index_of_cluster +=1
    return ans


def initial(t_spanner):
    for key,value  in t_spanner.vertList.iteritems():
         t_spanner.vertList[key].distance_from_vertex=-1
         t_spanner.vertList[key].parent=t_spanner.vertList[key]



# This test checks if every distance between two vertices in T_Spanner is <= the distance in G * |t|
def compare_graphs(g,t_spanner,t):
    ans = True
    for key,vertex in g.vertList.iteritems():

        # Compute the distance from all vertices in g, from them to vertex

        bfs(g,vertex)
        compute_distances_to_v(g,vertex)

        # Compute the distance from all vertices in t_spanner, from them to the parallel vertex

        parallel_vertex = t_spanner.vertList[vertex.getId()]
        bfs(t_spanner,parallel_vertex)
        compute_distances_to_v(t_spanner,parallel_vertex)
        for key,v in g.vertList.iteritems():
            if (v.distance_from_vertex * t) < t_spanner.vertList[v.getId()].distance_from_vertex and v.distance_from_vertex * t>0:
                ans = False
                #print "\n\nfailure: \n"
                #print  "vertex:  "+str(parallel_vertex)
                #print str(v.getId()) +"  dist in g: " + str (v.distance_from_vertex)
                #print str(v.getId()) + " dist in t: " + str(t_spanner.vertList[v.getId()].distance_from_vertex)

        initial(g)
        initial(t_spanner)

    return ans




def run_exp(g_tag,clusters,t,n):
    c= create_clusters(g_tag,clusters,0,t,n,1)
    edges_between_every_two_clusters = find_edge_between_clusters(g_tag,c)
    no_dups_list = remove_duplicates_from_Matrix(edges_between_every_two_clusters)
    t_cluster = build_Connections_in_each_cluster(clusters)
    t_spanner = add_edges_between_the_clusters(t_cluster,no_dups_list)

    return t_spanner,no_dups_list




if __name__ == '__main__':
    ######################
    # Read data from file
    ######################

    text_file = open("random.txt","r")
    lines = text_file.readlines()
    t =  float(lines[0][2:])
    #n = int(lines[1][2:])

    num_of_removed_edges = []
    probabilities = []



    ##########################
    ##                      ##
    ##                      ##
    ##     EXP Number 1     ##
    ##                      ##
    ##                      ##
    ##########################
    ############################################################################################
    #   Y - Average of Number of edges have been removed from Original Graph (in 10 experiments)
    #   X - Graph density
    ############################################################################################
    '''

    print (" ----------------------------------Experiment Number 1---------------------------------- \n\n")

    for i in range(0,1):
        print "\n\n\n"
        print "EXP number: "+str(i)
        print "\n\n\n"
        probability = ((i+1)/20)
        sum_removed=0
        for j in range(0,10):   # how accurate is the exp
            print "\n\n\n"
            print "iteration number: "+str(j)
            print "\n\n\n"
            g_tag,Matrix2 = create_Random_graph(probability,n)

            covered = {}
            clusters = []


            t_spanner,no_dups_list= run_exp(g_tag,clusters,t,n)

            #Tests(g_tag,t_spanner,n,t,no_dups_list,clusters)
            removed = show_graphs(g_tag,t_spanner,Matrix2,n)
            sum_removed += removed
            global_covered = []
            #removed=0


        num_of_removed_edges.append(sum_removed/10)
        probabilities.append((probability))
        #sum_removed=0
        #removed=0



    print"\n\n\n"
    print num_of_removed_edges
    print probabilities
    plt.plot(probabilities,num_of_removed_edges)
    plt.ylabel('Average of Number of edges have been removed from Original Graph (in 10 experiments) ')
    plt.xlabel('Graph density ')
    plt.show()
    '''
    '''
    print (" ----------------------------------Experiment Number 2----------------------------------\n\n")

    num_of_removed_edges = []
    ##########################
    ##                      ##
    ##                      ##
    ##     EXP Number 2     ##
    ##                      ##
    ##                      ##
    ##########################
    ############################################################################################
    #   Y - Average of Number of edges have been removed from Original Graph (in 10 experiments)
    #   X - t
    ############################################################################################

    tis = []
    for i in range(0,10):
        print "\n\n\n"
        print "EXP number: "+str(i)
        print "\n\n\n"
        t =i+1
        sum_removed=0
        for j in range(0,10):     # how accurate is the exp
                print "\n\n\n"
                print "iteration number: "+str(j)
                print "\n\n\n"
                g_tag,Matrix2 = create_Random_graph(0.05,n)

                covered = {}
                clusters = []


                t_spanner,no_dups_list= run_exp(g_tag,clusters,t,n)
                #Tests(g_tag,t_spanner,n,t,no_dups_list,clusters)
                removed = show_graphs(g_tag,t_spanner,Matrix2,n)
                sum_removed += removed
                global_covered = []


        num_of_removed_edges.append(sum_removed/10)
        tis.append(t)



    plt.plot(tis,num_of_removed_edges)
    plt.ylabel('Average of Number of edges have been removed from Original Graph (in 10 experiments) ')
    plt.xlabel('t')
    plt.show()



    print (" ----------------------------------Experiment Number 3----------------------------------\n\n")

    '''
    num_of_removed_edges = []
    ##########################
    ##                      ##
    ##                      ##
    ##     EXP Number 3     ##
    ##                      ##
    ##                      ##
    ##########################
    ############################################################################################
    #   Y - Average of Number of edges have been removed from Original Graph (in 10 experiments)
    #   X - Number of vertices
    ############################################################################################


    nis = []
    for i in range(0,10):
        print "\n\n\n"
        print "EXP number: "+str(i)
        print "\n\n\n"
        n =((i+1)*20)
        Matrix2 = [[0 for x in range(n)] for x in range(n)]
        sum_removed=0
        total_edges=0
        for j in range(0,10):     # how accurate is the exp
                print "\n\n\n"
                print "iteration number: "+str(j)
                print "\n\n\n"
                g_tag,Matrix2 = create_Random_graph(0.05,n)

                covered = {}
                clusters = []


                t_spanner,no_dups_list= run_exp(g_tag,clusters,t,n)
                #Tests(g_tag,t_spanner,n,t,no_dups_list,clusters)
                removed = show_graphs(g_tag,t_spanner,Matrix2,n)
                sum_removed += removed
                global_covered = []


                #for i in range(0,n):
                #    for j in range(0,n):
                #        if Matrix2[i][j]==1:
                #            total_edges +=1
        #if total_edges!=0:
        num_of_removed_edges.append((sum_removed/10))
        #else:
        #    num_of_removed_edges.append((sum_removed/10))

        nis.append(n)



    print"\n\n\n"
    print num_of_removed_edges
    #print g_tag.vertList

    plt.plot(nis,num_of_removed_edges)
    plt.ylabel('Average of Number of edges have been removed from Original Graph (in 10 experiments) ')
    plt.xlabel('Number of vertices')
    plt.show()
