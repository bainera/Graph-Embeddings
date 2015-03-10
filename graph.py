class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}
        self.distance_from_vertex = -1  ## dictinary saves the distance to each vertex
        self.parent = self

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] =  nbr.getId()

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id


class Graph_:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None
    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t])
        self.vertList[t].addNeighbor(self.vertList[f])

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

