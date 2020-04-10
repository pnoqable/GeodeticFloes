import numpy as np

from scipy.spatial import SphericalVoronoi

class Model:
    @staticmethod
    def arrangedPointsOnSphere( n ):
        indices = np.arange( n ).astype( np.float32 ) + 0.5
        thetas = np.pi * ( 1 + 5 ** 0.5 ) * indices
        phis = np.arccos( 1 - 2 * indices / indices.shape[0] )
        widths = np.sin( phis )
        heights = np.cos( phis )
        return np.array( [ widths * np.cos( thetas ), heights, widths * np.sin( thetas ) ] ).T

    @staticmethod
    def randomPointsOnSphere( n ):
        thetas = 2 * np.pi * np.random.sample( n ).astype( np.float32 )
        heights = 2 * np.random.sample( n ).astype( np.float32 ) - 1
        widths = ( 1 - heights ** 2 ) ** 0.5
        return np.array( [ widths * np.cos( thetas ), heights, widths * np.sin( thetas ) ] ).T

    def __init__( self, count ):
        np.random.seed()
        self.vertices = self.randomPointsOnSphere( count )
        self.translations = np.zeros( self.vertices.shape, dtype = np.float32 )
        self.invalidate()
    
    def invalidate( self ):
        self.sv = None

    def needsUpdate( self ):
        return self.sv is None
        
    def count( self ):
        return self.vertices.shape[0]

    def temperature( self ):
        return np.square( self.translations ).sum()

    def addVerticesAt( self, positions ):
        newVertices = positions / np.linalg.norm( positions, axis = 1 )[:,np.newaxis]
        newTranslations = np.zeros_like( newVertices )
        self.vertices = np.append( self.vertices, newVertices, axis = 0 )
        self.translations = np.append( self.translations, newTranslations, axis = 0 )
        self.invalidate()
    
    def addVertices( self, count ):
        newVertices = self.randomPointsOnSphere( count )
        self.addVerticesAt( newVertices )

    def vertexIdAt( self, pos ):
        return np.dot( self.vertices, pos ).argmax()

    def resetVertex( self, i, pos ):
        self.vertices[i] = pos / np.linalg.norm( pos )
        self.translations[i] = 0
        self.invalidate()
    
    def resetAllVertices( self, random ):
        makePoints = self.randomPointsOnSphere if random else self.arrangedPointsOnSphere 

        self.vertices = makePoints( self.count() )
        self.translations[:] = 0
        self.invalidate()
    
    def removeVertexIds( self, ids ):
        self.vertices = np.delete( self.vertices, ids, axis = 0 )
        self.translations = np.delete( self.translations, ids, axis = 0 )
        self.invalidate()

    def removeVertices( self, count ):
        count = min( count, self.count() - 4 )
        ids = self.vertices.shape[0] - count + np.arange( count )
        self.removeVertexIds( ids )

    def _updateSV( self ):
        self.sv = SphericalVoronoi( self.vertices )
        self.sv.sort_vertices_of_regions()
        
    def _updateVertices( self ):
        self.allVertices = np.append( self.vertices, self.sv.vertices ).astype( np.float32 )

    def _updateLinks( self ):
        links = [set() for _ in self.vertices]
        for simplex in self.sv._tri.simplices:
            a, b, c = simplex[0], simplex[1], simplex[2]
            links[a].update( ( b, c ) )
            links[b].update( ( a, c ) )
            links[c].update( ( a, b ) )

        self.degrees = np.empty( len( links ), dtype = np.int32 )
        self.links = np.empty( len( links ), dtype = np.object_ )
        for i, ends in enumerate( links ):
            degree =  len( ends )
            array = np.empty( ( degree, 2 ), np.int32 )
            array[:,0] = i
            array[:,1] = np.fromiter( ends, np.int32 )
            self.links[i] = array
            self.degrees[i] = degree

        self.allLinks = np.concatenate( self.links )

    def _updateBordersAndTris( self ):
        self.tris = np.empty( len( self.sv.regions ), dtype = np.object_ )
        self.borders = np.empty( len( self.sv.regions ), dtype = np.object_ )
        for i, region in enumerate( self.sv.regions ):
            array = np.array( region ) + self.count()
            tris = np.empty( ( array.size, 3 ), np.int32 )
            borders = np.empty( ( array.size, 2 ), np.int32 )
            tris[:,0] = i
            tris[0,1] = borders[0,0] = array[-1]
            tris[1:,1] = borders[1:,0] = array[:-1]
            tris[:,2] = borders[:,1] = array
            self.tris[i] = tris
            self.borders[i] = borders

        self.allBorders = np.concatenate( self.borders )

    def updateGeometry( self ):
        self._updateSV()
        self._updateVertices()
        self._updateLinks()
        self._updateBordersAndTris()
