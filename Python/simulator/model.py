import numpy as np

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
        self.tri = None

    def needsUpdate( self ):
        return self.tri is None
        
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
        self.tri = True
        
    def _updateVertices( self ):
        self.allVertices = self.vertices

    def _updateLinks( self ):
        self.links = np.empty( ( self.count(), 0 ), dtype = np.int32 )
        self.degrees = np.full( self.count(), 0, dtype = np.int32 )

    def _updateBordersAndTris( self ):
        self.tris = np.empty( ( self.count(), 0 ), dtype = np.int32 )
        self.borders = np.empty( ( self.count(), 0 ), dtype = np.int32 )

    def updateGeometry( self ):
        self._updateSV()
        self._updateVertices()
        self._updateLinks()
        self._updateBordersAndTris()
