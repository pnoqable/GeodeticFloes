import numpy as np
import numba
from numba import njit, prange

class Simulator:
    def __init__( self, friction = 100, repulsion = 5e-06, steps = 0 ):
        self.friction  = friction
        self.repulsion = repulsion
        self.steps     = steps

    @staticmethod
    @njit( parallel = True, fastmath = True, cache = True )
    def _rejectionForId( vertices, i ):
        diffs = vertices[i] - vertices
        dists = np.square( diffs )
        result = np.zeros_like( vertices[i] )
        for j in prange( vertices.shape[0] ):
            dist = dists[j].sum()
            if dist > 0:
                result += diffs[j] / dist ** 1.5
        return result

    def simulateStep( self, model ):
        for i in range( model.count() ):
            model.translations[i] += self.repulsion * self._rejectionForId( model.vertices, i )

        projections = np.sum( model.vertices * model.translations, axis = 1 )
        model.translations -= model.vertices * projections[:,np.newaxis]

        translationsSquared = np.square( model.translations ).sum( axis = 1 )
        model.translations *= np.power( np.e, -self.friction*translationsSquared )[:,np.newaxis]

        model.vertices += model.translations
        model.vertices /= np.linalg.norm( model.vertices, axis = 1 )[:,np.newaxis]

        model.invalidate()

    def simulate( self, model ):    
        for _ in range( 1 if self.steps < 0 else self.steps ):
            self.simulateStep( model )
        
        if self.steps < 0:
            self.steps += 1
