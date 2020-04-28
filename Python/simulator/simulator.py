from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing import cpu_count
import numpy as np

class Simulator:
    def __init__( self, friction = 100, repulsion = 5e-06, steps = 0 ):
        self.executor = ThreadPoolExecutor( cpu_count() )
        self.friction  = friction
        self.repulsion = repulsion
        self.steps     = steps

    @staticmethod
    def _rejectionForIds( vertices, ids ):
        result = np.empty( ( ids.size, 3 ), dtype = np.float32 )
        for o, i in enumerate( ids ):
            diffs = vertices[i] - vertices
            dists = np.square( diffs ).sum( axis = 1 )
            diffs /= dists[:,np.newaxis] ** 1.5
            result[o] = np.nansum( diffs, axis = 0 )
        return result

    def simulateStep( self, model ):
        rangeSegments = np.array_split( range( model.count() ), min( cpu_count(), model.count() ) )
        futures = [self.executor.submit( self._rejectionForIds, model.vertices, ids ) for ids in rangeSegments] 
        
        for ids, future in zip( rangeSegments, futures ):
            model.translations[ids[0]:ids[0]+ids.size] += self.repulsion * future.result()

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
