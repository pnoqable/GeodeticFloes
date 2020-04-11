import glm
import numpy as np

class Camera:
    def __init__( self ):
        self.resolution = None
        self.view       = glm.translate( glm.mat4( 1.0 ), glm.vec3( 0, 0, -3 ) )
        self.proj       = None

    def setResolution( self, width, height ):
        self.resolution = ( width, height )
        self.proj = glm.perspective( np.pi/4, width / height, 0.1, 10 )

    def mvp( self ):
        return np.array( self.proj * self.view, dtype = float )

    def pos( self ):
        invMvp = np.linalg.inv( self.mvp() )
        return invMvp[2,:3] / invMvp[2,3]

    def rotate( self, a, b ):
        a /= np.linalg.norm( a )
        b /= np.linalg.norm( b )
        cosArc = ( a * b ).sum()
        if cosArc <= -1:
            self.view = glm.scale( self.view, glm.vec3( -1, 1, -1 ) )
        elif cosArc < 1:
            angle = np.arccos( cosArc )
            axis = np.cross( a, b ) 
            self.view = glm.rotate( self.view, angle, glm.vec3( axis ) )

    def unproject( self, screenPos ):
        screenPos = np.append( screenPos, [1] ).astype( float )
        screenPos[:2] /= self.resolution
        screenPos[:3] = 2 * screenPos[:3] - 1
        invMvp = np.linalg.inv( self.mvp() )
        worldPos = invMvp.T.dot( screenPos )
        return worldPos[:3] / worldPos[3]

    def unprojectToSphereNear( self, screenPos, snapToSphere = False ):
        screenPos = np.array( screenPos )
        screenPos[1] = self.resolution[1] - screenPos[1]
        worldNear = self.unproject( np.append( screenPos, [0] ) )
        worldRear = self.unproject( np.append( screenPos, [1] ) )

        p = np.array( worldNear, dtype = 'float32' )[:3]
        d = np.array( worldRear, dtype = 'float32' )[:3] - p
        d /= np.linalg.norm( d )
        
        term0 = np.dot( d, p )
        term1 = 1 + np.square( term0 ) - np.square( p ).sum()

        if term1 >= 0:
            sqrtTerm1 = np.sqrt( term1 )

            projection0 = term0 + sqrtTerm1
            if 0 > projection0:
                return p - d * projection0

            projection1 = term0 - sqrtTerm1
            if 0 > projection1:
                return p - d * projection1
        
        elif snapToSphere:
            res = p - d * ( term0 + 2 * term1 )
            eye = -self.pos()
            if np.dot( res, eye ) > 0:
                res = np.cross( eye, d )
                res = np.cross( res, eye )
            return res / np.linalg.norm( res )
        
        return None

    def zoom( self, exp ):
        self.view[3,2] *= 1.1 ** exp

    def ortho( self ):
        ortho = glm.ortho( 0, self.resolution[0], 0, self.resolution[1], 0, 1 )
        return np.array( ortho, dtype = float )
