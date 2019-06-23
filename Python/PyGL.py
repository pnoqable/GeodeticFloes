import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ctypes import *
import numpy as np
from scipy.spatial import *
from concurrent.futures import *

pygame.init ()
screen = pygame.display.set_mode ((800,600), pygame.OPENGL | pygame.DOUBLEBUF, 24)
glViewport (0, 0, 800, 600)
gluPerspective( 45, 4/3., 0.1, 50.0)
glTranslatef( 0, 0, -5 )
glClearColor (0.0, 0.5, 0.5, 1.0)

def len(v):
    return np.linalg.norm(v)

def norm(v):
    n = len(v)
    return v/n if n != 0 else v

vertices = np.array( [ norm( np.random.sample(3) - 0.5 ) for _ in range( 32 ) ], dtype='float32' )
translations = np.array( [ ( 0, 0, 0 ) for _ in vertices ], dtype='float32' )

# glEnableClientState( GL_VERTEX_ARRAY )
# vbo = glGenBuffers( 1 )
# glBindBuffer( GL_ARRAY_BUFFER, vbo )

glEnable( GL_DEPTH_TEST )
glDepthFunc( GL_LESS )

glEnable( GL_BLEND )
glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

glPolygonMode( GL_FRONT,  GL_FILL )

class GameState:
    running  = True
    points   = False
    delauney = False
    voronoi  = True
    dmin     = 2
    dmax     = 0

    def resetStats(self):
        self.dmin = 2
        self.dmax = 0

state = GameState()

while state.running:

    for e in pygame.event.get():
        # print( e.type, e.dict )
        if e.type == pygame.QUIT or \
           e.type == pygame.KEYDOWN and e.dict['key'] == 27:
            state.running = False
        elif e.type == pygame.MOUSEMOTION and e.dict['buttons'][0]:
            glRotate( np.pi / 12 * e.dict['rel'][0], 0,1,0 )
            # glRotate( np.pi / 12 * e.dict['rel'][1], 1,0,0 )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 4:
            glTranslatef( 0, 0,  0.5 )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 5:
            glTranslatef( 0, 0, -0.5 )
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == '+':
            vertices = np.append( vertices, [np.random.sample(3) - 0.5], axis=0 )
            translations = np.append( translations, [( 0, 0, 0 )], axis=0 )
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == '-':
            vertices = vertices[:-1]
            translations = translations[:-1]
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'p':
            state.points = not state.points
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'd':
            state.delauney = not state.delauney
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'v':
            state.voronoi = not state.voronoi

    state.resetStats()

    rejections = np.zeros( ( vertices.shape[0], vertices.shape[0] ), dtype=(float,3) )
    
    def calcRejectionFor( i ):
        diffs = vertices[i] - vertices
        distsSquared = np.square( diffs ).sum( axis=1 )[:,np.newaxis]
        dists = np.sqrt( distsSquared )
        directions = diffs / dists
        rejections = directions / np.maximum( 0.1, distsSquared )
        rejection = np.nansum( rejections, axis=0 )
        state.dmin = min( state.dmin, np.nanmin( dists ) )
        state.dmax = max( state.dmax, np.nanmax( dists ) )
        return rejection

    def project( pos, vel ):
        n = norm( pos )
        return vel - np.inner( n, vel ) * n

    for i in range( vertices.shape[0] ):
        translations[i] += 0.01 * calcRejectionFor( i )
        translations[i] = 0.9 * project( vertices[i], translations[i] )

    for i in range( vertices.shape[0] ):
        vertices[i] = norm( vertices[i] + translations[i] )
    
    hull = ConvexHull( vertices )
    sv = SphericalVoronoi( vertices )

    glClear ( GL_COLOR_BUFFER_BIT )
    glClear ( GL_DEPTH_BUFFER_BIT )
    
    # glBufferData (GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
    # glVertexPointer (3, GL_FLOAT, 0, None)

    if state.points:
        glPointSize( 5 )
        glColor4f( 0, 0, 1, 0.5 )
        # glDrawArrays( GL_POINTS, 0, vertices.shape[0] )
        glBegin( GL_POINTS )
        for vertex in vertices:
            glVertex3fv( vertex )
        glEnd()

    sv.sort_vertices_of_regions()
    if state.voronoi:
        voronoiVertices = np.array( sv.vertices, dtype='float32' )
        for region in sv.regions:
            glColor4f( 0, 0, 0, 1 )
            glBegin( GL_LINE_LOOP )
            for vertex in sv.vertices[region]:
                glVertex3f( vertex[0], vertex[1], vertex[2] )
            glEnd()
            if( sv.vertices[region].shape[0] % 2 ):
                glColor4f( 0.5, 0.5, 0.5, 1 )
            else:
                glColor4f( 1, 1, 1, 1 )
            glBegin( GL_POLYGON )
            for vertex in sv.vertices[region]:
                glVertex3fv( vertex )
            glEnd()

    glDisable( GL_DEPTH_TEST )
    if state.delauney:
        for simplex in hull.simplices:
            glColor4f( 0, 0, 1, 0.2 )
            glBegin( GL_LINE_LOOP )
            for vertex in vertices[simplex]:
                glVertex3fv( vertex )
            glEnd()
            glColor4f( 0, 0, 1, 0.1 )
            glBegin( GL_POLYGON )
            for vertex in vertices[simplex]:
                glVertex3fv( vertex )
            glEnd()
    glEnable( GL_DEPTH_TEST )

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0.0, 800.0, 0.0, 600.0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    def drawText( pos, size, text ):
        font = pygame.font.Font( None, size )
        for line in text.split( "\n" ):
            surface = font.render( line, True, (255,255,255,255), (0,0,0,0) )
            pixels  = pygame.image.tostring( surface, "RGBA", True )

            glRasterPos3d( *pos )     
            glDrawPixels( surface.get_width(), surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, pixels )
            pos = ( pos[0], pos[1]+surface.get_height(), pos[2] )

    drawText( (0,0,0), 14, "Voronoi Faces: " + str( vertices.shape[0] ) + "\n" + \
                           "Delauney Faces: " + str( vertices.shape[0] * 2 - 4 ) + "\n" + \
                           "dmin: " + str( round( state.dmin, 2 ) ) + "\n" + \
                           "dmax: " + str( round( state.dmax, 2 ) ) )

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    pygame.display.flip ()

    # pygame.time.wait(10)
    