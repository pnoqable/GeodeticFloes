import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ctypes import *
import numpy as np
from scipy.spatial import *

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

running = True
while running:

    for event in pygame.event.get():
        # print( event.type, event.dict )
        if ( event.type == pygame.QUIT ) or \
           ( event.type == pygame.KEYDOWN ) and ( event.dict['key'] == 27 ):
            running = False
        elif ( event.type == pygame.MOUSEMOTION ) and event.dict['buttons'][0]:
            glRotate( np.pi / 12 * event.dict['rel'][0], 0,1,0 )
            # glRotate( np.pi / 12 * event.dict['rel'][1], 1,0,0 )
        elif ( event.type == pygame.MOUSEBUTTONDOWN ) and event.dict['button'] == 4:
            glTranslatef( 0, 0,  0.5 )
        elif ( event.type == pygame.MOUSEBUTTONDOWN ) and event.dict['button'] == 5:
            glTranslatef( 0, 0, -0.5 )
        elif ( event.type == pygame.KEYDOWN ) and event.dict['unicode'] == '+':
            vertices = np.append( vertices, [np.random.sample(3) - 0.5], axis=0 )
            translations = np.append( translations, [( 0, 0, 0 )], axis=0 )
        elif ( event.type == pygame.KEYDOWN ) and event.dict['unicode'] == '-':
            vertices = vertices[:-1]
            translations = translations[:-1]

    def reflect( pos, vel ):
        n = norm( pos )
        return vel - np.inner( n, vel ) * n

    dmin = 20.
    dmax = 0.
    
    for i in range( vertices.shape[0] ):
        for j in range( i+1, vertices.shape[0] ):
            d = vertices[i] - vertices[j]
            rejection = norm(d) / max( 0.1, np.inner( d, d ) )
            translations[i] += 0.01 * rejection
            translations[j] -= 0.01 * rejection
            dmin = min( len(d), dmin )
            dmax = max( len(d), dmax )

        translations[i] = 0.9 * reflect( vertices[i], translations[i] )

    for i in range( vertices.shape[0] ):
        vertices[i] = norm( vertices[i] + translations[i] )
    
    glClear ( GL_COLOR_BUFFER_BIT )
    glClear ( GL_DEPTH_BUFFER_BIT )
    
    # glBufferData (GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
    # glVertexPointer (3, GL_FLOAT, 0, None)

    glEnable( GL_DEPTH_TEST )
    glDisable( GL_BLEND )

    sv = SphericalVoronoi( vertices )
    sv.sort_vertices_of_regions()
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
    glEnable( GL_BLEND )

    glPointSize( 5 )
    glColor4f( 0, 0, 1, 0.1 )
    # glDrawArrays( GL_POINTS, 0, vertices.shape[0] )
    glBegin( GL_POINTS )
    for vertex in vertices:
        glVertex3fv( vertex )
    glEnd()

    hull = ConvexHull( vertices )
    for simplex in hull.simplices:
        glBegin( GL_LINE_LOOP )
        for vertex in vertices[simplex]:
            glVertex3fv( vertex )
        glEnd()
        # glDrawElements( GL_LINE_LOOP, 3, GL_UNSIGNED_INT, simplex )

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

    drawText( (0,0,0), 14, "Vertices: " + str( np.array( hull.simplices ).shape[0] ) + "\n" + \
                           "Faces: " + str( vertices.shape[0] ) + "\n" + \
                           "dmin: " + str( round( state.dmin, 2 ) ) + "\n" + \
                           "dmax: " + str( round( state.dmax, 2 ) ) )

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    pygame.display.flip ()

    # pygame.time.wait(10)
    