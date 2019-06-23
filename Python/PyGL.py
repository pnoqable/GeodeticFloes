import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from ctypes import *
import numpy as np
from scipy.spatial import *
from concurrent.futures import *

pygame.init()
screen = pygame.display.set_mode( ( 800, 600 ), pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF, 24 )

np.random.seed()
vertices = np.random.sample( ( 169, 3 ) ).astype( 'float32' ) - 0.5
vertices /= np.linalg.norm( vertices, axis = 1 )[:,np.newaxis]
translations = np.zeros( vertices.shape, dtype = 'float32' )

glClearColor( 0.0, 0.5, 0.5, 1.0 )

glEnable( GL_DEPTH_TEST )
glDepthFunc( GL_LEQUAL )

glEnable( GL_BLEND )

glPolygonMode( GL_FRONT,  GL_FILL )

verticesBO = vbo.VBO( np.zeros( (0,3), dtype = 'float32' ) )
svVerticesBO = vbo.VBO( np.zeros( (0,3), dtype = 'float32' ) )
glEnableClientState( GL_VERTEX_ARRAY )

class GameState:
    resolution = ( 0, 0 )

    distance    = 5
    longitude   = 0
    latitude    = 0

    running     = True
    points      = True
    delauney    = False
    voronoi     = True
    borders     = True
    dmin        = 2.

    repulsion   = 0.00005
    deccelerate = 0

    def resetStats(self):
        self.dmin = 2.

state = GameState()

while state.running:

    for e in pygame.event.get():
        # print( e.type, e.dict )
        mod = 1 if 'mod' in e.dict and int( e.dict['mod'] ) & ( 64 + 128 ) else 0 # left or right CTRL
        mod2 = 1 if 'mod' in e.dict and int( e.dict['mod'] ) & ( 1 + 2 ) else 0 # left or right SHIFT
        if e.type == pygame.QUIT or \
           e.type == pygame.KEYDOWN and e.dict['key'] == 27: # ESC
            state.running = False
        elif e.type == pygame.VIDEORESIZE:
            state.resolution = ( e.w, e.h )
            glViewport( 0, 0, e.w, e.h )
        elif e.type == pygame.MOUSEMOTION and e.dict['buttons'][0]:
            state.longitude += np.pi / 12 * e.dict['rel'][0]
            state.latitude  += np.pi / 12 * e.dict['rel'][1]
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 4:
            state.distance += 0.5
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 5:
            state.distance -= 0.5
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 93: # '+'
            count = 1 * pow( 10, mod ) * pow( 100, mod2 )
            vertices = np.append( vertices, np.random.sample( ( count, 3 ) ).astype( 'float32' ) - 0.5, axis = 0 )
            translations = np.append( translations, np.zeros( ( count, 3 ), dtype = 'float32' ), axis = 0 )
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 47: # '-'
            count = 1 * pow( 10, mod ) * pow( 100, mod2 )
            count = min( vertices.shape[0] - 4, count )
            if count > 0:
                vertices = vertices[:-count]
                translations = translations[:-count]
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'p':
            state.points = not state.points
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'd':
            state.delauney = not state.delauney
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'v':
            state.voronoi = not state.voronoi
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'b':
            state.borders = not state.borders
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 32: # ' '
            state.deccelerate = 0.2 * pow( 10, mod ) * pow( -1, mod2 )
        elif e.type == pygame.KEYUP and e.dict['key'] == 32: # ' '
            state.deccelerate = 0
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 114: # 'r'
            state.repulsion = round( state.repulsion - 0.000001 * pow( 10, mod ) * pow( -1, mod2 ), 6 )

    state.resetStats()

    def calcRejectionFor( i ):
        diffs = vertices[i] - vertices
        distsSquared = np.square( diffs ).sum( axis = 1 )
        # mask = distsSquared < 0.1
        # diffs = diffs[mask]
        # distsSquared = distsSquared[mask]
        dists = np.sqrt( distsSquared )
        directions = diffs / dists[:,np.newaxis]
        rejections = directions / distsSquared[:,np.newaxis]
        rejection = np.nansum( rejections, axis = 0 )
        state.dmin = min( state.dmin, np.min( dists[dists>0] ) )
        return rejection

    translationsSquared = np.square( translations ).sum( axis = 1 )
    translations *= np.power( np.e, -10*translationsSquared )[:,np.newaxis]

    for i in range( vertices.shape[0] ):
        translations[i] += state.repulsion * calcRejectionFor( i )

    projection = ( vertices * translations ).sum( axis = 1 )
    translations -= vertices * projection[:,np.newaxis]

    if state.deccelerate:
        translations *= pow( np.e, -state.deccelerate )

    vertices += translations

    vertices /= np.linalg.norm( vertices, axis = 1 )[:,np.newaxis]
    
    hull = ConvexHull( vertices )
    sv = SphericalVoronoi( vertices )
    sv.sort_vertices_of_regions()

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

    glLoadIdentity()
    gluPerspective( 45, state.resolution[0] / state.resolution[1], 0.1, 50.0 )
    glTranslate( 0, 0, -state.distance )
    glRotate( state.latitude, 1, 0, 0 )
    glRotate( state.longitude, 0, 1, 0 )

    verticesBO.set_array( vertices )
    
    factor = 1.02
    glScalef( factor, factor, factor )
            
    if state.points:
        with verticesBO:
            glPointSize( 5 )
            glColor4f( 0, 0, 1, 0.5 )
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glDrawArrays( GL_POINTS, 0, vertices.shape[0] )

    if state.delauney:
        with verticesBO:
            glColor4f( 0, 0, 1, 0.2 )   
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            for simplex in hull.simplices:
                glDrawElements( GL_LINE_LOOP, simplex.shape[0], GL_UNSIGNED_INT, simplex )
    
    factor = 1 / factor
    glScalef( factor, factor, factor )

    svVerticesBO.set_array( np.array( sv.vertices, dtype='float32' ) )
                
    if state.voronoi:      
        with svVerticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, svVerticesBO )
            for region in sv.regions:
                brightness = 1 - 0.1 * ( len( region ) - 3 )
                glColor4f( brightness, brightness, brightness, 1 )
                glDrawElements( GL_POLYGON, len( region ), GL_UNSIGNED_INT, region )

    factor = 1.002
    glScalef( factor, factor, factor )
    
    if state.borders:
        with svVerticesBO:
            glColor4f( 0, 0, 0, 1 )
            glVertexPointer( 3, GL_FLOAT, 0, svVerticesBO )
            for region in sv.regions:
                glDrawElements( GL_LINE_LOOP, len( region ), GL_UNSIGNED_INT, region )
    
    factor = 1 / factor
    glScalef( factor, factor, factor )

    glMatrixMode( GL_PROJECTION )
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D( 0.0, state.resolution[0], 0.0, state.resolution[1] )
    glMatrixMode( GL_MODELVIEW )
    glLoadIdentity()
    glBlendFunc( GL_SRC_ALPHA, GL_ONE )

    def drawText( pos, size, text ):
        font = pygame.font.Font( None, size )
        lines = text.split( "\n" )
        lines.reverse()
        for line in lines:
            surface = font.render( line, True, (255,255,255,255), (0,0,0,0) )
            pixels  = pygame.image.tostring( surface, "RGBA", True )

            glRasterPos3d( *pos )     
            glDrawPixels( surface.get_width(), surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, pixels )
            pos = ( pos[0], pos[1] + surface.get_height(), pos[2] )

    drawText( (0,0,0), 24, "Voronoi Faces: " + str( vertices.shape[0] ) + "\n" + \
                           "Delauney Faces: " + str( vertices.shape[0] * 2 - 4 ) + "\n" + \
                           "Repulsion: " + str( round( 1000000 * state.repulsion ) ) + "uf^-2\n"
                           "Dmin: " + str( int( 1000 * state.dmin ) ) + "mU\n" + \
                           "Temperature: " + str( round( 1000000 * translationsSquared.sum() ) ) + "uU²/f²" )

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    pygame.display.flip ()
