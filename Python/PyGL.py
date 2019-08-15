import numpy as np
from scipy.spatial import SphericalVoronoi

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import itertools as it

pygame.init()
screen = pygame.display.set_mode( ( 800, 600 ), pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF, 24 )

np.random.seed()
vertices = np.random.sample( ( 32, 3 ) ).astype( 'float32' ) - 0.5
vertices /= np.linalg.norm( vertices, axis = 1 )[:,np.newaxis]
translations = np.zeros( vertices.shape, dtype = 'float32' )

glClearColor( 0.0, 0.5, 0.5, 1.0 )

glEnable( GL_DEPTH_TEST )
glDepthFunc( GL_LEQUAL )

glEnable( GL_BLEND )

glPolygonMode( GL_FRONT,  GL_FILL )

verticesBO = vbo.VBO( np.zeros( (0,3), dtype = 'float32' ) )
glEnableClientState( GL_VERTEX_ARRAY )

vtxShader = glCreateShader( GL_VERTEX_SHADER )
glShaderSource( vtxShader, """
    void main() {
        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
        
        vec4 color = gl_Color;
        gl_FrontColor = color;
    }
""")
glCompileShader( vtxShader )

if glGetShaderiv( vtxShader, GL_COMPILE_STATUS ) != GL_TRUE:
    message = glGetShaderInfoLog( vtxShader )
    raise RuntimeError( message )

lineShader = glCreateShader( GL_GEOMETRY_SHADER )
glShaderSource( lineShader, """

    layout( lines ) in;
    layout( line_strip, max_vertices = 9 ) out;

    const int lines = 8;

    void main()
    {
        for( int i = 0; i <= lines; i++) {
            float b = 1. * i / lines;
            float a = 1. - b;

            vec4 middle = a * gl_in[0].gl_Position + b * gl_in[1].gl_Position;
            vec4 middleWorld = inverse( gl_ModelViewProjectionMatrix ) * middle;

            middleWorld.xyz = normalize( middleWorld.xyz );

            gl_Position = gl_ModelViewProjectionMatrix * middleWorld;
            gl_FrontColor = a * gl_in[0].gl_FrontColor + b * gl_in[1].gl_FrontColor;
            EmitVertex();
        }
        EndPrimitive();
    }
""" )
glCompileShader( lineShader )

if glGetShaderiv( lineShader, GL_COMPILE_STATUS ) != GL_TRUE:
    message = glGetShaderInfoLog( lineShader )
    raise RuntimeError( message )

shaderProgramLines = glCreateProgram()
glAttachShader( shaderProgramLines, vtxShader )
glAttachShader( shaderProgramLines, lineShader )
glLinkProgram( shaderProgramLines )

if glGetProgramiv( shaderProgramLines, GL_LINK_STATUS ) != GL_TRUE:
    message = glGetProgramInfoLog( shaderProgramLines )
    raise RuntimeError( message )

triShader = glCreateShader( GL_GEOMETRY_SHADER )
glShaderSource( triShader, """

    layout( triangles ) in;
    layout( triangle_strip, max_vertices = 27 ) out;

    const int lines = 8;

    void main()
    {
        vec4 lastPos = gl_in[1].gl_Position;
        vec4 lastCol = gl_in[1].gl_FrontColor;

        for( int i = 1; i <= lines; i++) {
            gl_Position = gl_in[0].gl_Position;
            gl_FrontColor = gl_in[0].gl_FrontColor;
            EmitVertex();

            gl_Position = lastPos;
            gl_FrontColor = lastCol;
            EmitVertex();

            float b = 1. * i / lines;
            float a = 1. - b;

            vec4 middle = a * gl_in[1].gl_Position + b * gl_in[2].gl_Position;
            vec4 middleWorld = inverse( gl_ModelViewProjectionMatrix ) * middle;

            middleWorld.xyz = normalize( middleWorld.xyz );

            lastPos = gl_ModelViewProjectionMatrix * middleWorld;  // middle;
            lastCol = a * gl_in[1].gl_FrontColor + b * gl_in[2].gl_FrontColor;

            gl_Position = lastPos;
            gl_FrontColor = lastCol;
            EmitVertex();
            
            EndPrimitive();
        }
    }
""" )
glCompileShader( triShader )

if glGetShaderiv( triShader, GL_COMPILE_STATUS ) != GL_TRUE:
    message = glGetShaderInfoLog( triShader )
    raise RuntimeError( message )

shaderProgramTris = glCreateProgram()
glAttachShader( shaderProgramTris, vtxShader )
glAttachShader( shaderProgramTris, triShader )
glLinkProgram( shaderProgramTris )

if glGetProgramiv( shaderProgramTris, GL_LINK_STATUS ) != GL_TRUE:
    message = glGetProgramInfoLog( shaderProgramTris )
    raise RuntimeError( message )

class GameState:
    resolution  = None

    selection   = None

    idsToRemove = None
    pointsToAdd = None

    distance    = 3.
    longitude   = 0.
    latitude    = 0.

    running     = True
    points      = True
    delauney    = False
    voronoi     = True
    borders     = True

    shader      = True

    freeze      = True

    repulsion   = 0.00005
    deccelerate = 0.
    
    dmin        = 2.
    temperature = 0.

    def resetStats(self):
        self.dmin = 2.

state = GameState()

while state.running:

    def unprojectToSphereNear( screenPos ):
        worldNear = gluUnProject( screenPos[0], state.resolution[1] - screenPos[1], 0 )
        worldRear = gluUnProject( screenPos[0], state.resolution[1] - screenPos[1], 1 )

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
        
        return None

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
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 1:
            interception = unprojectToSphereNear( e.pos )
            if interception is not None:
                state.selection = np.dot( vertices, interception ).argmax()
            else:
                state.selection = None
        elif e.type == pygame.MOUSEMOTION and e.dict['buttons'][0]:
            interception = unprojectToSphereNear( e.pos )
            if interception is not None and state.selection is not None:
                vertices[state.selection] = interception / np.linalg.norm( interception )
                translations[state.selection] = 0
        elif e.type == pygame.MOUSEMOTION and e.dict['buttons'][1]:
            state.longitude += np.pi / 12 * e.dict['rel'][0]
            state.latitude  += np.pi / 12 * e.dict['rel'][1]
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 4:
            state.distance *= 1.1
        elif e.type == pygame.MOUSEBUTTONDOWN and e.dict['button'] == 5:
            state.distance /= 1.1
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 93: # '+'
            count = pow( 10, mod ) * pow( 100, mod2 )
            state.pointsToAdd = np.random.sample( ( count, 3 ) ).astype( 'float32' ) - 0.5
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 47: # '-'
            count = min( pow( 10, mod ) * pow( 100, mod2 ), vertices.shape[0] - 4 )
            state.idsToRemove = vertices.shape[0] - count + np.arange( count )
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 127: # DEL
            if state.selection is not None and vertices.shape[0] > 4:
                state.idsToRemove = [state.selection]
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'p':
            state.points = not state.points
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'd':
            state.delauney = not state.delauney
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'v':
            state.voronoi = not state.voronoi
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'b':
            state.borders = not state.borders
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 's':
            state.shader = not state.shader
        elif e.type == pygame.KEYDOWN and e.dict['unicode'] == 'f':
            state.freeze = not state.freeze
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 32: # ' '
            state.deccelerate = 0.2 * pow( 10, mod ) * pow( -1, mod2 )
            state.freeze = False
        elif e.type == pygame.KEYUP and e.dict['key'] == 32: # ' '
            state.deccelerate = 0
        elif e.type == pygame.KEYDOWN and e.dict['key'] == 114: # 'r'
            state.repulsion -= 0.000001 * pow( 10, mod ) * pow( -1, mod2 )
            state.repulsion = max( 0, round( state.repulsion, 6 ) )

    if state.idsToRemove is not None:
        vertices = np.delete( vertices, state.idsToRemove, axis = 0 )
        translations = np.delete( translations, state.idsToRemove, axis = 0 )
        if state.selection in state.idsToRemove:
            state.selection = None
        state.idsToRemove = None

    if state.pointsToAdd is not None:
        state.pointsToAdd /= np.linalg.norm( state.pointsToAdd, axis=1 )[:,np.newaxis]
        vertices = np.append( vertices, state.pointsToAdd, axis = 0 )
        translations = np.append( translations, np.zeros_like( state.pointsToAdd ), axis = 0 )
        state.pointsToAdd = None
    
    if not state.freeze:
        state.resetStats()

        def calcRejectionFor( i ):
            diffs = vertices[i] - vertices
            distsSquared = np.square( diffs ).sum( axis = 1 )
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

        projections = np.sum( vertices * translations, axis = 1 )
        translations -= vertices * projections[:,np.newaxis]
        translations *= pow( np.e, -state.deccelerate )
        state.temperature = np.square( translations ).sum()

        vertices += translations
        vertices /= np.linalg.norm( vertices, axis = 1 )[:,np.newaxis]

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

    glLoadIdentity()
    gluPerspective( 45, state.resolution[0] / state.resolution[1], 0.1, 50.0 )
    glTranslate( 0, 0, -state.distance )
    glRotate( state.latitude, 1, 0, 0 )
    glRotate( state.longitude, 0, 1, 0 )
    
    sv = SphericalVoronoi( vertices )
    hull = sv._tri
    sv.sort_vertices_of_regions()

    factor = 1.002
    glScalef( factor, factor, factor )

    verticesBO.set_array( np.append( vertices, sv.vertices ).astype( 'float32' ) )
    
    if state.points:
        with verticesBO:
            glPointSize( 5 )
            glColor4f( 0, 0, 1, 0.5 )
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glDrawArrays( GL_POINTS, 0, vertices.shape[0] )
            if state.selection is not None:
                glColor4f( 1, 0, 0, 1 )
                glDrawArrays( GL_POINTS, state.selection, 1 )
                
    glUseProgram( shaderProgramLines if state.shader else 0 )

    if state.delauney:
        with verticesBO:
            glColor4f( 0, 0, 1, 0.2 )
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            for simplex in hull.simplices:
                glDrawElements( GL_LINE_LOOP, simplex.shape[0], GL_UNSIGNED_INT, simplex )
    
    regions = [np.array(region) + vertices.shape[0] for region in sv.regions]
    selectedRegion = regions[ state.selection ] if state.selection is not None else None
    
    if state.borders:
        with verticesBO:
            glColor4f( 0, 0, 0, 0.2 )
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            for region in regions:
                glDrawElements( GL_LINE_LOOP, len( region ), GL_UNSIGNED_INT, region )
                
    if selectedRegion is not None:
        with verticesBO:
            glColor4f( 1, 0, 0, 1 )
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glDrawElements( GL_LINE_LOOP, len( selectedRegion ), GL_UNSIGNED_INT, selectedRegion )
    
    factor = 1 / factor
    glScalef( factor, factor, factor )

    faces = [ np.concatenate( ( [index], region, [region[0]] ) ).astype( 'uint32' ) for ( index, region ) in zip( it.count(), regions )]
    selectedFace = faces[ state.selection ] if state.selection is not None else None
    
    glUseProgram( shaderProgramTris if state.shader else 0 )
                
    if state.voronoi:
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            for face in faces:
                brightness = 1 - 0.1 * ( face.shape[0] - 5 )
                glColor4f( brightness, brightness, brightness, 1 )
                if not state.shader:
                    face = face[1:]
                glDrawElements( GL_TRIANGLE_FAN, face.shape[0], GL_UNSIGNED_INT, face )
                
    if selectedFace is not None:
        with verticesBO:
            glColor4f( 1, 0, 0, 0.2 )
            if not state.shader:
                selectedFace = selectedFace[1:]
            glDrawElements( GL_TRIANGLE_FAN, selectedFace.shape[0], GL_UNSIGNED_INT, selectedFace )

    glUseProgram( 0 )

    glMatrixMode( GL_PROJECTION )
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D( 0.0, state.resolution[0], 0.0, state.resolution[1] )
    glMatrixMode( GL_MODELVIEW )
    glPushMatrix()
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
                           "Temperature: " + str( int( 1000000 * state.temperature ) ) + "uU²/f²" )

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    pygame.display.flip ()
