import numpy as np
from scipy.spatial import SphericalVoronoi

import pygame
from OpenGL.GL import *
from OpenGL.arrays import vbo
import glm
import itertools as it
from pathlib import Path

pygame.init()
pygameFlags = pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF
screen = pygame.display.set_mode( ( 800, 600 ), pygameFlags, 24 )

np.random.seed()
vertices = np.random.sample( ( 1000, 3 ) ).astype( 'float32' ) - 0.5
vertices /= np.linalg.norm( vertices, axis = 1 )[:,np.newaxis]
translations = np.zeros( vertices.shape, dtype = 'float32' )

glEnable( GL_DEPTH_TEST )
glDepthFunc( GL_LEQUAL )

glEnable( GL_BLEND )

glEnable( GL_POLYGON_OFFSET_FILL )
glEnable( GL_POLYGON_OFFSET_LINE )
glPolygonOffset( 1, 1 )

glEnable( GL_CULL_FACE )

glEnable( GL_POINT_SMOOTH )
glEnable( GL_LINE_SMOOTH )

verticesBO = vbo.VBO( np.zeros( (0,3), dtype = 'float32' ) )
glEnableClientState( GL_VERTEX_ARRAY )

vtxShader = glCreateShader( GL_VERTEX_SHADER )
glShaderSource( vtxShader, Path( __file__ ).with_name( 'vertex.glsl' ).read_text() )
glCompileShader( vtxShader )

if glGetShaderiv( vtxShader, GL_COMPILE_STATUS ) != GL_TRUE:
    raise RuntimeError( glGetShaderInfoLog( vtxShader ).decode() )

shaderProgramPoints = glCreateProgram()
glAttachShader( shaderProgramPoints, vtxShader )
glLinkProgram( shaderProgramPoints )

if glGetProgramiv( shaderProgramPoints, GL_LINK_STATUS ) != GL_TRUE:
    raise RuntimeError(  glGetProgramInfoLog( shaderProgramPoints ).decode() )

shaderProgramPointsView = glGetUniformLocation( shaderProgramPoints, "view" )
shaderProgramPointsProj = glGetUniformLocation( shaderProgramPoints, "proj" )

lineShader = glCreateShader( GL_GEOMETRY_SHADER )
glShaderSource( lineShader, Path( __file__ ).with_name( 'lines.glsl' ).read_text() )
glCompileShader( lineShader )

if glGetShaderiv( lineShader, GL_COMPILE_STATUS ) != GL_TRUE:
    raise RuntimeError(glGetShaderInfoLog( lineShader ).decode() )

shaderProgramLines = glCreateProgram()
glAttachShader( shaderProgramLines, vtxShader )
glAttachShader( shaderProgramLines, lineShader )
glLinkProgram( shaderProgramLines )

if glGetProgramiv( shaderProgramLines, GL_LINK_STATUS ) != GL_TRUE:
    raise RuntimeError( glGetProgramInfoLog( shaderProgramLines ).decode() )

shaderProgramLinesView = glGetUniformLocation( shaderProgramLines, "view" )
shaderProgramLinesProj = glGetUniformLocation( shaderProgramLines, "proj" )
shaderProgramLinesSides = glGetUniformLocation( shaderProgramLines, "sides" )
shaderProgramLinesMinOut = glGetUniformLocation( shaderProgramLines, "minOut" )

triShader = glCreateShader( GL_GEOMETRY_SHADER )
glShaderSource( triShader, Path( __file__ ).with_name( 'tris.glsl' ).read_text() )
glCompileShader( triShader )

if glGetShaderiv( triShader, GL_COMPILE_STATUS ) != GL_TRUE:
    raise RuntimeError( glGetShaderInfoLog( triShader ).decode() )

shaderProgramTris = glCreateProgram()
glAttachShader( shaderProgramTris, vtxShader )
glAttachShader( shaderProgramTris, triShader )
glLinkProgram( shaderProgramTris )

if glGetProgramiv( shaderProgramTris, GL_LINK_STATUS ) != GL_TRUE:
    raise RuntimeError( glGetProgramInfoLog( shaderProgramTris ) )

shaderProgramTrisView = glGetUniformLocation( shaderProgramTris, "view" )
shaderProgramTrisProj = glGetUniformLocation( shaderProgramTris, "proj" )
shaderProgramTrisSides = glGetUniformLocation( shaderProgramTris, "sides" )
shaderProgramTrisMinOut = glGetUniformLocation( shaderProgramTris, "minOut" )

class GameState:
    selection   = None
    idsToRemove = None
    pointsToAdd = None

    frames      = 0
    repulsion   = 0.00005
    deccelerate = 0
    
    dmin        = 2
    temperature = 0

    resolution  = None
    proj        = None
    view        = None

    running     = True
    points      = False
    delauney    = False
    voronoi     = True
    borders     = True

    alpha       = 0
    culling     = False
    maskClear   = True
    maskWrite   = True
    shader      = 1
    wireframe   = False

    def __init__( self ):
        self.view = glm.translate( glm.mat4(), ( 0, 0, -3 ) )

    def setResolution( self, res ):
        self.resolution = res
        self.proj = glm.perspective( np.pi/4, res[0] / res[1], 0.1, 10 )

    def mvp( self ):
        return np.array( self.proj * self.view, dtype = float )

    def camera( self ):
        invMvp = np.linalg.inv( self.mvp() )
        return invMvp[2,:3] / invMvp[2,3]

    def rotate( self, a, b ):
        a = glm.normalize( a )
        b = glm.normalize( b )
        cosArc = glm.dot( a, b )
        if cosArc <= -1:
            self.view = glm.scale( self.view, ( -1, -1, -1 ) )
        elif cosArc < 1:
            angle = glm.acos( cosArc )
            axis = glm.cross( a, b ) 
            self.view = glm.rotate( self.view, angle, axis )

    def zoom( self, exp ):
        self.view[3,2] *= glm.pow( 1.1, exp )

    def ortho( self ):
        ortho = glm.ortho( 0, self.resolution[0], 0, self.resolution[1], 0, 1 )
        return np.array( ortho, dtype = float )

    def resetStats( self ):
        self.dmin = 2

state = GameState()

while state.running:

    def unproject( screenPos ):
        screenPos = np.append( screenPos, [1] ).astype( float )
        screenPos[:2] /= state.resolution
        screenPos[:3] = 2 * screenPos[:3] - 1
        invMvp = np.linalg.inv( state.mvp() )
        worldPos = invMvp.T.dot( screenPos )
        return worldPos[:3] / worldPos[3]

    def unprojectToSphereNear( screenPos, snapToSphere = False ):
        screenPos = np.array( screenPos )
        screenPos[1] = state.resolution[1] - screenPos[1]
        worldNear = unproject( np.append( screenPos, [0] ) )
        worldRear = unproject( np.append( screenPos, [1] ) )

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
            eye = -state.camera()
            if np.dot( res, eye ) > 0:
                res = np.cross( eye, d )
                res = np.cross( res, eye )
            return res / np.linalg.norm( res )
        
        return None

    for e in pygame.event.get():
        mod = 1 if 'mod' in e.dict and int( e.mod ) & ( 64 + 128 ) else 0 # left or right CTRL
        mod2 = 1 if 'mod' in e.dict and int( e.mod ) & ( 1 + 2 ) else 0 # left or right SHIFT
        if e.type == pygame.QUIT or \
           e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            state.running = False
        elif e.type == pygame.VIDEORESIZE:
            state.setResolution( ( e.w, e.h ) )
            glViewport( 0, 0, e.w, e.h )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            interception = unprojectToSphereNear( e.pos )
            if interception is not None:
                state.selection = np.dot( vertices, interception ).argmax()
            else:
                state.selection = None
        elif e.type == pygame.MOUSEMOTION and e.buttons == ( 1, 0, 0 ):
            interception = unprojectToSphereNear( e.pos, True )
            if interception is not None and state.selection is not None:
                vertices[state.selection] = interception / np.linalg.norm( interception )
                translations[state.selection] = 0
                if 'sv' in globals(): del sv
        elif e.type == pygame.MOUSEMOTION and e.buttons in [ ( 0, 0, 1 ), ( 0, 1, 0 ) ]:
            lastPos = unprojectToSphereNear( np.array( e.pos ) - e.rel, True )
            currPos = unprojectToSphereNear( np.array( e.pos ), True )
            if lastPos is not None and currPos is not None:
                state.rotate( lastPos, currPos )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 4:
            state.zoom( 1 )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 5:
            state.zoom( -1 )
        elif e.type == pygame.KEYDOWN and e.key == 93: # '+'
            count = pow( 10, mod ) * pow( 100, mod2 )
            state.pointsToAdd = np.random.sample( ( count, 3 ) ).astype( 'float32' ) - 0.5
        elif e.type == pygame.KEYDOWN and e.key == 47: # '-'
            count = min( pow( 10, mod ) * pow( 100, mod2 ), vertices.shape[0] - 4 )
            state.idsToRemove = vertices.shape[0] - count + np.arange( count )
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_DELETE:
            if state.selection is not None and vertices.shape[0] > 4:
                state.idsToRemove = [state.selection]
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
            indices = np.arange( vertices.shape[0] ).astype( 'float32' ) + 0.5
            phi = np.arccos( 1 - 2 * indices / indices.shape[0] )
            theta = np.pi * ( 1 + 5 ** 0.5 ) * indices
            vertices = np.array( [ np.cos( theta ) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi) ] ).T
            translations[:] = 0
            if 'sv' in globals(): del sv
        elif e.type == pygame.KEYDOWN and e.unicode == 'p':
            state.points = not state.points
        elif e.type == pygame.KEYDOWN and e.unicode == 'd':
            state.delauney = not state.delauney
        elif e.type == pygame.KEYDOWN and e.unicode == 'v':
            state.voronoi = not state.voronoi
        elif e.type == pygame.KEYDOWN and e.unicode == 'b':
            state.borders = not state.borders
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_a:
            state.alpha -= 0.1 * pow( 10, mod ) * pow( -1, mod2 )
            state.alpha = max( 0, min( 1, state.alpha ) )
        elif e.type == pygame.KEYDOWN and e.unicode == 'c':
            state.culling = not state.culling
        elif e.type == pygame.KEYDOWN and e.unicode == 'm':
            state.maskWrite = not state.maskWrite
        elif e.type == pygame.KEYDOWN and e.unicode == 'M':
            state.maskClear = not state.maskClear
            state.maskWrite = state.maskClear
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
            state.shader -= pow( 8, mod ) * pow( -1, mod2 )
            state.shader = max( 1, min( 8, state.shader ) )
        elif e.type == pygame.KEYDOWN and e.unicode == 'w':
            state.wireframe = not state.wireframe
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_f:
            state.frames -= pow( 10, mod ) * pow( -1, mod2 )
            state.frames = max( 0, state.frames )
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            state.deccelerate = 0.2 * pow( 10, mod ) * pow( -1, mod2 )
            if state.frames == 0:
                state.frames = 1
        elif e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
            state.deccelerate = 0
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            state.repulsion -= 0.000001 * pow( 10, mod ) * pow( -1, mod2 )
            state.repulsion = max( 0, round( state.repulsion, 6 ) )

    if state.idsToRemove is not None:
        vertices = np.delete( vertices, state.idsToRemove, axis = 0 )
        translations = np.delete( translations, state.idsToRemove, axis = 0 )
        if state.selection in state.idsToRemove:
            state.selection = None
        state.idsToRemove = None
        if 'sv' in globals(): del sv

    if state.pointsToAdd is not None:
        state.pointsToAdd /= np.linalg.norm( state.pointsToAdd, axis=1 )[:,np.newaxis]
        vertices = np.append( vertices, state.pointsToAdd, axis = 0 )
        translations = np.append( translations, np.zeros_like( state.pointsToAdd ), axis = 0 )
        state.pointsToAdd = None
        if 'sv' in globals(): del sv
    
    for _ in range( state.frames ):
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

    if state.frames or 'sv' not in globals():
    
        sv = SphericalVoronoi( vertices )

        sv.sort_vertices_of_regions()
        for region in sv.regions:
            a = sv.vertices[region[0]]
            b = sv.vertices[region[1]]
            c = sv.vertices[region[2]]
            if np.sum( np.cross( b - a, c - b ) * b ) < 0:
                region.reverse()

        verticesBO.set_array( np.append( vertices, sv.vertices ).astype( 'float32' ) )

        hull = sv._tri
        
        regions = np.array( [np.array(region).astype( 'uint32' ) for region in sv.regions] ) + vertices.shape[0]

        def face( i, region ): return np.concatenate( ( [i], region, [region[0]] ) )
        faces = np.array( [ face( i, region ) for ( i, region ) in zip( it.count(), regions ) ] )
    
    selectedRegion = regions[state.selection] if state.selection is not None else None
    selectedFace = faces[ state.selection ] if state.selection is not None else None

    glClearColor( 0.2, 0.4, 0.4, 1.0 )
    glClear( GL_COLOR_BUFFER_BIT )

    if state.maskClear:
        glDepthMask( GL_TRUE )
        glClear( GL_DEPTH_BUFFER_BIT )
    
    if state.culling:
        glEnable( GL_CULL_FACE )
    else:
        glDisable( GL_CULL_FACE )

    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
    glDepthMask( state.maskWrite )
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE if state.wireframe else GL_FILL )
       
    glUseProgram( shaderProgramTris )
    glUniformMatrix4fv( shaderProgramTrisView, 1, False, glm.value_ptr( state.view ) )
    glUniformMatrix4fv( shaderProgramTrisProj, 1, False, glm.value_ptr( state.proj ) )
    glUniform1f( shaderProgramTrisSides, state.shader )
    glUniform1i( shaderProgramTrisMinOut, 1 )

    if state.voronoi:
        zOrder = np.argsort( np.dot( vertices, state.camera() ) )
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glLineWidth( 1 )
            for face in faces[zOrder]:
                brightness = 1.5 - 0.1 * face.shape[0]
                alpha = state.alpha - brightness * state.alpha + brightness
                glColor4f( brightness, brightness, brightness, alpha )
                glDrawElements( GL_TRIANGLE_FAN, face.shape[0], GL_UNSIGNED_INT, face )

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
                
    if selectedFace is not None:
        with verticesBO:
            glColor4f( 1, 0, 0, 0.2 )
            glDrawElements( GL_TRIANGLE_FAN, selectedFace.shape[0], GL_UNSIGNED_INT, selectedFace )
    
    glUseProgram( shaderProgramLines )
    glUniformMatrix4fv( shaderProgramLinesView, 1, False, glm.value_ptr( state.view ) )
    glUniformMatrix4fv( shaderProgramLinesProj, 1, False, glm.value_ptr( state.proj ) )
    glUniform1f( shaderProgramLinesSides, state.shader )
    glUniform1i( shaderProgramLinesMinOut, 1 )

    if selectedRegion is not None:
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glLineWidth( 2 )
            glColor4f( 1, 0, 0, 0.5 )
            glDrawElements( GL_LINE_LOOP, len( selectedRegion ), GL_UNSIGNED_INT, selectedRegion )

    if state.borders:
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glLineWidth( 1 )
            glColor4f( 0, 0, 0, 0.2 if not state.wireframe or not state.voronoi else 1 )
            for region in regions:
                glDrawElements( GL_LINE_LOOP, len( region ), GL_UNSIGNED_INT, region )

    glDepthMask( GL_FALSE )

    scaledView = glm.scale( state.view, ( 1.002, 1.002, 1.002 ) )
    glUniformMatrix4fv( shaderProgramLinesView, 1, False, glm.value_ptr( scaledView ) )
    glUniform1i( shaderProgramLinesMinOut, 2 )

    if state.delauney:
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glLineWidth( 1 )
            glColor4f( 0, 0, 1, 0.2 )
            for simplex in hull.simplices:
                glDrawElements( GL_LINE_LOOP, simplex.shape[0], GL_UNSIGNED_INT, simplex )

    glDepthMask( state.maskWrite )
    
    glUseProgram( shaderProgramPoints )
    glUniformMatrix4fv( shaderProgramPointsView, 1, False, glm.value_ptr( scaledView ) )
    glUniformMatrix4fv( shaderProgramPointsProj, 1, False, glm.value_ptr( state.proj ) )
                
    if state.points:
        with verticesBO:
            glVertexPointer( 3, GL_FLOAT, 0, verticesBO )
            glPointSize( 5 )
            glColor4f( 0, 0, 1, 0.5 )
            glDrawArrays( GL_POINTS, 0, vertices.shape[0] )

            if state.selection is not None:
                glPointSize( 6 )
                glColor4f( 0.9, 0, 0, 1 )
                glDrawArrays( GL_POINTS, state.selection, 1 )

    glUseProgram( 0 )
    glLoadMatrixf( state.ortho() )
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
                           "Dmin: " + str( int( 1000000 * state.dmin ) ) + "uU\n" + \
                           "Temperature: " + str( int( 1000000 * state.temperature ) ) + "uU²/f²" )

    pygame.display.flip ()
