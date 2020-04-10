import numpy as np

import pygame
from OpenGL.GL import *
import glm
from pathlib import Path

from simulator.model import Model
from simulator.simulator import Simulator

pygame.init()
pygameFlags = pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF
screen = pygame.display.set_mode( ( 800, 600 ), pygameFlags, 24 )

glEnable( GL_DEPTH_TEST )
glDepthFunc( GL_LEQUAL )

glEnable( GL_BLEND )

glEnable( GL_POLYGON_OFFSET_FILL )
glEnable( GL_POLYGON_OFFSET_LINE )
glPolygonOffset( 1, 1 )

glEnable( GL_POINT_SMOOTH )
glEnable( GL_LINE_SMOOTH )

vao = glGenVertexArrays( 1 )
glBindVertexArray( vao )

vboVertices, vboColors = glGenBuffers( 2 )

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
    def __init__( self, count ):
        self.model       = Model( count )
        self.simulator   = Simulator()

        self.exit        = False

        self.selection   = None

        self.resolution  = None
        self.proj        = None
        self.view        = glm.translate( glm.mat4( 1.0 ), glm.vec3( 0, 0, -3 ) )

        self.running     = True
        self.points      = False
        self.delauney    = False
        self.voronoi     = True
        self.borders     = True

        self.alpha       = 0
        self.shader      = 1
        self.wireframe   = False

    def setResolution( self, res ):
        self.resolution = res
        self.proj = glm.perspective( np.pi/4, res[0] / res[1], 0.1, 10 )

    def mvp( self ):
        return np.array( self.proj * self.view, dtype = float )

    def camera( self ):
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
            eye = -self.camera()
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

state = GameState( 1000 )

while not state.exit:

    for e in pygame.event.get():
        mod = 1 if 'mod' in e.dict and int( e.mod ) & ( 64 + 128 ) else 0 # left or right CTRL
        mod2 = 1 if 'mod' in e.dict and int( e.mod ) & ( 1 + 2 ) else 0 # left or right SHIFT
        if e.type == pygame.QUIT or \
           e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            state.exit = True
        elif e.type == pygame.VIDEORESIZE:
            state.setResolution( ( e.w, e.h ) )
            glViewport( 0, 0, e.w, e.h )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            interception = state.unprojectToSphereNear( e.pos )
            if interception is not None:
                state.selection = state.model.vertexIdAt( interception )
            else:
                state.selection = None
        elif e.type == pygame.MOUSEMOTION and e.buttons == ( 1, 0, 0 ):
            if state.selection is not None:
                interception = state.unprojectToSphereNear( e.pos, True )
                state.model.resetVertex( state.selection, interception )
        elif e.type == pygame.MOUSEMOTION and e.buttons in [ ( 0, 0, 1 ), ( 0, 1, 0 ) ]:
            lastPos = state.unprojectToSphereNear( np.array( e.pos ) - e.rel, True )
            currPos = state.unprojectToSphereNear( np.array( e.pos ), True )
            if lastPos is not None and currPos is not None:
                state.rotate( lastPos, currPos )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 4:
            state.zoom( 1 )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 5:
            state.zoom( -1 )
        elif e.type == pygame.KEYDOWN and e.key == 93: # '+'
            state.model.addVertices( pow( 10, mod ) * pow( 100, mod2 ) )
        elif e.type == pygame.KEYDOWN and e.key == 47: # '-'
            state.model.removeVertices( pow( 10, mod ) * pow( 100, mod2 ) )
            if state.selection is not None and state.selection >= state.model.count():
                state.selection = None
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_DELETE:
            if state.selection is not None and state.model.count() > 4:
                state.model.removeVertexIds( [state.selection] )
                state.selection = None
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
            state.model.resetAllVertices( mod )
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
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
            state.shader -= pow( 8, mod ) * pow( -1, mod2 )
            state.shader = max( 1, min( 8, state.shader ) )
        elif e.type == pygame.KEYDOWN and e.unicode == 'w':
            state.wireframe = not state.wireframe
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_f:
            state.simulator.friction -= 10 * pow( 10, mod ) * pow( -1, mod2 )
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            state.simulator.repulsion -= 0.000001 * pow( 10, mod ) * pow( -1, mod2 )
            state.simulator.repulsion = max( 0, round( state.simulator.repulsion, 6 ) )
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            if mod:
                state.simulator.steps -= pow( 10, mod2 )
            else:
                state.simulator.steps = abs( state.simulator.steps - pow( -1, mod2 ) )

    state.simulator.simulate( state.model )
    
    if state.model.needsUpdate():
        state.model.updateGeometry()
        glBindBuffer( GL_ARRAY_BUFFER, vboVertices )
        glBufferData( GL_ARRAY_BUFFER, state.model.allVertices.nbytes, state.model.allVertices, GL_DYNAMIC_DRAW )
        glVertexPointer( 3, GL_FLOAT, 0, GLvoidp( 0 ) )
        glEnableClientState( GL_VERTEX_ARRAY )

    # todo: pass degrees to shader and calculate color there
    brightnesses = 1.3 - 0.1 * state.model.degrees[:,np.newaxis]
    alphas = state.alpha + ( 1 - state.alpha ) * brightnesses
    colors = np.hstack( ( brightnesses, brightnesses, brightnesses, alphas ) ).astype( 'float32' )
    glBindBuffer( GL_ARRAY_BUFFER, vboColors )
    glBufferData( GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW )
    glColorPointer( 4, GL_FLOAT, 0, GLvoidp( 0 ) )

    def render():
        if state.selection is not None:
            selectedBorder = state.model.borders[state.selection]
            selectedTris = state.model.tris[state.selection]
        else:
            selectedBorder = None
            selectedTris = None

        glClearColor( 0.2, 0.4, 0.4, 1.0 )
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
        glDepthMask( GL_TRUE )
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE if state.wireframe else GL_FILL )
        
        glUseProgram( shaderProgramTris )
        glUniformMatrix4fv( shaderProgramTrisView, 1, False, glm.value_ptr( state.view ) )
        glUniformMatrix4fv( shaderProgramTrisProj, 1, False, glm.value_ptr( state.proj ) )
        glUniform1f( shaderProgramTrisSides, state.shader )
        glUniform1i( shaderProgramTrisMinOut, 1 )

        if state.voronoi:
            zOrder = np.argsort( np.dot( state.model.vertices, state.camera() ) )
            allTris = np.concatenate( state.model.tris[zOrder] )
            glLineWidth( 1 )
            glEnableClientState( GL_COLOR_ARRAY )
            glDrawElements( GL_TRIANGLES, allTris.size, GL_UNSIGNED_INT, allTris )
            glDisableClientState(  GL_COLOR_ARRAY )

        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
                    
        if selectedTris is not None:
            glColor4f( 1, 0, 0, 0.2 )
            glDrawElements( GL_TRIANGLES, selectedTris.shape[0], GL_UNSIGNED_INT, selectedTris )
        
        glUseProgram( shaderProgramLines )
        glUniformMatrix4fv( shaderProgramLinesView, 1, False, glm.value_ptr( state.view ) )
        glUniformMatrix4fv( shaderProgramLinesProj, 1, False, glm.value_ptr( state.proj ) )
        glUniform1f( shaderProgramLinesSides, state.shader )
        glUniform1i( shaderProgramLinesMinOut, 1 )

        if selectedBorder is not None:
            glLineWidth( 2 )
            glColor4f( 1, 0, 0, 0.5 )
            glDrawElements( GL_LINE_LOOP, selectedBorder.size, GL_UNSIGNED_INT, selectedBorder )

        if state.borders:
            allBorders = state.model.allBorders
            glLineWidth( 1 )
            glColor4f( 0, 0, 0, 0.2 if not state.wireframe or not state.voronoi else 1 )
            glDrawElements( GL_LINES, allBorders.size, GL_UNSIGNED_INT, allBorders )

        glDepthMask( GL_FALSE )

        scaledView = glm.scale( state.view, glm.vec3( 1.002, 1.002, 1.002 ) )
        glUniformMatrix4fv( shaderProgramLinesView, 1, False, glm.value_ptr( scaledView ) )
        glUniform1i( shaderProgramLinesMinOut, 2 )

        if state.delauney:
            allLinks = state.model.allLinks
            glLineWidth( 1 )
            glColor4f( 0, 0, 1, 0.2 )
            glDrawElements( GL_LINES, allLinks.size, GL_UNSIGNED_INT, allLinks )

        glDepthMask( GL_TRUE )
        
        glUseProgram( shaderProgramPoints )
        glUniformMatrix4fv( shaderProgramPointsView, 1, False, glm.value_ptr( scaledView ) )
        glUniformMatrix4fv( shaderProgramPointsProj, 1, False, glm.value_ptr( state.proj ) )

        if state.points:
            glPointSize( 5 )
            glColor4f( 0, 0, 1, 0.5 )
            glDrawArrays( GL_POINTS, 0, state.model.count() )

            if state.selection is not None:
                glPointSize( 6 )
                glColor4f( 0.8, 0, 0, 1 )
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

        drawText( (0,0,0), 24, "Simulation: " + str( state.simulator.steps ) + "\n"
                               "Vertices: " + str( state.model.count() ) + "\n"
                               "Repulsion: " + str( round( 1000000 * state.simulator.repulsion ) ) + "uf^-2\n"
                               "Friction: " + str( state.simulator.friction ) + "fU^-2\n"
                               "Temperature: " + str( int( 1000000 * state.model.temperature() ) ) + "uU²f²" )
    
    render()

    pygame.display.flip()
