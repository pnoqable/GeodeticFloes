import glm
import numpy as np
from OpenGL.GL import *
from pathlib import Path

class Shader:
    def __init__( self, type, filename ):
        self.id = glCreateShader( type )
        glShaderSource( self.id, Path( __file__ ).with_name( filename ).read_text() )
        glCompileShader( self.id )

        if glGetShaderiv( self.id, GL_COMPILE_STATUS ) != GL_TRUE:
            raise RuntimeError( glGetShaderInfoLog( self.id ).decode() )

class Program:
    def __init__( self, shaders, uniforms ):
        self.id = glCreateProgram()

        for shader in shaders:
            glAttachShader( self.id, shader.id )
        
        glLinkProgram( self.id )

        if glGetProgramiv( self.id, GL_LINK_STATUS ) != GL_TRUE:
            raise RuntimeError(  glGetProgramInfoLog( self.id ).decode() )

        self.uniforms = {}
        for uniform in uniforms:
            self.uniforms[uniform] = glGetUniformLocation( self.id, uniform )
    
    def __getitem__( self, key ):
        return self.uniforms[key]

class Renderer:
    @staticmethod
    def setupGL():
        glEnable( GL_DEPTH_TEST )
        glDepthFunc( GL_LEQUAL )

        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        glEnable( GL_POLYGON_OFFSET_FILL )
        glEnable( GL_POLYGON_OFFSET_LINE )
        glPolygonOffset( 1, 1 )

        glEnable( GL_POINT_SMOOTH )
        glEnable( GL_LINE_SMOOTH )

    @staticmethod
    def setViewport( width, height ):
        glViewport( 0, 0, width, height )

    def __init__( self, points = False, links = False, voronoi = True, borders = True, alpha = 0, shader = 1, wireframe = False ):
        self.points    = points
        self.links     = links
        self.voronoi   = voronoi
        self.borders   = borders

        self.alpha     = alpha
        self.shader    = shader
        self.wireframe = wireframe

        self.vao = glGenVertexArrays( 1 )
        self.vboVertices, self.vboColors = glGenBuffers( 2 )

        self.shaders = {}
        self.shaders["vertex"] = Shader( GL_VERTEX_SHADER, "vertex.glsl" )
        self.shaders["lines"] = Shader( GL_GEOMETRY_SHADER, "lines.glsl" )
        self.shaders["tris"] = Shader( GL_GEOMETRY_SHADER, "tris.glsl" )

        self.programs = {}
        self.programs["points"] = Program( [self.shaders["vertex"]], ["view", "proj"] )
        self.programs["lines"] = Program( [self.shaders["vertex"], self.shaders["lines"]],
                                          ["view", "proj", "sides", "minOut"] )
        self.programs["tris"] = Program( [self.shaders["vertex"], self.shaders["tris"]],
                                         ["view", "proj", "sides", "minOut"] )

    def setVertices( self, vertices ):
        glBindVertexArray( self.vao )
        glBindBuffer( GL_ARRAY_BUFFER, self.vboVertices )
        glBufferData( GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW )
        glVertexPointer( 3, GL_FLOAT, 0, GLvoidp( 0 ) )

    def setColors( self, colors ):
        glBindVertexArray( self.vao )
        glBindBuffer( GL_ARRAY_BUFFER, self.vboColors )
        glBufferData( GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW )
        glColorPointer( 4, GL_FLOAT, 0, GLvoidp( 0 ) )
    
    def updateColors( self, degrees ):
        # todo: pass degrees to shader and calculate color there
        brightnesses = 1.3 - 0.1 * degrees.astype( 'float32' )
        rgbs = np.repeat( brightnesses[:,np.newaxis], 3, axis = 1 )
        alphas = self.alpha + ( 1 - self.alpha ) * brightnesses[:,np.newaxis]
        colors = np.append( rgbs, alphas, axis = 1 )
        self.setColors( colors )

    def renderVoronoi( self, camera, tris, indices, selection ):
        glUseProgram( self.programs["tris"].id )
        glUniformMatrix4fv( self.programs["tris"]["view"], 1, False, glm.value_ptr( camera.view ) )
        glUniformMatrix4fv( self.programs["tris"]["proj"], 1, False, glm.value_ptr( camera.proj ) )
        glUniform1f( self.programs["tris"]["sides"], self.shader )
        glUniform1i( self.programs["tris"]["minOut"], 1 )

        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE if self.wireframe else GL_FILL )
        glLineWidth( 1 )

        glEnableClientState( GL_COLOR_ARRAY )
        indexedTris = np.concatenate( tris if indices is None else  tris[indices] )
        glDrawElements( GL_TRIANGLES, indexedTris.size, GL_UNSIGNED_INT, indexedTris )
        glDisableClientState(  GL_COLOR_ARRAY )

        if selection is not None and ( indices is None or selection in indices ):
            glColor4f( 1, 0, 0, 0.2 )
            selectedTris = tris[selection]
            glDrawElements( GL_TRIANGLES, selectedTris.size, GL_UNSIGNED_INT, selectedTris )

        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ) 

    def renderBorders( self, camera, borders, indices, selection ):
        glUseProgram( self.programs["lines"].id )
        glUniformMatrix4fv( self.programs["lines"]["view"], 1, False, glm.value_ptr( camera.view ) )
        glUniformMatrix4fv( self.programs["lines"]["proj"], 1, False, glm.value_ptr( camera.proj ) )
        glUniform1f( self.programs["lines"]["sides"], self.shader )
        glUniform1i( self.programs["lines"]["minOut"], 1 )
        
        glColor4f( 0, 0, 0, 0.2 if not self.wireframe or not self.voronoi else 1 )
        indexedBorders = np.concatenate( borders if indices is None else borders[indices] )
        glDrawElements( GL_LINES, indexedBorders.size, GL_UNSIGNED_INT, indexedBorders )
        
        if selection is not None and ( indices is None or selection in indices ):
            glLineWidth( 2 )
            glColor4f( 1, 0, 0, 0.5 )
            selectedBorders = borders[selection]
            glDrawElements( GL_LINES, selectedBorders.size, GL_UNSIGNED_INT, selectedBorders )
            glLineWidth( 1 )
    
    def renderLinks( self, camera, links, indices, selection ):
        glUseProgram( self.programs["lines"].id )
        scaledView = glm.scale( camera.view, glm.vec3( 1.002, 1.002, 1.002 ) )
        glUniformMatrix4fv( self.programs["lines"]["view"], 1, False, glm.value_ptr( scaledView ) )
        glUniformMatrix4fv( self.programs["lines"]["proj"], 1, False, glm.value_ptr( camera.proj ) )
        glUniform1f( self.programs["lines"]["sides"], self.shader )
        glUniform1i( self.programs["lines"]["minOut"], 2 )

        glColor4f( 0, 0, 1, 0.2 )
        indexedLinks = np.concatenate( links if indices is None else links[indices] )
        glDrawElements( GL_LINES, indexedLinks.size, GL_UNSIGNED_INT, indexedLinks )
    
    def renderPoints( self, camera, count, indices, selection ):
        glUseProgram( self.programs['points'].id )
        scaledView = glm.scale( camera.view, glm.vec3( 1.002, 1.002, 1.002 ) )
        glUniformMatrix4fv( self.programs['points']['view'], 1, False, glm.value_ptr( scaledView ) )
        glUniformMatrix4fv( self.programs['points']['proj'], 1, False, glm.value_ptr( camera.proj ) )

        glPointSize( 5 )
        glColor4f( 0, 0, 1, 0.5 )
        indexedPoints = np.arange( count ) if indices is None else indices
        glDrawElements( GL_POINTS, indexedPoints.size, GL_UNSIGNED_INT, indexedPoints )

        if selection is not None and ( indices is None or selection in indices ):
            glPointSize( 6 )
            glColor4f( 0.8, 0, 0, 1 )
            glDrawArrays( GL_POINTS, selection, 1 )

        glPointSize( 1 )
    
    def render( self, camera, model, selection = None ):
        glClearColor( 0.2, 0.4, 0.4, 1.0 )
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        glBindVertexArray( self.vao )
        glEnableClientState( GL_VERTEX_ARRAY )

        cameraPos = camera.pos()
        depths = np.dot( model.vertices, cameraPos )
        zOrder = np.argsort( depths )
        horizon = np.searchsorted( depths[zOrder], 1 )

        indicesBack  = zOrder[:horizon]
        indicesFront = zOrder[horizon:]

        glDepthMask( GL_FALSE )

        if len( indicesBack ) > 0:
            if self.voronoi:
                self.renderVoronoi( camera, model.tris, indicesBack, selection )
            if self.borders:
                self.renderBorders( camera, model.borders, indicesBack, selection )
            if self.links:
                self.renderLinks( camera, model.links, indicesBack, selection )
            if self.points:
                self.renderPoints( camera, model.count(), indicesBack, selection )

        if len( indicesFront ) > 0:
            if self.voronoi:
                glDepthMask( GL_TRUE )
                self.renderVoronoi( camera, model.tris, indicesFront, selection )
                glDepthMask( GL_FALSE )
            if self.borders:
                self.renderBorders( camera, model.borders, indicesFront, selection )
            if self.links:
                self.renderLinks( camera, model.links, indicesFront, selection )
            if self.points:
                self.renderPoints( camera, model.count(), indicesFront, selection )

        glDepthMask( GL_TRUE )
        
    def drawPixels( self, camera, pos, width, height, pixels ):
        glLoadMatrixf( camera.ortho() )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE )
        glRasterPos3fv( pos )     
        glDrawPixels( width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
