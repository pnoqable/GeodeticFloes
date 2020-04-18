import numpy as np
import re
from OpenGL.GL import *
from pathlib import Path

def buildShader( filename, constants = {}, feedbackVaryings = [] ):

    source = Path( __file__ ).with_name( filename ).read_text()

    for name, value in constants.items():
        source = re.sub( name, str( value ), source )
    
    shader = glCreateShader( GL_VERTEX_SHADER )
    glShaderSource( shader, source )
    glCompileShader( shader )

    if glGetShaderiv( shader, GL_COMPILE_STATUS ) != GL_TRUE:
        raise RuntimeError( glGetShaderInfoLog( shader ).decode() )

    program = glCreateProgram()
    glAttachShader( program, shader )

    buff = ( ctypes.c_char_p * len( feedbackVaryings ) )()
    buff[:] = [ string.encode( "utf-8" ) for string in feedbackVaryings ]
    cBuff = ctypes.cast( buff, ctypes.POINTER( ctypes.POINTER( GLchar ) ) )
    glTransformFeedbackVaryings( program, len( buff ), cBuff, GL_SEPARATE_ATTRIBS )

    glLinkProgram( program )
    glUseProgram( program )

    if glGetProgramiv( program, GL_LINK_STATUS ) != GL_TRUE:
        raise RuntimeError(  glGetProgramInfoLog( program ).decode() )
    
    return program

class Simulator:
    def __init__( self, friction = 100, repulsion = 5e-06, steps = 0 ):
        self.friction  = friction
        self.repulsion = repulsion
        self.steps     = steps

        self._maxVerticesCount = glGetInteger( GL_MAX_VERTEX_UNIFORM_VECTORS ) - 1
        self._maxRejectionsCount = glGetInteger( GL_MAX_VARYING_VECTORS ) - 2

        # set up rejection shader

        self._rejectionProgram = buildShader( "rejection.glsl",
                                              { "maxVerticesCount" : self._maxVerticesCount },
                                              [ "rejection" ] )

        self._verticesCount = glGetUniformLocation( self._rejectionProgram, "verticesCount" )
        self._vertices = glGetUniformBlockIndex( self._rejectionProgram, "Vertices" )

        self._vao = glGenVertexArrays( 1 )
        self._vboVertices, self._vboRejections = glGenBuffers( 2 )

        self._vertex = glGetAttribLocation( self._rejectionProgram, "vertex" )

    def _simulateStep( self, model, vertices, rejections ):
    
        glUseProgram( self._rejectionProgram )
        glBufferSubData( GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices )
        glBindBufferBase( GL_TRANSFORM_FEEDBACK_BUFFER, 0, self._vboRejections )

        glBeginTransformFeedback( GL_POINTS )
        offset = 0
        for segment in np.array_split( vertices, rejections.shape[0] ):
            glUniform1i( self._verticesCount, segment.shape[0] )
            glUniformBlockBinding( self._rejectionProgram, self._vertices, 0 ) # needed here?
            glBindBufferRange( GL_UNIFORM_BUFFER, 0, self._vboVertices, offset, segment.nbytes )
            offset += segment.nbytes
            glDrawArrays( GL_POINTS, 0, vertices.shape[0] )
        glEndTransformFeedback()
        
        glFlush()

        glGetBufferSubData( GL_TRANSFORM_FEEDBACK_BUFFER, 0, rejections.nbytes, rejections.data )

        model.translations += self.repulsion * rejections.sum( axis = 0 )[:,:3]

        projections = np.sum( model.vertices * model.translations, axis = 1 )
        model.translations -= model.vertices * projections[:,np.newaxis]

        translationsSquared = np.square( model.translations ).sum( axis = 1 )
        model.translations *= np.power( np.e, -self.friction*translationsSquared )[:,np.newaxis]

        model.vertices += model.translations
        model.vertices /= np.linalg.norm( model.vertices, axis = 1 )[:,np.newaxis]

        model.invalidate()

    def simulate( self, model ):

        rejectionsCount = int( np.ceil( model.count() / self._maxVerticesCount ) )
        assert rejectionsCount <= self._maxRejectionsCount

        vertices = np.empty( ( model.count(), 4 ), dtype = np.float32 )
        vertices[:,:3] = model.vertices
        vertices[:,3] = 0

        glBindVertexArray( self._vao )

        glBindBuffer( GL_ARRAY_BUFFER, self._vboVertices )
        glBufferData( GL_ARRAY_BUFFER, vertices.nbytes, None, GL_DYNAMIC_DRAW )
        
        glEnableVertexAttribArray( self._vertex )
        glVertexAttribPointer( self._vertex, 4, GL_FLOAT, GL_FALSE, 0, GLvoidp( 0 ) )
        
        rejections = np.empty( ( rejectionsCount, model.count(), 4 ), dtype = np.float32 )

        glBindBuffer( GL_TRANSFORM_FEEDBACK_BUFFER, self._vboRejections )
        glBufferData( GL_TRANSFORM_FEEDBACK_BUFFER, rejections.nbytes, None, GL_DYNAMIC_READ )

        glEnable( GL_RASTERIZER_DISCARD )

        for _ in range( 1 if self.steps < 0 else self.steps ):
            self._simulateStep( model, vertices, rejections )

        glDisable( GL_RASTERIZER_DISCARD )
        
        if self.steps < 0:
            self.steps += 1
