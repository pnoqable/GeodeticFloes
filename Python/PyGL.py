import pygame

from simulator.model import Model
from simulator.simulator import Simulator
from renderer.camera import Camera
from renderer.renderer import Renderer

pygame.init()
pygameFlags = pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF
screen = pygame.display.set_mode( ( 800, 600 ), pygameFlags, 24 )

Renderer.setupGL()

class GameState:
    def __init__( self, count ):
        self.model       = Model( count )
        self.simulator   = Simulator()
        self.camera      = Camera()
        self.renderer    = Renderer()

        self.exit        = False

        self.selection   = None

state = GameState( 1000 )

while not state.exit:

    for e in pygame.event.get():
        mod = 1 if 'mod' in e.dict and int( e.mod ) & ( 64 + 128 ) else 0 # left or right CTRL
        mod2 = 1 if 'mod' in e.dict and int( e.mod ) & ( 1 + 2 ) else 0 # left or right SHIFT
        if e.type == pygame.QUIT or \
           e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            state.exit = True
        elif e.type == pygame.VIDEORESIZE:
            state.camera.setResolution( e.w, e.h )
            state.renderer.setViewport( e.w, e.h )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            interception = state.camera.unprojectToSphereNear( e.pos )
            if interception is not None:
                state.selection = state.model.vertexIdAt( interception )
            else:
                state.selection = None
        elif e.type == pygame.MOUSEMOTION and e.buttons == ( 1, 0, 0 ):
            if state.selection is not None:
                interception = state.camera.unprojectToSphereNear( e.pos, True )
                state.model.resetVertex( state.selection, interception )
        elif e.type == pygame.MOUSEMOTION and e.buttons in [ ( 0, 0, 1 ), ( 0, 1, 0 ) ]:
            pos = tuple( pos - rel for pos, rel in zip( e.pos, e.rel ) ) 
            lastPos = state.camera.unprojectToSphereNear( pos, True )
            currPos = state.camera.unprojectToSphereNear( e.pos, True )
            if lastPos is not None and currPos is not None:
                state.camera.rotate( lastPos, currPos )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 4:
            state.camera.zoom( 1 )
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 5:
            state.camera.zoom( -1 )
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
            state.renderer.points = not state.renderer.points
        elif e.type == pygame.KEYDOWN and e.unicode == 'l':
            state.renderer.links = not state.renderer.links
        elif e.type == pygame.KEYDOWN and e.unicode == 'v':
            state.renderer.voronoi = not state.renderer.voronoi
        elif e.type == pygame.KEYDOWN and e.unicode == 'b':
            state.renderer.borders = not state.renderer.borders
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_a:
            state.renderer.alpha -= 0.1 * pow( 10, mod ) * pow( -1, mod2 )
            state.renderer.alpha = max( 0, min( 1, state.renderer.alpha ) )
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
            state.renderer.shader -= pow( 8, mod ) * pow( -1, mod2 )
            state.renderer.shader = max( 1, min( 8, state.renderer.shader ) )
        elif e.type == pygame.KEYDOWN and e.unicode == 'w':
            state.renderer.wireframe = not state.renderer.wireframe
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
        state.renderer.setVertices( state.model.allVertices )
        state.renderer.setDegrees( state.model.degrees )

    state.renderer.render( state.camera, state.model, state.selection )

    text = [
        "Simulation: " + str( state.simulator.steps ),
        "Vertices: " + str( state.model.count() ),
        "Repulsion: " + str( round( 1000000 * state.simulator.repulsion ) ) + "uf^-2",
        "Friction: " + str( state.simulator.friction ) + "fU^-2",
        "Temperature: " + str( int( 1000000 * state.model.temperature() ) ) + "uU²f²"
    ]

    font = pygame.font.Font( None, 24 )
    pos = [0., 0., 0.]
    for line in reversed( text ):
        surface = font.render( line, True, ( 255, 255, 255, 255 ), ( 0, 0, 0, 0 ) )
        width, height = surface.get_width(), surface.get_height()
        pixels  = pygame.image.tostring( surface, "RGBA", True )
        state.renderer.drawPixels( state.camera, pos, width, height, pixels )
        pos[1] += height

    pygame.display.flip()
