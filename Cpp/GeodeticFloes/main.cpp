#include <iostream>

#define _WIN32_WINNT 0x0501 // minimum deployment target (WinXP)

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include "DynamicBoard.hpp"

int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow ) 
{
	sf::Window window( sf::VideoMode( 800, 600 ), "OpenGL", sf::Style::Default, sf::ContextSettings( 32 ) );
	window.setVerticalSyncEnabled( true );
	window.setActive( true );

	auto onResize = []( sf::Vector2u size ) {
		glViewport( 0, 0, size.x, size.y );
		glLoadIdentity();
		gluPerspective( 45, GLdouble( size.x ) / size.y, 0.1, 50.0 );
		glTranslatef( 0, 0, -3 );
	};

	onResize( window.getSize() );

	GLenum err = glewInit();
	if( GLEW_OK != err ) {
		std::cerr << "Error: " << glewGetErrorString( err ) << std::endl;
	}

	glClearColor( 0.0, 0.5, 0.5, 1.0 );
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LESS );

	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	struct GameState {
		bool running = true;
		bool updating = false;
		bool drawCenters = false;
		bool drawNeighbors = false;
		bool drawVertices = true;
		bool drawBorders = true;
		bool drawFaces = true;
		bool occludeLines = true;

		DynamicBoard board;

		explicit GameState( unsigned int numFaces )
			: board( numFaces ) {
		}

		sf::Event::MouseMoveEvent mouseMotion( sf::Event::MouseMoveEvent pos ) {
			static sf::Event::MouseMoveEvent last = pos;
			sf::Event::MouseMoveEvent result{ pos.x - last.x, pos.y - last.y };
			last = pos;
			return result;
		}
	};

	GameState state( 100 );

	while( state.running ) {

		sf::Event event;
		while( window.pollEvent( event ) ) {
			auto keyPressed = [&]( sf::Keyboard::Key key ) {
				return event.type == sf::Event::KeyPressed
					&& event.key.code == key;
			};

			struct OneOfKeysHeld {
				bool operator()( sf::Keyboard::Key key ) {
					return sf::Keyboard::isKeyPressed( key );
				};
				bool operator()( sf::Keyboard::Key key, sf::Keyboard::Key keys... ) {
					return (*this)( key ) | (*this)( keys );
				}
			};

			static OneOfKeysHeld oneOfKeysHeld;

			static auto getMultiplier = [] {
				return ( oneOfKeysHeld( sf::Keyboard::LShift, sf::Keyboard::RShift ) ? 10 : 1 )
				     * ( oneOfKeysHeld( sf::Keyboard::LControl, sf::Keyboard::RControl ) ? 100 : 1 );
			};

			if( event.type == sf::Event::Closed || keyPressed( sf::Keyboard::Escape ) ) {
				state.running = false;
			} else if( event.type == sf::Event::Resized ) {
				onResize( { event.size.width, event.size.height } );
			} else if( keyPressed( sf::Keyboard::Space ) ) {
				state.updating = !state.updating;
			} else if( keyPressed( sf::Keyboard::C ) ) {
				state.drawCenters = !state.drawCenters;
			} else if( keyPressed( sf::Keyboard::N ) ) {
				state.drawNeighbors = !state.drawNeighbors;
			} else if( keyPressed( sf::Keyboard::V ) ) {
				state.drawVertices = !state.drawVertices;
			} else if( keyPressed( sf::Keyboard::B ) ) {
				state.drawBorders = !state.drawBorders;
			} else if( keyPressed( sf::Keyboard::F ) ) {
				state.drawFaces = !state.drawFaces;
			} else if( keyPressed( sf::Keyboard::O ) ) {
				state.occludeLines = !state.occludeLines;
			} else if( keyPressed( sf::Keyboard::Equal ) ) {
				state.board.addFaces( getMultiplier() );
			} else if( keyPressed( sf::Keyboard::Dash ) ) {
				state.board.addFaces( -getMultiplier() );
			} else if( event.type == sf::Event::MouseMoved ) {
				auto rel = state.mouseMotion( event.mouseMove );
				if( sf::Mouse::isButtonPressed( sf::Mouse::Button::Left ) ) {
					glRotated( rel.x, 0, 1, 0 );
				}
			}
		}

		if( state.updating ) {
			state.board.updateDispersion();
		}

		state.board.updateGeometryIfNeeded();

		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		if( state.drawCenters ) {
			glPointSize( 5 );
			glColor4d( 0, 0, 1, 0.5 );
			glBegin( GL_POINTS );
			for( int i = 0; i < state.board.faceCount(); i++ ) {
				glVertex3dv( state.board.faceCenter( i ) );
			}
			glEnd();
		}

		if( state.drawVertices ) {
			glPointSize( 5 );
			glColor4d( 1, 0, 0, 0.5 );
			glBegin( GL_POINTS );
			for( int i = 0; i < state.board.vertexCount(); i++ ) {
				glVertex3dv( state.board.vertexPosition( i ) );
			}
			glEnd();
		}

		if( state.drawFaces ) {
			for( int i = 0; i < state.board.faceCount(); i++ ) {
				auto& vertices = state.board.faceVertices( i );
				glScaled( 0.999, 0.999, 0.999 );
				glBegin( GL_POLYGON );
				if( vertices.size() % 2 ) {
					glColor4f( 0.5, 0.5, 0.5, 1 );
				} else {
					glColor4f( 1, 1, 1, 1 );
				}
				for( int v : vertices ) {
					glVertex3dv( state.board.vertexPosition( v ) );
				}
				glEnd();
				glScaled( 1. / 0.999, 1. / 0.999, 1. / 0.999 );
			}
		}

		if( state.drawNeighbors ) {
			if( !state.occludeLines ) {
				glDisable( GL_DEPTH_TEST );
			}
			glColor4d( 0, 0, 1, 0.5 );
			glBegin( GL_LINES );
			for( int from = 0; from < state.board.faceCount(); from++ ) {
				for ( int to : state.board.faceNeighbors( from ) ) {
					if( from < to ) {
						glVertex3dv( state.board.faceCenter( from ) );
						glVertex3dv( state.board.faceCenter( to ) );
					}
				}
			}
			glEnd();
			glEnable( GL_DEPTH_TEST );
		}

		if( state.drawBorders ) {
			if( !state.occludeLines ) {
				glDisable( GL_DEPTH_TEST );
			}
			glBegin( GL_LINES );
			glColor4d( 0, 0, 0, 1 );
			for( int i = 0; i < state.board.borderCount(); i++ ) {
				const auto& edge = state.board.borderVertices( i );
				if( edge->from() < edge->to() ) {
					glVertex3dv( state.board.vertexPosition( edge->from() ) );
					glVertex3dv( state.board.vertexPosition( edge->to() ) );
				}
			}
			glEnd();
			glEnable( GL_DEPTH_TEST );
		}

		window.display();
	}

	return 0;
} 
