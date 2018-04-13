#include <iostream>
#include <fstream>

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include <boost/version.hpp>
#include <boost/date_time.hpp>
#include <boost/regex.hpp>

#include <Eigen/Dense>

#include <boost/math/constants/constants.hpp>

#undef min // undo somebody's mess with global namespace
#undef max

std::ofstream logStream("log.txt");
std::ofstream errStream("err.txt");

const double pi = boost::math::constants::pi<double>();

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) 
{
	std::cout.rdbuf(logStream.rdbuf());
	std::cerr.rdbuf(errStream.rdbuf());

	sf::Window window(sf::VideoMode(800, 600), "OpenGL", sf::Style::Default, sf::ContextSettings(32));
	window.setVerticalSyncEnabled(true);
	window.setActive(true);

	auto onResize = [](sf::Vector2u size) {
		glViewport(0, 0, size.x, size.y);
		glLoadIdentity();
		gluPerspective(45, GLdouble(size.x) / size.y, 0.1, 50.0);
		glTranslatef(0, 0, -5);
	};

	onResize( window.getSize() );

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}

	glClearColor(0.0, 0.5, 0.5, 1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	struct GameState {
		bool running = true;
		bool drawPoints = true;
		bool rotate = false;

		Eigen::Matrix3Xd points;
		Eigen::Matrix3Xd translations;

		GameState( unsigned int numFaces ) {
			points = Eigen::Matrix3Xd::Random(3, numFaces);
			points.colwise().normalize();
			translations = Eigen::Matrix3Xd::Zero(3, numFaces);
		}

		Eigen::Vector2i mouseMotion(Eigen::Vector2i cur) {
			static Eigen::Vector2i last = cur;
			Eigen::Vector2i result = cur - last;
			last = cur;
			return result;
		}
	};

	GameState state( 800 );
	while (state.running) {

		sf::Event event;
		while (window.pollEvent(event)) {
			auto keyPressed = [&](sf::Keyboard::Key key) {
				return event.type == sf::Event::KeyPressed
					&& event.key.code == key;
			};

			if (event.type == sf::Event::Closed ||
				keyPressed(sf::Keyboard::Escape)) {
				state.running = false;
			} else if (event.type == sf::Event::Resized) {
				onResize({ event.size.width, event.size.height });
			} else if (keyPressed(sf::Keyboard::P)) {
				state.drawPoints = !state.drawPoints;
			} else if (event.type == sf::Event::MouseMoved) {
				auto& pos = event.mouseMove;
				auto rel = state.mouseMotion({ pos.x, pos.y });
				if (state.rotate) {
					glRotated(pi / 12 * rel.x(), 0, 1, 0);
				}
			} else if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Button::Left) {
					state.rotate = true;
				}
			} else if(event.type == sf::Event::MouseButtonReleased) {
				if (event.mouseButton.button == sf::Mouse::Button::Left) {
					state.rotate = false;
				}
			}
		}

		auto project = [](const auto& v, const auto& n) {
			return v - n.dot(v) * n;
		};

		for (int i = 0; i < state.points.cols(); i++) {
			auto pos = state.points.array().col(i);
			auto differences = state.points.array().colwise() - pos;
			//std::cout << "differences:" << std::endl << differences << std::endl;
			auto squareNorms = differences.matrix().colwise().squaredNorm().unaryExpr([](float x) { return x ? x : 1; });
			//std::cout << "squareNorms:" << std::endl << squareNorms << std::endl;
			auto directions = differences.array() / squareNorms.array().replicate<3,1>().sqrt();
			//std::cout << "directions:" << std::endl << directions << std::endl;
			auto rejections  = directions.array() / squareNorms.array().max( 0.1 ).replicate<3,1>();
			//std::cout << "rejections:" << std::endl << rejections << std::endl;
			auto rejection = rejections.rowwise().sum();
			//std::cout << "rejection:" << std::endl << rejection << std::endl;
			state.translations.col(i) -= 0.01 * rejection.matrix();
			state.translations.col(i) = 0.2 * project(state.translations.col(i), state.points.col(i));
		}

		state.points += state.translations;
		state.points.colwise().normalize();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (state.drawPoints) {
			glEnable(GL_BLEND);
			glPointSize(5);
			glColor4f(0, 0, 1, 0.5);
			glBegin(GL_POINTS);
			for (int i = 0; i < state.points.cols(); i++) {
				glVertex3d(state.points(0,i), state.points(1,i), state.points(2,i));
			}
			glEnd();
			glDisable(GL_BLEND);
		}

		window.display();
	}

	return 0;
} 
