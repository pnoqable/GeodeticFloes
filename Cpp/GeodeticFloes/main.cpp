#include <fstream>
#include <iostream>
#include <thread>

#define _WIN32_WINNT 0x0501 // minimum deployment target (WinXP)

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/math/constants/constants.hpp>

#include <windows.h>
#include <GL/glew.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include <Eigen/Dense>

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
		glTranslatef(0, 0, -3);
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

		void addPoints(int delta) {
			int count = points.cols();

			if (delta < -count) {
				delta = -count;
			}

			points.conservativeResize(Eigen::NoChange, count + delta);
			translations.conservativeResize(Eigen::NoChange, count + delta);

			if (delta > 0) {
				points.rightCols(delta) = Eigen::Matrix3Xd::Random(3, delta);
				points.rightCols(delta).colwise().normalize();
				translations.rightCols(delta).colwise() = Eigen::Vector3d::Zero();
			}
		}
	};

	GameState state( 100 );

	while (state.running) {

		sf::Event event;
		while (window.pollEvent(event)) {
			auto keyPressed = [&](sf::Keyboard::Key key) {
				return event.type == sf::Event::KeyPressed
					&& event.key.code == key;
			};

			struct OneOfKeysHeld {
				bool operator()(sf::Keyboard::Key key) {
					return sf::Keyboard::isKeyPressed(key);
				};
				bool operator()(sf::Keyboard::Key key, sf::Keyboard::Key keys...) {
					return (*this)(key) | (*this)(keys);
				}
			};

			static OneOfKeysHeld oneOfKeysHeld;

			static auto getMultiplier = [] {
				return (oneOfKeysHeld(sf::Keyboard::LShift, sf::Keyboard::RShift) ? 10 : 1)
				     * (oneOfKeysHeld(sf::Keyboard::LControl, sf::Keyboard::RControl) ? 100 : 1);
			};

			if (event.type == sf::Event::Closed ||
				keyPressed(sf::Keyboard::Escape)) {
				state.running = false;
			} else if (event.type == sf::Event::Resized) {
				onResize({ event.size.width, event.size.height });
			} else if (keyPressed(sf::Keyboard::P)) {
				state.drawPoints = !state.drawPoints;
			} else if (keyPressed(sf::Keyboard::Equal)) {
				state.addPoints(getMultiplier());
			} else if (keyPressed(sf::Keyboard::Dash)) {
				state.addPoints(-getMultiplier());
			} else if (event.type == sf::Event::MouseMoved) {
				auto& pos = event.mouseMove;
				auto rel = state.mouseMotion({ pos.x, pos.y });
				if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
					glRotated(pi / 12 * rel.x(), 0, 1, 0);
				}
			}
		}

		static auto project = [](const auto& v, const auto& n) {
			return v - n.dot(v) * n;
		};

		{
			int t = std::thread::hardware_concurrency();

			boost::asio::thread_pool worker(t);
			int n = state.points.cols();

			for (int j=0; j<t; j++) {
				boost::asio::post(worker, [&, j] {
					for (int i=n*j/ t; i<n*(j+1)/ t; i++) {
						auto pos = state.points.col(i);
						auto differences = state.points.colwise() - pos;
						auto squareNorms = differences.colwise().squaredNorm().unaryExpr([](float x) { return x ? x : 1; });
						auto directions = differences.array() / squareNorms.array().replicate<3, 1>().sqrt();
						auto rejections = directions.array() / squareNorms.array().replicate<3, 1>();
						auto rejection = rejections.rowwise().sum();
						state.translations.col(i) -= 0.1 / sqrt(n) * rejection.matrix();
						state.translations.col(i) = 0.5 / sqrt(n) * project(state.translations.col(i), state.points.col(i));
					}
				});
			}

			worker.join();
		}

		state.points += state.translations;
		state.points.colwise().normalize();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (state.drawPoints) {
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glPointSize(3);
			glColor4d(0, 0, 1, 0.5);
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
