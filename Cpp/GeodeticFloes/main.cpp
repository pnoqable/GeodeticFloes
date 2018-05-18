#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#define _WIN32_WINNT 0x0501 // minimum deployment target (WinXP)

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/math/constants/constants.hpp>

#include <windows.h>
//#include <GL/glew.h>
#include <gl/GLU.h>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include <Eigen/Dense>

#include <QuickHull.hpp>

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

	//GLenum err = glewInit();
	//if (GLEW_OK != err) {
	//	std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	//}

	glClearColor(0.0, 0.5, 0.5, 1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	struct GameState {
		bool running = true;
		bool updating = false;
		bool drawPoints = false;
		bool drawDelaunay = false;
		bool drawVoronoi = true;
		bool occludeLines = true;

		Eigen::Matrix3Xd particles;
		Eigen::Matrix3Xd translations;

		explicit GameState( unsigned int numFaces ) {
			particles = Eigen::Matrix3Xd::Random(3, numFaces);
			particles.colwise().normalize();
			translations = Eigen::Matrix3Xd::Zero(3, numFaces);
		}

		Eigen::Vector2i mouseMotion(Eigen::Vector2i cur) {
			static Eigen::Vector2i last = cur;
			Eigen::Vector2i result = cur - last;
			last = cur;
			return result;
		}

		void addPoints(int delta) {
			int count = particles.cols();

			if (delta < -count) {
				delta = -count;
			}

			particles.conservativeResize(Eigen::NoChange, count + delta);
			translations.conservativeResize(Eigen::NoChange, count + delta);

			if (delta > 0) {
				particles.rightCols(delta) = Eigen::Matrix3Xd::Random(3, delta);
				particles.rightCols(delta).colwise().normalize();
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
			} else if (keyPressed(sf::Keyboard::Space)) {
				state.updating = !state.updating;
			} else if (keyPressed(sf::Keyboard::P)) {
				state.drawPoints = !state.drawPoints;
			} else if (keyPressed(sf::Keyboard::D)) {
				state.drawDelaunay = !state.drawDelaunay;
			} else if (keyPressed(sf::Keyboard::V)) {
				state.drawVoronoi = !state.drawVoronoi;
			} else if (keyPressed(sf::Keyboard::O)) {
				state.occludeLines = !state.occludeLines;
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

		if (state.updating) {
			{
				int t = std::thread::hardware_concurrency();

				boost::asio::thread_pool worker(t);
				int n = state.particles.cols();

				for (int j = 0; j<t; j++) {
					boost::asio::post(worker, [&, j] {
						for (int i = n*j / t; i<n*(j + 1) / t; i++) {
							auto pos = state.particles.col(i);
							auto differences = state.particles.colwise() - pos;
							auto squareNorms = differences.colwise().squaredNorm().unaryExpr([](float x) { return x ? x : 1; });
							auto directions = differences.array() / squareNorms.array().replicate<3, 1>().sqrt();
							auto rejections = directions.array() / squareNorms.array().replicate<3, 1>();
							auto rejection = rejections.rowwise().sum();
							state.translations.col(i) -= 0.1 / sqrt(n) * rejection.matrix();
							state.translations.col(i) = 0.5 / sqrt(n) * project(state.translations.col(i), state.particles.col(i));
						}
					});
				}

				worker.join();
			}

			state.particles += state.translations;
			state.particles.colwise().normalize();

		}

		quickhull::QuickHull<double> qh;
		auto hull = qh.getConvexHullAsMesh(state.particles.data(), state.particles.cols(), true);

		struct Board {
			typedef quickhull::HalfEdgeMesh<double, size_t> ConvexHull;
			typedef std::vector<int> Neighbors;

			struct Vertex {
				int id;
				double angle;
				Vertex(int id, double angle) : id(id), angle(angle) {}
				bool operator<(const Vertex& other) const {
					return angle < other.angle;
				}
			};
			
			typedef std::vector<Vertex> Face;

			Eigen::Matrix3Xd nodes;
			std::vector<Neighbors> neighbors;

			Eigen::Matrix3Xd vertices;
			std::vector<Face> faces;

			explicit Board(const ConvexHull& hull) {

				// add nodes:
				int nodesCount = hull.m_vertices.size();
				nodes = Eigen::Map<Eigen::Matrix3Xd>((double*)hull.m_vertices.data(), 3, nodesCount);

				int nodeCount = nodes.cols() > 3 ? nodes.cols() : 0;
				int vertCount = nodeCount ? hull.m_faces.size() : 0;

				// calculate vertices:
				// \todo parallelize this:
				vertices = Eigen::Matrix3Xd::Zero(3, vertCount);
				for (size_t i = 0; i < vertCount; i++) {
					size_t v[3], e = hull.m_faces[i].m_halfEdgeIndex;
					for (int j=0; j<3; j++) {
						v[j] = hull.m_halfEdges[e].m_endVertex;
						e = hull.m_halfEdges[e].m_next;
					}
					// next of next of next vertex should be first:
					assert(e == hull.m_faces[i].m_halfEdgeIndex);
					auto& a = nodes.col(v[1]) - nodes.col(v[0]);
					auto& b = nodes.col(v[2]) - nodes.col(v[1]);
					vertices.col(i) = a.cross(b);
				}
				vertices.colwise().normalize();

				// set neighbors and face vertices:
				// \todo parallelize this:
				neighbors.resize(nodeCount);
				faces.resize(nodeCount);
				if (nodeCount) {
					for (auto& edge : hull.m_halfEdges) {
						auto& from = edge.m_endVertex;
						auto& to = hull.m_halfEdges[edge.m_opp].m_endVertex;
						neighbors[from].push_back(to);
						faces[from].push_back(Vertex(edge.m_face, 0.));
					}
				}

				// sort face vertices counterclockwise
				// \todo parallelize this:
				for (int i = 0; i < nodeCount; i++) {
					auto& vertIDs = faces[i];
					assert(vertIDs.size() > 2);
					auto& m = nodes.col(i);
					auto& f = vertices.col(vertIDs.front().id);
					auto r = f.cross(m).normalized();
					auto u = m.cross(r).normalized();
					for (auto& v : vertIDs) {
						auto& p = vertices.col(v.id);
						double x = r.dot(p);
						double y = u.dot(p);
						double r = sqrt(x*x + y*y);
						v.angle = x >= 0 ? acos(y / r) : 2 * pi - acos(y / r);
					}
					std::sort(vertIDs.begin(), vertIDs.end());
				}
			}
		};

		Board board(hull);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (state.drawPoints) {
			glPointSize(5);
			glColor4d(0, 0, 1, 0.5);
			glBegin(GL_POINTS);
			for (int i = 0; i < board.nodes.cols(); i++) {
				glVertex3dv(board.nodes.col(i).data());
			}
			glEnd();
		}

		if (state.drawVoronoi) {
			for (auto& face : board.faces) {
				glBegin(GL_LINE_LOOP);
				glColor4d(0, 0, 0, 1);
				for (auto& node : face) {
					glVertex3dv(board.vertices.col(node.id).data());
				}
				glEnd();
				glScaled(0.999, 0.999, 0.999);
				glBegin(GL_POLYGON);
				if (face.size() % 2) {
					glColor4f(0.5, 0.5, 0.5, 1);
				} else {
					glColor4f(1, 1, 1, 1);
				}
				for (auto& node : face) {
					glVertex3dv(board.vertices.col(node.id).data());
				}
				glEnd();
				glScaled(1. / 0.999, 1. / 0.999, 1. / 0.999);
			}
		}

		if (state.drawDelaunay) {
			if (!state.occludeLines) {
				glDisable(GL_DEPTH_TEST);
			}
			glColor4d(0, 0, 1, 0.5);
			glBegin(GL_LINES);
			for (int from = 0; from < board.neighbors.size(); from++) {
				for (int to : board.neighbors[from]) {
					if (from < to) {
						glVertex3dv(board.nodes.col(from).data());
						glVertex3dv(board.nodes.col(to).data());
					}
				}
			}
			glEnd();
			glEnable(GL_DEPTH_TEST);
		}

		window.display();
	}

	return 0;
} 
