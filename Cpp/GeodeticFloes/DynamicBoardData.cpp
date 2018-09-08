#include "DynamicBoardData.hpp"

#include <boost/math/constants/constants.hpp>

#include <condition_variable>
#include <map>
#include <mutex>

const double pi = boost::math::constants::pi<double>();

DynamicBoardData::DynamicBoardData( int nodeCount ) :
	threadCount( std::thread::hardware_concurrency() ),
	workers( threadCount )
{
	nodes = Eigen::Matrix3Xd::Random( 3, nodeCount );
	nodes.colwise().normalize();
	nextStep = Eigen::Matrix3Xd::Zero( 3, nodeCount );
}

void DynamicBoardData::addNodes( int delta ) {
	int count = nodes.cols();

	if( delta < -count ) {
		delta = -count;
	}

	nodes.conservativeResize( Eigen::NoChange, count + delta );
	nextStep.conservativeResizeLike( nodes );

	if( delta > 0 ) {
		nodes.rightCols( delta ) = Eigen::Matrix3Xd::Random( 3, delta );
		nodes.rightCols( delta ).colwise().normalize();
		nextStep.rightCols( delta ).colwise() = Eigen::Vector3d::Zero();
	}
}

void DynamicBoardData::removeNode( int nodeId ) {
	nodeId = ( nodeId + nodes.cols() ) % nodes.cols();
	int rightCols = nodes.cols() - nodeId - 1;
	if( rightCols > 0 ) {
		// \todo: need temporal veriable?
		nodes.middleCols( nodeId, rightCols ) = nodes.rightCols( rightCols );
		nextStep.middleCols( nodeId, rightCols ) = nextStep.rightCols( rightCols );
	}
	nodes.conservativeResize( Eigen::NoChange, nodes.cols() - 1 );
	nextStep.conservativeResize( Eigen::NoChange, nodes.cols() - 1 );
}

void DynamicBoardData::updateParallel( int n, std::function< void( int, int ) > f ) {
	std::mutex m;
	std::unique_lock<std::mutex> lock( m );
	std::condition_variable c;
	int running = 0;
	for( int i = 0; i<threadCount; i++ ) {
		int min = n * i++/threadCount;
		int max = n * i--/threadCount;
		if( min != max ) {
			running++;
			boost::asio::post( workers, [&,f,min,max] {
				f( min, max );
				std::unique_lock<std::mutex> lock( m );
				running--;
				c.notify_one();
			} );
		}
	}
	c.wait( lock, [&]{ return 0 == running; } );
}

void DynamicBoardData::updateGeometrie() {
	auto hull = qh.getConvexHullAsMesh( nodes.data(), nodes.cols(), true );

	int nodeCount = nodes.cols() > 3 ? nodes.cols() : 0;
	int vertCount = nodeCount ? hull.m_faces.size() : 0;
	int edgeCount = nodeCount ? hull.m_halfEdges.size() : 0;

	// copy node data back (because quickhull reorders data):
	memcpy( nodes.data(), hull.m_vertices.data(),
			3 * nodeCount * sizeof( double ) );

	// calculate vertices:
	vertices = Eigen::Matrix3Xd::Zero( 3, vertCount );
	updateParallel( vertCount, [this,&hull]( int min, int max ) {
		for( size_t i = min; i < max; i++ ) {
			size_t v[3], e = hull.m_faces[i].m_halfEdgeIndex;
			for( int j = 0; j < 3; j++ ) {
				v[j] = hull.m_halfEdges[e].m_endVertex;
				e = hull.m_halfEdges[e].m_next;
			}
			// next of next of next vertex should be first:
			assert( e == hull.m_faces[i].m_halfEdgeIndex );
			auto& a = nodes.col( v[1] ) - nodes.col( v[0] );
			auto& b = nodes.col( v[2] ) - nodes.col( v[1] );
			vertices.col( i ) = a.cross( b );
		}
		vertices.middleCols( min, max - min ).colwise().normalize();
	} );

	// set neighbors, faces and edges:
	neighbors = std::vector<Neighbors>( nodeCount );
	faces = std::vector<Vertices>( nodeCount );
	edges = Eigen::Matrix2Xi( 2, edgeCount );
	std::vector<std::mutex> m( nodeCount );
	updateParallel( edgeCount, [this,&hull,&m]( int min, int max ) {
		for( int i = min; i < max ; i++ ) {
			auto& edge = hull.m_halfEdges[i];
			auto& from = edge.m_endVertex;
			auto& to = hull.m_halfEdges[edge.m_opp].m_endVertex;
			auto& vertexA = edge.m_face;
			auto& vertexB = hull.m_halfEdges[edge.m_opp].m_face;
			edges.col( i ) = Eigen::Vector2i( vertexA, vertexB );
			// lock before modifying std::vectors:
			std::lock_guard<std::mutex> lock( m[from] );
			neighbors[from].push_back( to );
			faces[from].push_back( vertexA );
		}
	} );

	// sort face vertices counterclockwise
	updateParallel( nodeCount, [this,&hull]( int min, int max ) {
		for( int i = min; i < max; i++ ) {
			auto& faceVertices = faces[i];
			assert( faceVertices.size() > 2 );
			auto& m = nodes.col( i );
			auto& f = vertices.col( faceVertices.front() );
			auto r = f.cross( m ).normalized();
			auto u = m.cross( r ).normalized();
			std::map<int,double> angles;
			for( auto& v : faceVertices ) {
				auto& p = vertices.col( v );
				double x = r.dot( p );
				double y = u.dot( p );
				double r = sqrt( x*x + y*y );
				double angle = x >= 0 ? acos( y / r ) : 2 * pi - acos( y / r );
				angles.emplace( v, angle );
			}
			std::sort( faceVertices.begin(), faceVertices.end(), [&angles]( int x, int y ) {
				return angles[x] < angles[y];
			} );
		}
	} );
}
