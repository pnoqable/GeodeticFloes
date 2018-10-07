#include "DynamicBoardData.hpp"

#include <boost/math/constants/constants.hpp>

#include <condition_variable>
#include <map>
#include <mutex>

const double pi = boost::math::constants::pi<double>();

DynamicBoardData::DynamicBoardData( int nodeCount ) :
	threadCount( std::thread::hardware_concurrency() ),
	workers( threadCount ),
	writeStats( false )
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

Eigen::Vector3d project( const Eigen::Vector3d& v, const Eigen::Vector3d& n ) {
	return v - n.dot( v ) * n;
}

struct Statistics {

	std::vector<std::stringstream> stats;
	DynamicBoardData& data;

	Statistics( DynamicBoardData& data ) :
		data( data )
	{
		if( data.writeStats ) {
			stats.resize( data.nodes.cols() );
			data.writeStats = false;
		}
	}
	
	void addIfEnabled( int i, 
		const Eigen::Vector3d& pos,
		const Eigen::Matrix3Xd& rejections,
		const Eigen::RowVectorXd& norms )
	{
		if( i < stats.size() ) {
			auto& ss = stats[i];

			Eigen::RowVectorXd rejectionsNorm = Eigen::RowVectorXd::Zero( stats.size() );
			for( int j = 0; j < stats.size(); j++ ) {
				rejectionsNorm( 0, j ) = project( rejections.col( j ), pos ).norm();
			}

			ss << " " << rejectionsNorm << " | " << norms << std::endl;
		}
	}

	~Statistics() {
		for( auto& stream : stats ) {
			std::cout << stream.str();
		}
	}
};

void DynamicBoardData::updateDispersion() {
	Statistics stats( *this );
	const int n = nodes.cols();
	updateParallel( n, [this, n, &stats]( int min, int max ) {
		for( int i = min; i < max; i++ ) {
			auto pos = nodes.col( i );
			auto differences = nodes.colwise() - pos;
			auto squareNorms = differences.colwise().squaredNorm().unaryExpr([]( float x ) { return x ? x : 1; });
			auto norms = squareNorms.array().sqrt();
			auto directions = differences.array() / norms.replicate<3, 1>();
			auto rejections = directions.array() / squareNorms.array().replicate<3, 1>();
			auto rejection = rejections.rowwise().sum();
			nextStep.col( i ) -= 0.1 / sqrt( n ) * rejection.matrix();
			nextStep.col( i ) = 0.5 / sqrt( n ) * project( nextStep.col( i ), nodes.col( i ) );
			stats.addIfEnabled( i, nodes.col( i ), rejections, norms );
		}
	} );
	updateParallel( nodes.cols(), [this]( int min, int max ) {
		int cols = max - min;
		nodes.middleCols( min, cols ) += nextStep.middleCols( min, cols );
		nodes.middleCols( min, cols ).colwise().normalize();
	} );
}

void DynamicBoardData::updateGeometrie() {
	auto hull = qh.getConvexHullAsMesh( nodes.data(), nodes.cols(), true );

	int nodeCount = nodes.cols() > 3 ? nodes.cols() : 0;
	int vertCount = nodeCount ? hull.m_faces.size() : 0;
	int borderCount = nodeCount ? hull.m_halfEdges.size() : 0;

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

	// set neighbors, faces and borders:
	neighbors = std::vector<Neighbors>( nodeCount );
	faces = std::vector<Vertices>( nodeCount );
	borders = Eigen::Matrix2Xi( 2, borderCount );
	std::vector<std::mutex> m( nodeCount );
	updateParallel( borderCount, [this,&hull,&m]( int min, int max ) {
		for( int i = min; i < max ; i++ ) {
			auto& edge = hull.m_halfEdges[i];
			auto& from = edge.m_endVertex;
			auto& to = hull.m_halfEdges[edge.m_opp].m_endVertex;
			auto& vertexA = edge.m_face;
			auto& vertexB = hull.m_halfEdges[edge.m_opp].m_face;
			borders.col( i ) = Eigen::Vector2i( vertexA, vertexB );
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
