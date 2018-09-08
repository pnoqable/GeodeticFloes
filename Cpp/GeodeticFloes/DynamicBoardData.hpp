#pragma once

#include <functional>
#include <thread>
#include <vector>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

#include <Eigen/Dense>

#include <Quickhull.hpp>

struct DynamicBoardData
{
	// nested types:

	typedef std::vector<int> Neighbors;
	typedef std::vector<int> Vertices;

	// attributes

	Eigen::Matrix3Xd nodes;
	std::vector<Neighbors> neighbors;

	Eigen::Matrix3Xd vertices;
	std::vector<Vertices> faces;

	Eigen::Matrix2Xi borders;

	Eigen::Matrix3Xd nextStep;

	// concurrency stuff:
	
	const int threadCount;
	boost::asio::thread_pool workers;

	quickhull::QuickHull<double> qh;

	// methods:

	explicit DynamicBoardData( int nodeCount );
	void addNodes( int delta );
	void removeNode( int nodeId = -1 );

	void updateParallel( int n, std::function< void( int, int ) > f );

	void updateDispersion();
	void updateGeometrie();
};
