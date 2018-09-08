#include "DynamicBoard.hpp"
#include "DynamicBoardData.hpp"

DynamicBoard::DynamicBoard( int faceCount ) :
	data( new DynamicBoardData( faceCount ) ),
	needsUpdate( true ) {
}

DynamicBoard::~DynamicBoard() {
	delete data;
}

const DynamicBoardData& DynamicBoard::internalData() const {
	throwIfUpdateNeeded();
	return *data;
}

const int DynamicBoard::faceCount() const {
	return data->nodes.cols();
}

const DynamicBoard::Neighbors& DynamicBoard::faceNeighbors( int faceId ) const {
	throwIfUpdateNeeded();
	return data->neighbors[faceId];
}

const double* DynamicBoard::faceCenters() const {
	return data->nodes.data();
}

const double* DynamicBoard::faceCenter( int faceId ) const {
	return data->nodes.col( faceId ).data();
}

const DynamicBoard::Vertices& DynamicBoard::faceVertices( int faceId ) const {
	throwIfUpdateNeeded();
	return data->faces[faceId];
}

const int DynamicBoard::vertexCount() const {
	throwIfUpdateNeeded();
	return data->vertices.cols();
}

const double* DynamicBoard::vertexPositions() const {
	throwIfUpdateNeeded();
	return data->vertices.data();
}

const double* DynamicBoard::vertexPosition( int nodeId ) const {
	throwIfUpdateNeeded();
	return data->vertices.col( nodeId ).data();
}

const int DynamicBoard::edgeCount() const {
	throwIfUpdateNeeded();
	return data->edges.cols();
}

const DynamicBoard::Edge* DynamicBoard::edgeVertices() const {
	throwIfUpdateNeeded();
	return (Edge*) data->edges.data();
}

const DynamicBoard::Edge* DynamicBoard::edgeVertices( int edgeId ) const {
	throwIfUpdateNeeded();
	return (Edge*) data->edges.col( edgeId ).data();
}

void DynamicBoard::addFaces( int delta ) {
	if( delta ) {
		data->addNodes( delta );
		needsUpdate = true;
	}
}

void DynamicBoard::removeFace( int faceId ) {
	data->removeNode( faceId );
	needsUpdate = true;
}

Eigen::Vector3d project( const Eigen::Vector3d& v, const Eigen::Vector3d& n ) {
	return v - n.dot( v ) * n;
};

void DynamicBoard::updateDispersion() {
	data->updateParallel( data->nodes.cols(), [this]( int min, int max ) {
		const int n = faceCount();
		for( int i = min; i < max; i++ ) {
			auto pos = data->nodes.col( i );
			auto differences = data->nodes.colwise() - pos;
			auto squareNorms = differences.colwise().squaredNorm().unaryExpr([]( float x ) { return x ? x : 1; });
			auto directions = differences.array() / squareNorms.array().replicate<3, 1>().sqrt();
			auto rejections = directions.array() / squareNorms.array().replicate<3, 1>();
			auto rejection = rejections.rowwise().sum();
			data->nextStep.col( i ) -= 0.1 / sqrt( n ) * rejection.matrix();
			data->nextStep.col( i ) = 0.5 / sqrt( n ) * project(data->nextStep.col( i ), data->nodes.col( i ));
		}
	} );
	data->updateParallel( data->nodes.cols(), [this]( int min, int max ) {
		int cols = max - min;
		data->nodes.middleCols( min, cols ) += data->nextStep.middleCols( min, cols );
		data->nodes.middleCols( min, cols ).colwise().normalize();
	} );
	needsUpdate = true;
}

void DynamicBoard::updateGeometryIfNeeded() {
	if( needsUpdate ) {
		data->updateGeometrie();
		needsUpdate = false;
	}
}

void DynamicBoard::throwIfUpdateNeeded() const {
	if( needsUpdate ) {
		throw "data is outdated -> call updateGeometrie() first";
	}
}
