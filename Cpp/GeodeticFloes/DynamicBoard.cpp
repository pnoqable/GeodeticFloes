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
	int faces = data->nodes.cols();
	return faces > 3 ? faces : 0;
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

const int DynamicBoard::borderCount() const {
	throwIfUpdateNeeded();
	return data->borders.cols();
}

const DynamicBoard::Edge* DynamicBoard::borderVertices() const {
	throwIfUpdateNeeded();
	return (Edge*) data->borders.data();
}

const DynamicBoard::Edge* DynamicBoard::borderVertices( int borderId ) const {
	throwIfUpdateNeeded();
	return (Edge*) data->borders.col( borderId ).data();
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

void DynamicBoard::updateDispersion() {
	data->updateDispersion();
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
