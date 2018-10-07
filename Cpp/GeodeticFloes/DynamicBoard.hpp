#pragma once

#include <vector>

struct DynamicBoardData;

class DynamicBoard
{
public:
	typedef std::vector<int> Neighbors;
	typedef std::vector<int> Vertices;
	
	struct Edge {
		int vertices[2];

		int from() const {
			return vertices[0];
		}

		int to() const {
			return vertices[1];
		}
	};

private:
	DynamicBoardData* data;
	bool needsUpdate;

public:

	explicit DynamicBoard( int faceCount );
	DynamicBoard::~DynamicBoard();

	// native data getters:
	const DynamicBoardData& internalData() const;

	const int faceCount() const;
	const Neighbors& faceNeighbors( int faceId ) const;
	
	const double* faceCenters() const;
	const double* faceCenter( int faceId ) const;
	
	const Vertices& faceVertices( int faceId ) const;

	const int vertexCount() const;
	const double* vertexPositions() const;
	const double* vertexPosition( int vertexId ) const;

	const int borderCount() const;
	const Edge* borderVertices() const;
	const Edge* borderVertices( int borderId ) const;

	// add or remove arbitrary amount of nodes
	// delta can be < 0
	void addFaces( int delta );

	// delete specific node, values < 0 remove from end
	void removeFace( int faceId = -1 );

	// one step towards better spatial distribution of nodes
	void updateDispersion();

	// calculate vertices, borders and faces according to current node
	void updateGeometryIfNeeded();

	// write current dispersion and geometrie values on next update
	void writeStatisticsOnce();

private:

	void throwIfUpdateNeeded() const;

};
