#version 150 core

uniform int verticesCount;
uniform Vertices {
    vec4 vertices[maxVerticesCount];
};

in vec4 vertex;
out vec4 rejection;

void main() {
    rejection = vec4( 0., 0., 0., 0. );
    for( int i = 0; i < verticesCount; i++ ) {
        vec4 diff = vertex - vertices[i];
        float dist = dot( diff, diff );
        if( dist != 0. )
            rejection += diff / sqrt( dist ) / dist;
    }
}
