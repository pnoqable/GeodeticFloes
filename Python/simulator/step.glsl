#version 150 core

uniform int rejectionsCount;
uniform float repulsion;
uniform float friction;

in vec4 vertex;
in vec4 translation;
in vec4 rejections[maxRejectionsCount];
out vec4 newVertex;
out vec4 newTranslation;

void main() {
    vec4 rejection = vec4( 0. );
    for( int i = 0; i < summandsCount; i++ ) {
        rejection += rejections[i];
    }

    newTranslation = translation + repulsion * rejection;
    newTranslation -= dot( vertex, newTranslation ) * vertex;
    newTranslation *= exp( -friction, dot( newTranslation, newTranslation ) );

    newVertex = normalize( vertex + newTranslation );
}
