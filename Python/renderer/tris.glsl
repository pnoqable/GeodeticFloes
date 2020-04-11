#version 150

uniform mat4 view;
uniform mat4 proj;

uniform float sides;
uniform int minOut;

mat4 mvp = proj * view;
mat4 invMvp = inverse( mvp );

layout( triangles ) in;
layout( triangle_strip, max_vertices = 48 ) out;

void main()
{
    vec4 lastPos = gl_in[1].gl_Position;

    float length = distance( invMvp * gl_in[1].gl_Position, invMvp * gl_in[2].gl_Position );

    int lines = max( minOut, int( ceil( sides * length ) ) );

    for( int i = 1; i <= lines; i++ ) {
        gl_Position = gl_in[0].gl_Position;
        gl_FrontColor = gl_in[0].gl_FrontColor;
        EmitVertex();

        gl_Position = lastPos;
        gl_FrontColor = gl_in[0].gl_FrontColor;
        EmitVertex();

        float a = float( i ) / float( lines );

        vec4 middle = mix( gl_in[1].gl_Position, gl_in[2].gl_Position, a );
        vec4 middleWorld = invMvp * middle;

        middleWorld.xyz = normalize( middleWorld.xyz );

        lastPos = mvp * middleWorld;  // middle;

        gl_Position = lastPos;
        gl_FrontColor = gl_in[0].gl_FrontColor;
        EmitVertex();
        
        EndPrimitive();
    }
}
