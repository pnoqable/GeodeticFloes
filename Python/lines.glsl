#version 150

uniform mat4 view;
uniform mat4 proj;

uniform float sides;
uniform int minOut;

mat4 mvp = proj * view;
mat4 invMvp = inverse( mvp );

layout( lines ) in;
layout( line_strip, max_vertices = 17 ) out;

void main()
{
    float length = distance( invMvp * gl_in[0].gl_Position, invMvp * gl_in[1].gl_Position );

    int lines = max( minOut, int( ceil( sides * length ) ) );

    for( int i = 0; i <= lines; i++ ) {
        float a = float( i ) / float( lines );

        vec4 middle = mix( gl_in[0].gl_Position, gl_in[1].gl_Position, a );
        vec4 middleWorld = invMvp * middle;

        middleWorld.xyz = normalize( middleWorld.xyz );

        gl_Position = mvp * middleWorld;
        gl_FrontColor = mix( gl_in[0].gl_FrontColor, gl_in[1].gl_FrontColor, a );
            
        EmitVertex();
    }
    
    EndPrimitive();
}
