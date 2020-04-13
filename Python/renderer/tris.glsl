#version 150

uniform mat4 view;
uniform mat4 proj;

uniform float sides;
uniform int minOut;
uniform float alpha;

mat4 mvp = proj * view;
mat4 invMvp = inverse( mvp );

layout( triangles ) in;
layout( triangle_strip, max_vertices = 48 ) out;

in float vDegree[]; // only vDegree[0] is used

void main()
{
    vec4 lastPos = gl_in[1].gl_Position;
    vec4 lastCol = gl_in[1].gl_FrontColor;
    vec4 triColor = vec4( 1., 1., 1., 1 );

    if( vDegree[0] != 0 ) {
        float brightness = 1.3 - 0.1 * vDegree[0];
        triColor.rgb = vec3( brightness );
        triColor.a = mix( brightness, 1, alpha );
    }

    float length = distance( invMvp * gl_in[1].gl_Position, invMvp * gl_in[2].gl_Position );

    int lines = max( minOut, int( ceil( sides * length ) ) );

    for( int i = 1; i <= lines; i++ ) {
        gl_Position = gl_in[0].gl_Position;
        gl_FrontColor = gl_in[0].gl_FrontColor * triColor;
        EmitVertex();

        gl_Position = lastPos;
        gl_FrontColor = lastCol * triColor;
        EmitVertex();

        float a = float( i ) / float( lines );

        vec4 middle = mix( gl_in[1].gl_Position, gl_in[2].gl_Position, a );
        vec4 middleWorld = invMvp * middle;

        middleWorld.xyz = normalize( middleWorld.xyz );

        lastPos = mvp * middleWorld;  // middle;
        lastCol = mix( gl_in[1].gl_FrontColor, gl_in[2].gl_FrontColor, a ); 

        gl_Position = lastPos;
        gl_FrontColor = lastCol * triColor;
        EmitVertex();
        
        EndPrimitive();
    }
}
