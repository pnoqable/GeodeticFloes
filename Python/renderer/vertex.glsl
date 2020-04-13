#version 120

uniform mat4 view;
uniform mat4 proj;

mat4 mvp = proj * view;

in float degree;
out float vDegree;

void main()
{
    gl_Position = mvp * gl_Vertex;
    gl_FrontColor = gl_Color;

    float fogSwitch = clamp( 4. * mvp[3].z - 3.5, 0., 1. );
    float fogValue = smoothstep( mvp[3].z - .3, mvp[3].z - .1, gl_Position.z );
    float insight = clamp( 4. * gl_Position.z, 0., 1. );
    gl_FrontColor.a *= insight * ( 1. - .75 * fogSwitch * fogValue );

    vDegree = degree;
}
