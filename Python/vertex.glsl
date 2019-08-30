#version 120

uniform mat4 view;
uniform mat4 proj;

mat4 mvp = proj * view;

void main()
{
    gl_Position = mvp * gl_Vertex;
    gl_FrontColor = gl_Color;
}
