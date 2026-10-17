in vec2 uv;
out vec4 o;
layout(binding = 0) uniform sampler2D tex;
uniform int u_mode;
void main() {
    o = texture(tex, uv);
    if (u_mode == 0) o.a = 1.0;
}
