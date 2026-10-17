// A rectangle (u_rect: x, y, w, h in clip space) with the texture's first row at the top.
uniform vec4 u_rect;
out vec2 uv;
void main() {
    vec2 c[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));
    vec2 q = c[gl_VertexID];
    uv = vec2(q.x, 1.0 - q.y);
    gl_Position = vec4(u_rect.xy + q * u_rect.zw, 0.0, 1.0);
}
