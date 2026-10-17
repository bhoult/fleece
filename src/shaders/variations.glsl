// flam3's variations (variations.c), numbered as flam3 numbers them. Each
// returns its contribution to the xform's output; `p` holds the parameters
// with flam3's precalcs already applied on the CPU (genome.rs, gpu_params).
// c0 / c1: the xform's affine, which waves, popcorn, rings and fan read.

const float PI = 3.14159265358979;
const float EPS = 1e-10;

float atan2s(float y, float x) { return (x == 0.0 && y == 0.0) ? 0.0 : atan(y, x); }
float fmodc(float a, float b) { return a - b * trunc(a / b); }
float sinhf(float x) { float e = exp(clamp(x, -80.0, 80.0)); return 0.5 * (e - 1.0 / e); }
float coshf(float x) { float e = exp(clamp(x, -80.0, 80.0)); return 0.5 * (e + 1.0 / e); }
float powp(float a, float b) { return a <= 0.0 ? 0.0 : exp2(b * log2(a)); }
float log10f(float x) { return log2(x) * 0.30102999566; }

// q: the parameters past the fourth (super_shape, lazysusan, wedge_julia, mobius)
vec2 variation(int id, float w, vec4 p, vec4 q, float tx, float ty, float sumsq, float sq, vec4 c0, vec4 c1) {
    switch (id) {
    case 0: return w * vec2(tx, ty);
    case 1: return w * vec2(sin(tx), sin(ty));
    case 2: { float r2 = w / (sumsq + EPS); return r2 * vec2(tx, ty); }
    case 3: { float s = sin(sumsq), c = cos(sumsq); return w * vec2(s * tx - c * ty, c * tx + s * ty); }
    case 4: { float r = w / (sq + EPS); return vec2((tx - ty) * (tx + ty) * r, 2.0 * tx * ty * r); }
    case 5: return w * vec2(atan2s(tx, ty) / PI, sq - 1.0);
    case 6: { float a = atan2s(tx, ty); return w * sq * vec2(sin(a + sq), cos(a - sq)); }
    case 7: { float a = sq * atan2s(tx, ty); float r = w * sq; return vec2(r * sin(a), -r * cos(a)); }
    case 8: { float a = atan2s(tx, ty) / PI; float r = PI * sq; return w * a * vec2(sin(r), cos(r)); }
    case 9: { float r = sq + EPS; float r1 = w / r; return r1 * vec2(ty / sq + sin(r), tx / sq - cos(r)); }
    case 10: { float r = sq + EPS; return w * vec2(tx / sq / r, ty / sq * r); }
    case 11: return w * vec2(tx / sq * cos(sq), ty / sq * sin(sq));
    case 12: {
        float a = atan2s(tx, ty);
        float n0 = sin(a + sq), n1 = cos(a - sq);
        float m0 = n0 * n0 * n0 * sq, m1 = n1 * n1 * n1 * sq;
        return w * vec2(m0 + m1, m0 - m1);
    }
    case 13: {
        float a = 0.5 * atan2s(tx, ty);
        if ((rnd() & 0x80000000u) != 0u) a += PI;
        float r = w * sqrt(sq);
        return r * vec2(cos(a), sin(a));
    }
    case 14: return w * vec2(tx < 0.0 ? tx * 2.0 : tx, ty < 0.0 ? ty * 0.5 : ty);
    case 15: {
        float dx2 = 1.0 / (c1.x * c1.x + EPS), dy2 = 1.0 / (c1.y * c1.y + EPS);
        return w * vec2(tx + c0.z * sin(ty * dx2), ty + c0.w * sin(tx * dy2));
    }
    case 16: { float r = 2.0 * w / (sq + 1.0); return r * vec2(ty, tx); }
    case 17: return w * vec2(tx + c1.x * sin(tan(3.0 * ty)), ty + c1.y * sin(tan(3.0 * tx)));
    case 18: { float dx = w * exp(tx - 1.0); float dy = PI * ty; return dx * vec2(cos(dy), sin(dy)); }
    case 19: { float sa = tx / sq, ca = ty / sq; float r = w * powp(sq, sa); return r * vec2(ca, sa); }
    case 20: { float a = tx * PI; return w * vec2(cos(a) * coshf(ty), -sin(a) * sinhf(ty)); }
    case 21: {
        float dx = c1.x * c1.x + EPS;
        float r = w * (fmodc(sq + dx, 2.0 * dx) - dx + sq * (1.0 - dx));
        return r * vec2(ty / sq, tx / sq);
    }
    case 22: {
        float dx = PI * (c1.x * c1.x + EPS), dy = c1.y, dx2 = 0.5 * dx;
        float a = atan2s(tx, ty);
        a += (fmodc(a + dy, dx) > dx2) ? -dx2 : dx2;
        return w * sq * vec2(cos(a), sin(a));
    }
    case 23: {
        float a = atan2s(tx, ty);
        float r = sq * (p.x + (p.y - p.x) * (0.5 + 0.5 * sin(p.z * a)));
        return w * r * vec2(tx / sq, ty / sq);
    }
    case 24: return w * vec2(sin(p.x * ty) - cos(p.y * tx), sin(p.z * tx) - cos(p.w * ty));
    case 25: {
        float dy = p.y, dx = PI * (p.x * p.x + EPS), dx2 = 0.5 * dx;
        float a = atan2s(tx, ty);
        float t = a + dy - dx * trunc((a + dy) / dx);
        a = t > dx2 ? a - dx2 : a + dx2;
        return w * sq * vec2(sin(a), cos(a));
    }
    case 26: {
        float dx = p.x * p.x + EPS;
        float r = sq - 2.0 * dx * trunc((sq + dx) / (2.0 * dx)) + sq * (1.0 - dx);
        return w * r * vec2(tx / sq, ty / sq);
    }
    case 27: { float r = 2.0 * w / (sq + 1.0); return r * vec2(tx, ty); }
    case 28: { float r = w / (0.25 * sumsq + 1.0); return r * vec2(tx, ty); }
    case 29: return w * vec2(sin(tx), ty);
    case 30: { float t = 1.0 / (p.x - ty * p.y); return w * vec2(p.x * tx * t, p.z * ty * t); }
    case 31: { float a = rnd01() * 2.0 * PI; float r = w * rnd01(); return r * vec2(tx * cos(a), ty * sin(a)); }
    case 32: {
        float t = trunc(p.y * rnd01());
        float a = (atan2s(ty, tx) + 2.0 * PI * t) / p.x;
        float r = w * powp(sumsq, p.z);
        return r * vec2(cos(a), sin(a));
    }
    case 33: {
        float t = trunc(p.y * rnd01());
        float yx = atan2s(ty, tx);
        float a = ((int(t) & 1) == 0 ? (2.0 * PI * t + yx) : (2.0 * PI * t - yx)) / p.x;
        float r = w * powp(sumsq, p.z);
        return r * vec2(cos(a), sin(a));
    }
    case 34: { float a = rnd01() * 2.0 * PI; float r = w * rnd01(); return r * vec2(cos(a), sin(a)); }
    case 35: {
        float a = rnd01() * 2.0 * PI;
        float r = w * (rnd01() + rnd01() + rnd01() + rnd01() - 2.0);
        return r * vec2(cos(a), sin(a));
    }
    case 36: {
        float g = w * (rnd01() + rnd01() + rnd01() + rnd01() - 2.0);
        float a = atan2s(ty, tx) + p.x * g;
        float rz = p.y * g - 1.0;
        return sq * vec2(cos(a), sin(a)) + rz * vec2(tx, ty);
    }
    case 37: {
        float sl = trunc(rnd01() * p.x + 0.5);
        float a = p.y + 2.0 * PI * (sl + rnd01() * p.z) / p.x;
        float r = w * rnd01();
        return r * vec2(cos(a), sin(a));
    }
    case 38: {
        float rf = powp(sumsq, p.y * 0.5);
        float th = atan2s(ty, tx);
        float b = 2.0 * PI / p.x;
        float ph = th - b * floor(th / b);
        if (ph > 0.5 * b) ph -= b;
        float amp = (p.w * (1.0 / (cos(ph) + EPS) - 1.0) + p.z) / (rf + EPS);
        return w * amp * vec2(tx, ty);
    }
    case 39: {
        float re = 1.0 + p.x * tx + p.y * (tx * tx - ty * ty);
        float im = p.x * ty + 2.0 * p.y * tx * ty;
        float r = w / (re * re + im * im);
        return r * vec2(tx * re + ty * im, ty * re - tx * im);
    }
    case 40: {
        float x = p.x == 0.0 ? tx : (2.0 * floor(tx / p.x) + 1.0) * p.x - tx;
        float y = p.y == 0.0 ? ty : (2.0 * floor(ty / p.y) + 1.0) * p.y - ty;
        return w * vec2(x, y);
    }
    case 41: { float a = rnd01() * w * PI; float s = sin(a), c = cos(a); return w * vec2(s, s * s / c); }
    case 42: return w * vec2(sin(tx) / cos(ty), tan(ty));
    case 43: return w * vec2(rnd01() - 0.5, rnd01() - 0.5);
    case 44: {
        float a = w * rnd01() * PI;
        float r = w / (sumsq + EPS);
        float t = w * tan(a) * r;
        return t * vec2(cos(tx), sin(ty));
    }
    case 45: { float r = rnd01() * w * sq; float s = sin(r), c = cos(r); return w * tx * vec2(c + s, c - s); }
    case 46: {
        float c = cos(w * sq);
        float ic = 1.0 / c;
        return w * vec2(tx, c < 0.0 ? ic + 1.0 : ic - 1.0);
    }
    case 47: {
        float r = rnd01() * w * sq;
        float s = sin(r), c = cos(r);
        float d = log10f(s * s) + c;
        if (isnan(d) || isinf(d) || abs(d) > 1e10) d = -30.0;
        return w * tx * vec2(d, d - s * PI);
    }
    case 48: { float s = tx * tx - ty * ty; float r = w * sqrt(1.0 / (s * s + EPS)); return r * vec2(tx, ty); }
    case 49: {
        float t = p.x * (tx + ty);
        float r = w * atan2s(tx, ty) / PI;
        return r * vec2(sin(t) + p.z, cos(t) + p.y);
    }
    case 50: {
        float th = p.x * atan2s(ty, tx) + 0.25 * PI;
        float t1 = powp(abs(cos(th)), p.y);
        float t2 = powp(abs(sin(th)), p.z);
        float r = w * ((q.x * rnd01() + (1.0 - q.x) * sq) - q.y) * pow(max(t1 + t2, 1e-30), p.w) / sq;
        return r * vec2(tx, ty);
    }
    case 51: { float r = w * (rnd01() - p.y) * cos(p.x * atan2s(ty, tx)) / sq; return r * vec2(tx, ty); }
    case 52: {
        float ct = tx / sq;
        float r = w * (rnd01() - p.y) * p.x / (1.0 + p.x * ct) / sq;
        return r * vec2(tx, ty);
    }
    case 53: { float s = sin(sq), c = cos(sq); return w * vec2(p.x * s * s * rnd01(), p.y * c * rnd01()); }
    case 54: return w * vec2(tx < 0.0 ? tx * p.x : tx, ty < 0.0 ? ty * p.y : ty);
    case 55: {
        float t = sumsq + 1.0, x2 = 2.0 * tx;
        float y = 0.5 * atan2s(2.0 * ty, sumsq - 1.0) - 0.5 * PI * p.x;
        if (y > 0.5 * PI) y = -0.5 * PI + fmodc(y + 0.5 * PI, PI);
        else if (y < -0.5 * PI) y = 0.5 * PI - fmodc(0.5 * PI - y, PI);
        return w * vec2(0.25 * (2.0 / PI) * log((t + x2) / (t - x2)), (2.0 / PI) * y);
    }
    case 56: {
        float rx = roundEven(tx), ry = roundEven(ty);
        float ox = tx - rx, oy = ty - ry;
        if (rnd01() >= 0.75) return w * vec2(ox * 0.5 + rx, oy * 0.5 + ry);
        if (abs(ox) >= abs(oy)) {
            if (ox >= 0.0) return w * vec2(ox * 0.5 + rx + 0.25, oy * 0.5 + ry + 0.25 * oy / ox);
            return w * vec2(ox * 0.5 + rx - 0.25, oy * 0.5 + ry - 0.25 * oy / ox);
        }
        if (oy >= 0.0) return w * vec2(ox * 0.5 + rx + ox / oy * 0.25, oy * 0.5 + ry + 0.25);
        return w * vec2(ox * 0.5 + rx - ox / oy * 0.25, oy * 0.5 + ry - 0.25);
    }
    case 57: {
        float y2 = ty * 2.0;
        float r = w * 1.30294003174112 * sqrt(abs(ty * tx) / (EPS + tx * tx + y2 * y2));
        return r * vec2(tx, y2);
    }
    case 58: {
        float inv = 1.0 / p.x;
        float x = floor(tx * inv), y = floor(ty * inv);
        float dx = tx - x * p.x, dy = ty - y * p.x;
        x = x >= 0.0 ? 2.0 * x : -(2.0 * x + 1.0);
        y = y >= 0.0 ? 2.0 * y : -(2.0 * y + 1.0);
        return w * vec2(dx + x * p.x, -(dy + y * p.x));
    }
    case 59: {
        float a = atan2s(ty, tx);
        float lnr = 0.5 * log(sumsq);
        float va = 2.0 * PI / p.z, vc = p.x / p.z, vd = p.y / p.z;
        float ang = vc * a + vd * lnr + va * floor(p.z * rnd01());
        float m = w * exp(vc * lnr - vd * a);
        return m * vec2(cos(ang), sin(ang));
    }
    case 60: {
        float xl = max(p.z * p.z, 1e-20), yl = max(p.w * p.w, 1e-20);
        return w * vec2(tx + p.x * exp(-ty * ty / xl), ty + p.y * exp(-tx * tx / yl));
    }
    case 61: {
        float tmp = sumsq + 1.0, tmp2 = 2.0 * tx;
        float xmax = (sqrt(tmp + tmp2) + sqrt(tmp - tmp2)) * 0.5;
        float a1 = log(xmax + sqrt(max(xmax - 1.0, 0.0)));
        float a2 = -acos(clamp(tx / xmax, -1.0, 1.0));
        float ww = w / 11.57034632;
        float snv = sin(a1), csv = cos(a1);
        if (ty > 0.0) snv = -snv;
        return ww * vec2(coshf(a2) * csv, sinhf(a2) * snv);
    }
    case 62: {
        float tmp = sumsq + 1.0, x2 = 2.0 * tx;
        float xmax = 0.5 * (sqrt(tmp + x2) + sqrt(tmp - x2));
        float a = tx / xmax;
        float b = sqrt(max(1.0 - a * a, 0.0));
        float ssx = sqrt(max(xmax - 1.0, 0.0));
        float ww = w / (0.5 * PI);
        float l = log(xmax + ssx);
        return ww * vec2(atan2s(a, b), ty > 0.0 ? l : -l);
    }
    case 63: {
        float a = atan2s(ty, tx);
        float lnr = 0.5 * log(sumsq);
        float m = w * exp(p.x * lnr - p.y * a);
        float n = p.x * a + p.y * lnr;
        return m * vec2(cos(n), sin(n));
    }
    case 64: {
        float ex = exp(tx) * 0.5, enx = 0.25 / ex;
        float t = w / (ex + enx - cos(ty));
        return t * vec2(ex - enx, sin(ty));
    }
    case 65: {
        float x = tx - p.x, y = ty + p.y;
        float r = sqrt(x * x + y * y);
        if (r < w) {
            float a = atan2s(y, x) + p.z + q.x * (w - r);
            r = w * r;
            return vec2(r * cos(a) + p.x, r * sin(a) - p.y);
        }
        r = w * (1.0 + p.w / r);
        return vec2(r * x + p.x, r * y - p.y);
    }
    case 66: {
        float w2 = w * w;
        if (sumsq < w2) { float r = w * sqrt(w2 / sumsq - 1.0); return r * vec2(tx, ty); }
        return w * vec2(tx, ty);
    }
    case 68: {
        float x = tx, y = ty;
        if (tx > p.x) x = -p.x + fmodc(tx + p.x, 2.0 * p.x);
        else if (tx < -p.x) x = p.x - fmodc(p.x - tx, 2.0 * p.x);
        if (ty > p.y) y = -p.y + fmodc(ty + p.y, 2.0 * p.y);
        else if (ty < -p.y) y = p.y - fmodc(p.y - ty, 2.0 * p.y);
        return w * vec2(x, y);
    }
    case 69: {
        float tpf = 2.0 * PI * p.y;
        float t = p.z * cos(tpf * tx) * (p.w == 0.0 ? 1.0 : exp(-abs(tx) * p.w)) + p.x;
        return w * vec2(tx, abs(ty) <= t ? -ty : ty);
    }
    case 70: { float v = w / PI; return vec2(v * atan2s(tx, ty), v * 0.5 * log(sumsq)); }
    case 71: return w * vec2(tx + p.x * sin(tan(ty * p.z)), ty + p.y * sin(tan(tx * p.z)));
    case 72: { float r = 1.0 / (sq * (sumsq + 1.0 / (w + EPS))); return r * vec2(tx, ty); }
    case 73: {
        float sx2 = p.x * p.x, sy2 = p.z * p.z;
        float x = tx > 0.0 ? sqrt(tx * tx + sx2) - tx * p.y : -(sqrt(tx * tx + sx2) + tx * p.y);
        float y = ty > 0.0 ? sqrt(ty * ty + sy2) - ty * p.w : -(sqrt(ty * ty + sy2) + ty * p.w);
        return w * vec2(x, y);
    }
    case 74: return w * vec2(cos(ty * p.y * PI) >= 0.0 ? tx : -tx, cos(tx * p.x * PI) >= 0.0 ? ty : -ty);
    case 75: return w * vec2(tx >= 0.0 ? tx + p.x : tx - p.x, ty >= 0.0 ? ty + p.y : ty - p.y);
    case 76: {
        float rx = floor(tx + 0.5), ox = tx - rx;
        return w * vec2(ox * (1.0 - p.x) + rx, ty + ox * ox * p.y);
    }
    case 77: {
        float a = atan2s(ty, tx) + p.w * sq;
        float c = floor((p.z * a + PI) / PI * 0.5);
        float cf = 1.0 - p.x * p.z / PI * 0.5;
        a = a * cf + c * p.x;
        float r = w * (sq + p.y);
        return r * vec2(cos(a), sin(a));
    }
    case 78: {
        float r = w * powp(sumsq, q.y);
        float t = trunc(q.x * rnd01());
        float a = (atan2s(ty, tx) + 2.0 * PI * t) / p.z;
        float c = floor((p.y * a + PI) / PI * 0.5);
        a = a * p.w + c * p.x;
        return r * vec2(cos(a), sin(a));
    }
    case 79: {
        float r = 1.0 / (sq + EPS);
        float a = atan2s(ty, tx) + p.w * r;
        float c = floor((p.y * a + PI) / PI * 0.5);
        float cf = 1.0 - p.x * p.y / PI * 0.5;
        a = a * cf + c * p.x;
        r = w * (r + p.z);
        return r * vec2(cos(a), sin(a));
    }
    case 80: {
        float a = atan2s(ty, tx) + (sq < w ? p.x : p.y) / (w - sq);
        return w * sq * vec2(cos(a), sin(a));
    }
    case 81: return w * vec2(tx + p.x * sin(ty * p.z), ty + p.y * sin(tx * p.w));
    case 82: { float e = exp(tx); return w * e * vec2(cos(ty), sin(ty)); }
    case 83: return w * vec2(0.5 * log(sumsq), atan2s(ty, tx));
    case 84: return w * vec2(sin(tx) * coshf(ty), cos(tx) * sinhf(ty));
    case 85: return w * vec2(cos(tx) * coshf(ty), -sin(tx) * sinhf(ty));
    case 86: { float d = 1.0 / (cos(2.0 * tx) + coshf(2.0 * ty)); return w * d * vec2(sin(2.0 * tx), sinhf(2.0 * ty)); }
    case 87: {
        float d = 2.0 / (cos(2.0 * tx) + coshf(2.0 * ty));
        return w * d * vec2(cos(tx) * coshf(ty), sin(tx) * sinhf(ty));
    }
    case 88: {
        float d = 2.0 / (coshf(2.0 * ty) - cos(2.0 * tx));
        return w * d * vec2(sin(tx) * coshf(ty), -cos(tx) * sinhf(ty));
    }
    case 89: { float d = 1.0 / (coshf(2.0 * ty) - cos(2.0 * tx)); return w * d * vec2(sin(2.0 * tx), -sinhf(2.0 * ty)); }
    case 90: return w * vec2(sinhf(tx) * cos(ty), coshf(tx) * sin(ty));
    case 91: return w * vec2(coshf(tx) * cos(ty), sinhf(tx) * sin(ty));
    case 92: { float d = 1.0 / (cos(2.0 * ty) + coshf(2.0 * tx)); return w * d * vec2(sinhf(2.0 * tx), sin(2.0 * ty)); }
    case 93: {
        float d = 2.0 / (cos(2.0 * ty) + coshf(2.0 * tx));
        return w * d * vec2(cos(ty) * coshf(tx), -sin(ty) * sinhf(tx));
    }
    case 94: {
        float d = 2.0 / (coshf(2.0 * tx) - cos(2.0 * ty));
        return w * d * vec2(sinhf(tx) * cos(ty), -coshf(tx) * sin(ty));
    }
    case 95: { float d = 1.0 / (coshf(2.0 * tx) - cos(2.0 * ty)); return w * d * vec2(sinhf(2.0 * tx), sin(2.0 * ty)); }
    case 96: {
        float s = sin(p.x * tx), t = sin(p.x * ty);
        float dy = ty + p.y * (p.z * s * 0.5 + abs(ty) * s);
        float dx = tx + p.y * (p.z * t * 0.5 + abs(tx) * t);
        return w * vec2(tx + p.w * (dx - tx), dy);
    }
    case 97: {
        float xpw = tx + w, xmw = tx - w;
        float r = w * (2.0 + p.x) * sqrt(sqrt(ty * ty + xpw * xpw) / sqrt(ty * ty + xmw * xmw));
        float a = (atan2s(ty, xmw) - atan2s(ty, xpw)) * 0.5;
        return r * vec2(cos(a), sin(a));
    }
    case 98: {
        float reu = p.x * tx - p.y * ty + p.z, imu = p.x * ty + p.y * tx + p.w;
        float rev = q.x * tx - q.y * ty + q.z, imv = q.x * ty + q.y * tx + q.w;
        float rad = w / (rev * rev + imv * imv);
        return rad * vec2(reu * rev + imu * imv, imu * rev - reu * imv);
    }
    }
    return vec2(0.0);
}
