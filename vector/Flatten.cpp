
#include "Flatten.h"

// Compute an approximation to int (1 + 4x^2) ^ -0.25 dx
// This isn't especially good but will do.
inline float approxIntegral(float x) {
   const float d = 0.67f; 
   return x / (1.0f - d + std::sqrt(std::sqrt(std::pow(d, 4.0f) + 0.25f * x * x)));
}

// Approximate the inverse of the function above.
// This is better.
inline float approxInvIntegral(float x) {
    const float b = 0.39f;
    return x * (1.0f - b + std::sqrt(b * b + 0.25f * x * x));
}

// Evaluating the cubic curve c at parameter t. Returns the (x, y) coordinate at that point.
inline VPoint evalCubicBez(const VPoint& p0, const VPoint& c0, const VPoint& c1, const VPoint& p1, float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float mt3 = mt2 * mt;
  
    float x = p0.x * mt3 + c0.x * 3.0f * mt2 * t + c1.x * 3.0f * mt * t2 + p1.x * t3;
    float y = p0.y * mt3 + c0.y * 3.0f * mt2 * t + c1.y * 3.0f * mt * t2 + p1.y * t3;
    return { x, y };
}

// Evaluating a quadratic curve q at the point t. Returns the (x, y) coordinate at that point.
inline VPoint evalQuadBez(const VPoint& p0, const VPoint& c0,const VPoint& p1, float t) {
    float mt = 1.0f - t;
    float mt2 = mt * mt;
    float t2 = t * t;
    float tmt = t * mt;

    float x = p0.x * mt2 + 2.0f * c0.x * tmt + p1.x * t2;
    float y = p0.y * mt2 + 2.0f * c0.y * tmt + p1.y * t2;
    return { x, y };
}

struct Basic {
    float x0;
    float x2;
    float scale;
    float cross;
};

inline Basic QuadBezMapToBasic(const VPoint& p0, const VPoint& c0, const VPoint& p1) {
    float ddx = 2.0f * c0.x - p0.x - p1.x;
    float ddy = 2.0f * c0.y - p0.y - p1.y;
    float u0 = (c0.x - p0.x) * ddx + (c0.y - p0.y) * ddy;
    float u2 = (p1.x - c0.x) * ddx + (p1.y - c0.y) * ddy;
    float cross = (p1.x - p0.x) * ddy - (p1.y - p0.y) * ddx;

    float inv_cross = 1.0f / cross;
    float x0 = u0 * inv_cross;
    float x2 = u2 * inv_cross;
    float scale = std::abs(cross) / (std::hypot(ddx, ddy) * std::abs(x2 - x0));
    
    return { x0, x2, scale, cross };
}

void QuadBezFlatten(const VPoint& p0, const VPoint& c0, const VPoint& p1, const float tolerance, std::vector<VPathVerb>& verbs, std::vector<VPoint>& points) {
    Basic params = QuadBezMapToBasic(p0, c0, p1);
    float a0 = approxIntegral(params.x0);
    float a2 = approxIntegral(params.x2);
    float count = 0.5f * std::abs(a2 - a0) * std::sqrt(params.scale / tolerance);
    uint32_t n = static_cast<uint32_t>(std::ceil(count));
    // Handle case where all the points are collinear and the end point is between the start point and the control point
    // if (!std::isinf(count) || n == 0u || n == 1u) {
    //   // Find t values where the derivative is 0
    //   float divx = p0.x + p1.x - 2 * c0.x;
    //   float divy = p0.y + p1.y - 2 * c0.y;
    //   float tx = (p0.x - c0.x) / divx;
    //   float ty = (p0.y - c0.y) / divy;
    //   let ts = [0.0];
    //   if (Number.isFinite(tx) && tx > 0 && tx < 1) {
    //     ts.push(tx);
    //   }
    //   if (Number.isFinite(ty) && ty > 0 && ty < 1) {
    //     if (ty > ts[ts.length - 1]) {
    //       ts.push(ty);
    //     }
    //   }
    //   ts.push(1.0);
    //   quadBezState.numberOfSegments = ts.length;
    //   return ts;
    // }

    float u0 = approxInvIntegral(a0);
    float u2 = approxInvIntegral(a2);

    float udiv = 1.0f / (u2 - u0);
    float ainc = (a2 - a0) / static_cast<float>(n);
    for (int i = 1; i < n; i++) {
      float u = approxInvIntegral(a0 + ainc);
      float t = (u - u0) * udiv;
      verbs.push_back(VPathVerb::kLine);
      points.push_back(evalQuadBez(p0, c0, p1, t));
      ainc += ainc;
    }

    verbs.push_back(VPathVerb::kLine);
    points.push_back(p1);
}

 // Returns the number of quadratics needed to approximate the cubic c, given the specified tolerance.
inline float CubicBezNumQuadratics(const VPoint& p0, const VPoint& c0, const VPoint& c1, const VPoint& p1, const float tolerance) {
    float x = p0.x - 3.0f * c0.x + 3.0f * c1.x - p1.x;
    float y = p0.y - 3.0f * c0.y + 3.0f * c1.y - p1.y;
    float err = x * x + y * y;

    float result = err / (432.0f * tolerance * tolerance);
    float cubeRoot = std::cbrt(result);
    float sixthRoot = std::sqrt(cubeRoot);
    return std::max(std::ceil(sixthRoot), 1.0f);
}

void cubicBezToQuadratic(const VPoint& p0, const VPoint& c0, const VPoint& c1, const VPoint& p1) {
    float c1x = (c0.x * 3.0f - p0.x) * 0.5f;
    float c1y = (c0.y * 3.0f - p0.y) * 0.5f;
    float c2x = (c1.x * 3.0f - p1.x) * 0.5f;
    float c2y = (c1.y * 3.0f - p1.y) * 0.5f;
    float cx = (c1x + c2x) * 0.5f;
    float cy = (c1y + c2y) * 0.5f;
  
    // return {
    //   p0: c.p0,
    //   p1: { x: cx, y: cy },
    //   p2: c.p3,
    // };
}
 
// Effectively gives the same result as blossom(c).middle, but is easier to implement.
// Stole it from lyon2d_geom: https://github.com/nical/lyon/blob/2407b7f5e326b2a8f66bfae81fe02d850d8b0acc/crates/geom/src/cubic_bezier.rs#L153
void CubicBezSplitRange(const VPoint& p0, const VPoint& c0, const VPoint& c1, const VPoint& p1, float t0, float t1, float tolerance, std::vector<VPathVerb>& verbs, std::vector<VPoint>& points) {
    VPoint from = evalCubicBez(p0, c0, c1, p1, t0);
    VPoint to   = evalCubicBez(p0, c0, c1, p1, t1);
    
    float dxFrom = c0.x - p0.x;
    float dyFrom = c0.y - p0.y;
    float dxCtrl = c1.x - c0.x;
    float dyCtrl = c1.y - c0.y;
    float dxTo   = p1.x - c1.x;
    float dyTo   = p1.y - c1.y;
    
    VPoint cp0 = VPoint::Make(dxFrom, dyFrom);
    VPoint cp1 = VPoint::Make(dxCtrl, dyCtrl);
    VPoint cp2 = VPoint::Make(dxTo, dyTo);
    
    VPoint ev0 = evalQuadBez(cp0, cp1, cp2, t0);
    VPoint ev1 = evalQuadBez(cp0, cp1, cp2, t1);

    float dt = t1 - t0;
    float xCtrl1 = from.x + ev0.x * dt;
    float yCtrl1 = from.y + ev0.y * dt;
    float xCtrl2 = to.x - ev1.x * dt;
    float yCtrl2 = to.y - ev1.y * dt;

    // To quadratic
    float c1x = xCtrl1 * 3.0f - from.x;
    float c1y = yCtrl1 * 3.0f - from.y;
    float c2x = xCtrl2 * 3.0f - to.x;
    float c2y = yCtrl2 * 3.0f - to.y;
    float cx = (c1x + c2x) * 0.25f;
    float cy = (c1y + c2y) * 0.25f;

    QuadBezFlatten(from, {cx, cy}, to, tolerance, verbs, points);
}


// Converting the cubic c to a sequence of quadratics, with the specified tolerance.
// Returns an array that contains these quadratics.
void CubicBezToQuadratics(const VPoint& p0, const VPoint& c0, const VPoint& c1, const VPoint& p1, const float tolerance, std::vector<VPathVerb>& verbs, std::vector<VPoint>& points) {
    float numQuads = CubicBezNumQuadratics(p0, c0, c1, p1, 0.05f);
    float step = 1.0f / numQuads;
    uint32_t n = static_cast<uint32_t>(std::trunc(numQuads));
    float t0 = 0.0f;

    for (int i = 0; i < n - 1u; i++) {
        float t1 = t0 + step;
        CubicBezSplitRange(p0, c0, c1, p1, t0, t1, tolerance, verbs, points);
        t0 = t1;
    }
  
    CubicBezSplitRange(p0, c0, c1, p1, t0, 1.0f, tolerance, verbs, points);  
}


void FlattenCommands2(
    const std::vector<VPathVerb>& verbs, 
    const std::vector<VPoint>& points, 
    std::vector<VPathVerb>& outVerbs,
    std::vector<VPoint>& outPoints,
    const float tolerance) {

    VPoint _last;
    VPoint first;

    size_t i = 0;
    const float tolerance4  = tolerance * 4.0f;
    const float sqrt_of_8 = 2.82842712475f;
    const float sqrt_of_8_tol = 2.82842712475f * tolerance;
    
    for (auto& verb: verbs) {
        switch (verb) {
            case VPathVerb::kClose:
                outPoints.push_back(first);
                outVerbs.push_back(VPathVerb::kLine);
                break;

            case VPathVerb::kMove:
                assert(i < points.size());
                first = points[i];
                [[fallthrough]];
            case VPathVerb::kLine:  {
                const VPoint& line = points[i];
    
                outPoints.push_back(line);
                outVerbs.push_back(verb);
                _last = line;
                i += 1;
                break;
            }
            case VPathVerb::kQuad: {
                const VPoint& control = points[i];
                const VPoint& point = points[i + 1];

                float l = (_last - 2.0f * control + point).Length();
                float dt = std::sqrt(tolerance4 / l);
                float t = std::min(dt, 1.0f);
                while (t < 1.0f) {
                    VPoint p01 = _last.Lerp(control, t);
                    VPoint p12 = control.Lerp(point, t);
                    VPoint line = p01.Lerp(p12, t);
                    outPoints.push_back(line);
                    outVerbs.push_back(VPathVerb::kLine);
                    t += dt;
                }
                outPoints.push_back(point);
                outVerbs.push_back(VPathVerb::kLine);

                // QuadBezFlatten(_last, control, point, tolerance, outVerbs, outPoints);
                _last = point;
                i += 2;
                break;
            }
            case VPathVerb::kCubic: {
                const VPoint& control1 = points[i];
                const VPoint& control2 = points[i + 1];
                const VPoint& point = points[i + 2];

                // VPoint a = -1.0f * _last + 3.0f * control1 - 3.0f * control2 + point;
                // VPoint b = 3.0f * (_last - 2.0f * control1 + control2);
                // float conc = std::max(b.Length(), (a + b).Length());
                // float dt = std::sqrt(sqrt_of_8_tol / conc);
                // float t = std::min(dt, 1.0f);
                // while (t < 1.0f) {
                //     // t = std::min(t + dt, 1.0f);
                //     VPoint p01 = _last.Lerp(control1, t);
                //     VPoint p12 = control1.Lerp(control2, t);
                //     VPoint p23 = control2.Lerp(point, t);
                //     VPoint p012 = p01.Lerp(p12, t);
                //     VPoint p123 = p12.Lerp(p23, t);
                //     VPoint line = p012.Lerp(p123, t);
                //     outPoints.push_back(line);
                //     outVerbs.push_back(VPathVerb::kLine);
                //     t += dt;
                // }
                // outPoints.push_back(point);
                // outVerbs.push_back(VPathVerb::kLine);

                CubicBezToQuadratics(_last, control1, control2, point, tolerance, outVerbs, outPoints);
                _last = point;
                i += 3;
                break;
            }

            default: {
                std::cerr << "Verb was: " << static_cast<uint32_t>(verbs[i]) << " At position: " << i << " buffer size: " << points.size()
                          << std::endl;
                exit(1);
            }
        }
    }
}
