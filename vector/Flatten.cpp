
#include "Flatten.h"

const float sqrt_of_8_tol = 2.82842712475f * 0.175f;

inline float fastInverseSqrt(float x) {
    union {
        float f;
        uint32_t i;
    } conv = {x};
    
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= 1.5f - 0.5f * x * conv.f * conv.f;
    return conv.f;
}

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
  
    float x = p0.x * mt3 + c0.x * mt2 * t + c1.x * mt * t2 + p1.x * t3;
    float y = p0.y * mt3 + c0.y * mt2 * t + c1.y * mt * t2 + p1.y * t3;
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

// Adaptive forward differencing for bezier tesselation.
// See Lien, Sheue-Ling, Michael Shantz, and Vaughan Pratt. "Adaptive forward differencing for rendering curves and surfaces." ACM SIGGRAPH Computer Graphics. Vol. 21. No. 4. ACM, 1987.
void nvg__tesselateBezierAFD(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, const float tolerance, std::vector<VPathVerb>& verbs, std::vector<VPoint>& points) {

	// Power basis.
	float ax = -x1 + 3*x2 - 3*x3 + x4;
	float ay = -y1 + 3*y2 - 3*y3 + y4;
	float bx = 3*x1 - 6*x2 + 3*x3;
	float by = 3*y1 - 6*y2 + 3*y3;
	float cx = -3*x1 + 3*x2;
	float cy = -3*y1 + 3*y2;

	// Transform to forward difference basis (stepsize 1)
	float px = x1;
	float py = y1;
	float dx = ax + bx + cx;
	float dy = ay + by + cy;
	float ddx = 6*ax + 2*bx;
	float ddy = 6*ay + 2*by;
	float dddx = 6*ax;
	float dddy = 6*ay;

	//printf("dx: %f, dy: %f\n", dx, dy);
	//printf("ddx: %f, ddy: %f\n", ddx, ddy);
	//printf("dddx: %f, dddy: %f\n", dddx, dddy);

	#define AFD_ONE (1<<10)

	int t = 0;
	int dt = AFD_ONE;

	float tol = tolerance * 4.0;

	while(t < AFD_ONE) {

		// Flatness measure.
		float d = ddx*ddx + ddy*ddy + dddx*dddx + dddy*dddy;

		// printf("d: %f, th: %f\n", d, th);

		// Go to higher resolution if we're moving a lot
		// or overshooting the end.
		while( (d > tol && dt > 1) || (t+dt > AFD_ONE) ) {

			// printf("up\n");

			// Apply L to the curve. Increase curve resolution.
			dx = .5 * dx - (1.0/8.0)*ddx + (1.0/16.0)*dddx;
			dy = .5 * dy - (1.0/8.0)*ddy + (1.0/16.0)*dddy;
			ddx = (1.0/4.0) * ddx - (1.0/8.0) * dddx;
			ddy = (1.0/4.0) * ddy - (1.0/8.0) * dddy;
			dddx = (1.0/8.0) * dddx;
			dddy = (1.0/8.0) * dddy;

			// Half the stepsize.
			dt >>= 1;

			// Recompute d
			d = ddx*ddx + ddy*ddy + dddx*dddx + dddy*dddy;

		}

		// Go to lower resolution if we're really flat
		// and we aren't going to overshoot the end.
		// XXX: tol/32 is just a guess for when we are too flat.
		while ( (d > 0 && d < tol/32.0f && dt < AFD_ONE) && (t+2*dt <= AFD_ONE) ) {

			// printf("down\n");

			// Apply L^(-1) to the curve. Decrease curve resolution.
			dx = 2 * dx + ddx;
			dy = 2 * dy + ddy;
			ddx = 4 * ddx + 4 * dddx;
			ddy = 4 * ddy + 4 * dddy;
			dddx = 8 * dddx;
			dddy = 8 * dddy;

			// Double the stepsize.
			dt <<= 1;

			// Recompute d
			d = ddx*ddx + ddy*ddy + dddx*dddx + dddy*dddy;

		}

		// Forward differencing.
		px += dx;
		py += dy;
		dx += ddx;
		dy += ddy;
		ddx += dddx;
		ddy += dddy;

		// Output a point.
		verbs.push_back(VPathVerb::kLine);
        points.push_back(VPoint::Make(px, py));

		// Advance along the curve.
		t += dt;

		// Ensure we don't overshoot.
		assert(t <= AFD_ONE);

	}

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
    
    
    for (const auto& verb: verbs) {
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
                    // VPoint p01 = _last.Lerp(control, t);
                    // VPoint p12 = control.Lerp(point, t);
                    // VPoint line = p01.Lerp(p12, t);

                    outPoints.push_back(evalQuadBez(_last, control, point, t));
                    // outPoints.push_back(line);
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
                const VPoint& c0 = points[i];
                const VPoint& c1 = points[i + 1];
                const VPoint& p1 = points[i + 2];

                VPoint a = p1 - _last + 3.0f * (c0 - c1);
                VPoint b = 3.0f * (_last - 2.0f * c0 + c1);

                float conc = std::max(b.Length(), (a + b).Length());
                // float dt = fastInverseSqrt(conc) * sqrt_of_8_tol;
                float dt = std::sqrt(sqrt_of_8_tol / conc);
                float t = dt;

                // // http://www.pennelynn.com/Documents/CUJ/HTML/15.11/BARTLEY/BARTLEY.HTM
                float dt2 = dt * dt;
                float dt3 = dt2 * dt; 
                VPoint c = 3.0f * (c0 - _last);            
                VPoint d = _last;
                VPoint adt3 = a * dt3;
                VPoint bdt2 = b * dt2;
                // Initial values
                VPoint f = d;
                VPoint df = adt3 + bdt2 + c * dt;
                VPoint dddf = 6.0f * adt3;
                VPoint ddf = dddf + 2.0f * bdt2;
                
                while (t < 1.0f) {
                    f = f + df;
                    df = df + ddf;
                    ddf = ddf + dddf;
                    outPoints.push_back(f);
                    outVerbs.push_back(VPathVerb::kLine);
                    t += dt;
                }

                // while (t < 1.0f) {
                //     // t = std::min(t + dt, 1.0f);
                //     VPoint p01 = _last.Lerp(c0, t);
                //     VPoint p12 = c0.Lerp(c1, t);
                //     VPoint p23 = c1.Lerp(p1, t);
                //     VPoint p012 = p01.Lerp(p12, t);
                //     VPoint p123 = p12.Lerp(p23, t);
                //     VPoint line = p012.Lerp(p123, t);
                //     outPoints.push_back(line);
                //     outVerbs.push_back(VPathVerb::kLine);
                //     t += dt;
                // }

                // while (t < 1.0f) {
                //     outPoints.push_back(evalCubicBez(p0, c0 * 3.0f, c1 * 3.0f, p1, t));
                //     outVerbs.push_back(VPathVerb::kLine);
                //     t += dt;
                // }

                // nvg__tesselateBezierAFD(p0.x, p0.y, c0.x, c0.y, c1.x, c1.y, p1.x, p1.y, tolerance, outVerbs, outPoints);

                outPoints.push_back(p1);
                outVerbs.push_back(VPathVerb::kLine);
                // CubicBezToQuadratics(_last, control1, control2, point, tolerance, outVerbs, outPoints);
                _last = p1;
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
