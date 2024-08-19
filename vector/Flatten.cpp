
#include "Flatten.h"

const float sqrt_of_8 = 2.82842712475f;
const float sqrt_of_8_tol = 2.82842712475f * 0.175f;


// y is in range [0, height - TILE_SIZE]
// x is in range [-inf, width - TILE_SIZE]
// TODO: clip lines more negative than -32767
static inline uint32_t PackPosition(int32_t x, int32_t y) {
    return (static_cast<uint32_t>(y) << 16u) | (static_cast<uint32_t>(x + 32767) & 0xffffu);
}

static inline int2 UnpackPosition(uint32_t v) {
  return {static_cast<int32_t>(v & 0xffffu) - 32767, static_cast<int32_t>(v >> 16u)};
}

static inline int32_t RoundDownToTile(float v) {
    return static_cast<int32_t>(v * TILE_SIZE_DIV) * TILE_SIZE;
}

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



uint32_t FlattenCommandsCombined(
    const std::vector<VPathVerb>& verbs, 
    const std::vector<VPoint>& points, 
    std::vector<VPathVerb>& outVerbs,
    std::vector<VPoint>& outPoints,
    const float tolerance,
    uint32_t baseLineIndex,
    std::vector<Span>& spans) {

    VPoint p0 = points[0];
    VPoint first = p0;
    outPoints[baseLineIndex] = p0;

    // Traversal stuff
    int32_t spanTileY = RoundDownToTile(p0.y);
    uint32_t spanEntryDirection = ~0u;

    float spanMinX = p0.x;
    float spanMaxX = p0.x;

    uint32_t spanLineStartIndex = baseLineIndex + 1u;
    uint32_t contourId = spans.size();

    auto EmitClose = [&](uint32_t i) {
        Span span;
        span.key = PackPosition(spanMinX, spanTileY);
        span.lineStartIndex = spanLineStartIndex;
        span.PackTypeLineEndIndex(0, i - 1);
        span.spanMaxX = spanMaxX;

        // if we didn't exit the current span we dont need to update 

        // TODO: when is contourType ever != 0?
        // if (contourId < spans.size() && spans[contourId].GetType() == spanEntryDirection) {
        //     spans[contourId].SetType(spanEntryDirection);
        // }
        if (contourId < spans.size()) {
            uint32_t contourType = spans[contourId].GetType();  
            
            if ((contourType == 0 && (spanEntryDirection == 0u || spanEntryDirection == ~0u)) || (contourType != spanEntryDirection)) {
                spans[contourId].SetType(0);
            } else {
                spans[contourId].SetType(spanEntryDirection);
            }
        }
        
        spans.push_back(span);
    };

    auto TraverseLine = [&](const VPoint& p1, uint32_t i) {
        // Horizontal lines
        if (p1.y >= spanTileY && p1.y < spanTileY + TILE_SIZE) {
            spanMaxX = std::max(spanMaxX, p1.x);
            spanMinX = std::min(spanMinX, p1.x);
            p0 = p1;
            return;
        }
        
        bool downward = p1.y > p0.y;
        int32_t step = downward ? TILE_SIZE : -TILE_SIZE;
        int32_t offset = downward ? TILE_SIZE : 0;

        float slope = (p1.x - p0.x) / (p1.y - p0.y);
        float ymin = std::min(p0.y, p1.y);
        float ymax = std::max(p0.y, p1.y);

        float xv0 = p0.x;
        float xv1 = p0.x + (std::clamp(static_cast<float>(spanTileY + offset), ymin, ymax) - p0.y) * slope; 

        spanMaxX = std::max(spanMaxX, xv1);
        spanMinX = std::min(spanMinX, xv1);
        
        int32_t yc = spanTileY + step;       
        int32_t y1 = RoundDownToTile(p1.y);

        while (yc != y1 + step) {
            // Push span
            Span span;
            span.key = PackPosition(spanMinX, spanTileY);
            span.lineStartIndex = spanLineStartIndex;
            span.spanMaxX = spanMaxX;

            uint32_t type = downward ? DIRECTION_DOWN : DIRECTION_UP;
            if (type == spanEntryDirection || spanEntryDirection == ~0u) {
                span.PackTypeLineEndIndex(type, i);
            } else {
                span.PackTypeLineEndIndex(0, i);
            }
            spans.push_back(span);

            // Update counters
            spanLineStartIndex = i;                    
            spanEntryDirection = type;    
            spanTileY = yc;

            xv0 = xv1;
            xv1 = p0.x + (std::clamp(static_cast<float>(yc + offset), ymin, ymax) - p0.y) * slope; 

            spanMinX = std::min(xv0, xv1);
            spanMaxX = std::max(xv0, xv1);
            
            yc += step;
        }

        p0 = p1;
    };


    size_t i = 1;
    const float tolerance4  = tolerance * 4.0f;
    
    uint32_t lineIndex = baseLineIndex + 1;
    for (int j = 1; j < verbs.size(); j++) {
        switch (verbs[j]) {
            case VPathVerb::kClose:
                outPoints[lineIndex++] = first;
                TraverseLine(first, lineIndex);
                break;
            case VPathVerb::kMove:
                first = points[i];
                p0 = first;
                outPoints[lineIndex++] = first;

                // Traversal close
                EmitClose(i);
                spanLineStartIndex = lineIndex;
                spanTileY = RoundDownToTile(p0.y);
                spanEntryDirection = ~0u;
                spanMaxX = p0.x;
                spanMinX = p0.x;
                contourId = spans.size();
                i++;
                break;
            case VPathVerb::kLine:  {
                const VPoint& line = points[i];
                outPoints[lineIndex++] = line;
                TraverseLine(line, lineIndex);
                p0 = line;
                i++;
                break;
            }
            case VPathVerb::kQuad: {
                const VPoint& control = points[i];
                const VPoint& point = points[i + 1];

                float l = (p0 - 2.0f * control + point).Length();
                float dt = std::sqrt(tolerance4 / l);
                float t = std::min(dt, 1.0f);

                VPoint ogp = p0;
                while (t < 1.0f) {
                    VPoint p = evalQuadBez(ogp, control, point, t);
                    outPoints[lineIndex++] = p;
                    TraverseLine(p, lineIndex);
                    t += dt;
                }
                outPoints[lineIndex++] = point;
                TraverseLine(point, lineIndex);
                p0 = point;
                i += 2;
                break;
            }
            case VPathVerb::kCubic: {
                const VPoint& c0 = points[i];
                const VPoint& c1 = points[i + 1];
                const VPoint& p1 = points[i + 2];

                VPoint a = p1 - p0 + 3.0f * (c0 - c1);
                VPoint b = 3.0f * (p0 - 2.0f * c0 + c1);

                float conc = std::max(b.Length(), (a + b).Length());
                // float dt = fastInverseSqrt(conc) * sqrt_of_8_tol;
                float dt = std::sqrt(sqrt_of_8_tol / conc);
                float t = dt;

                // // http://www.pennelynn.com/Documents/CUJ/HTML/15.11/BARTLEY/BARTLEY.HTM
                float dt2 = dt * dt;
                float dt3 = dt2 * dt; 
                VPoint c = 3.0f * (c0 - p0);            
                VPoint d = p0;
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
                    outPoints[lineIndex++] = f;
                    TraverseLine(f, lineIndex);
                    p0 = f;
                    t += dt;
                }

                outPoints[lineIndex++] = p1;
                TraverseLine(p1, lineIndex);
                p0 = p1;
                i += 3;
                break;
            }

            default: {
                std::cerr << "Verb was: " << " At position: " << i << " buffer size: " << points.size()
                          << std::endl;
                exit(1);
            }
        }
    }

    EmitClose(lineIndex);

    return lineIndex;
}


uint32_t FlattenCommands2(
    const std::vector<VPathVerb>& verbs, 
    const std::vector<VPoint>& points, 
    std::vector<VPathVerb>& outVerbs,
    std::vector<VPoint>& outPoints,
    const float tolerance,
    uint32_t baseLineIndex) {

    VPoint p0 = points[0];
    VPoint first = p0;
    
    uint32_t lineIndex = baseLineIndex;
    outPoints[lineIndex] = p0;
    outVerbs[lineIndex++] = VPathVerb::kMove;

    size_t i = 1;
    const float tolerance4  = tolerance * 4.0f;
    const float sqrt_of_8 = 2.82842712475f;
    for(int j = 1; j < verbs.size(); j++) {
        switch (verbs[j]) {
            case VPathVerb::kClose:
                // outPoints.push_back(first);
                // outVerbs.push_back(VPathVerb::kLine);

                outPoints[lineIndex] = first;
                outVerbs[lineIndex++] = VPathVerb::kLine;
                break;

            case VPathVerb::kMove:
                first = points[i];
                p0 = first;
                outPoints[lineIndex] = first;
                outVerbs[lineIndex++] = VPathVerb::kMove;
                i++;
                break;
            case VPathVerb::kLine:  {
                const VPoint& line = points[i];
                outPoints[lineIndex] = line;
                outVerbs[lineIndex++] = VPathVerb::kLine;
                p0 = line;
                i++;
                break;
            }
            case VPathVerb::kQuad: {
                const VPoint& control = points[i];
                const VPoint& point = points[i + 1];

                float l = (p0 - 2.0f * control + point).Length();
                float dt = std::sqrt(tolerance4 / l);
                float t = std::min(dt, 1.0f);
                while (t < 1.0f) {
                    outPoints[lineIndex] = evalQuadBez(p0, control, point, t);
                    outVerbs[lineIndex++] = VPathVerb::kLine;
                    t += dt;
                }

                outPoints[lineIndex] = point;
                outVerbs[lineIndex++] = VPathVerb::kLine;
 

                // QuadBezFlatten(_last, control, point, tolerance, outVerbs, outPoints);
                p0 = point;
                i += 2;
                break;
            }
            case VPathVerb::kCubic: {
                const VPoint& c0 = points[i];
                const VPoint& c1 = points[i + 1];
                const VPoint& p1 = points[i + 2];

                VPoint a = p1 - p0 + 3.0f * (c0 - c1);
                VPoint b = 3.0f * (p0 - 2.0f * c0 + c1);

                float conc = std::max(b.Length(), (a + b).Length());
                // float dt = fastInverseSqrt(conc) * sqrt_of_8_tol;
                float dt = std::sqrt(sqrt_of_8_tol / conc);
                float t = dt;

                // // http://www.pennelynn.com/Documents/CUJ/HTML/15.11/BARTLEY/BARTLEY.HTM
                float dt2 = dt * dt;
                float dt3 = dt2 * dt; 
                VPoint c = 3.0f * (c0 - p0);            
                VPoint d = p0;
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

                    outPoints[lineIndex] = f;
                    outVerbs[lineIndex++] = VPathVerb::kLine;
                    t += dt;
                }

                outPoints[lineIndex] = p1;
                outVerbs[lineIndex++] = VPathVerb::kLine;
                // CubicBezToQuadratics(_last, control1, control2, point, tolerance, outVerbs, outPoints);
                p0 = p1;
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

    return lineIndex;
}
