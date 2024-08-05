#pragma once

#include <limits>
#include <vector>
#include "VPoint.h"

namespace lyra {

enum class BoundsEdge {
    kLeft,
    kTop,
    kBottom,
    kRight,
    kTopLeft,
    kTopRight,
    kBottomRight,
    kBottomLeft,
    kNone,
    kCenter
};

enum class BoundsPosition {
    kLeft,
    kTop,
    kBottom,
    kRight,
    kTopLeft,
    kTopRight,
    kBottomRight,
    kBottomLeft,
    kHorizontal,
    kVertical,
    kCenter,
};

constexpr float fvalmax = std::numeric_limits<float>::infinity();
struct Rect {
    VPoint p0, p1, p2, p3;

    float TriangleArea(VPoint a, VPoint b, VPoint c) const {
        return std::abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2.0;
    }

    VPoint GetCenter() const { return (p0 + p1 + p2 + p3) / VPoint::Make(4.0, 4.0); }

    // Assumes non self intersecting, and
    float Size() const { return TriangleArea(p0, p1, p2) + TriangleArea(p0, p2, p3); }

    Rect Transformed(const float* transform) const {
        return {.p0 = p0.transform(transform),
                .p1 = p1.transform(transform),
                .p2 = p2.transform(transform),
                .p3 = p3.transform(transform)};
    }
};

struct LTRBRect {
    float l, t, r, b;

    LTRBRect() : l{fvalmax}, t{fvalmax}, r{-fvalmax}, b{-fvalmax} {}
    LTRBRect(float l, float t, float r, float b) : l{l}, t{t}, r{r}, b{b} {}

    std::vector<VPoint> Edge(const lyra::BoundsEdge& edge) const {
        switch (edge) {
            case lyra::BoundsEdge::kLeft:
                return {{l, t}, {l, b}};
            case lyra::BoundsEdge::kRight:
                return {{r, t}, {r, b}};
            case lyra::BoundsEdge::kTop:
                return {{l, t}, {r, t}};
            case lyra::BoundsEdge::kBottom:
                return {{l, b}, {r, b}};
            case lyra::BoundsEdge::kTopLeft:
                return {{l, t}};
            case lyra::BoundsEdge::kBottomRight:
                return {{r, b}};
            case lyra::BoundsEdge::kTopRight:
                return {{r, t}};
            case lyra::BoundsEdge::kBottomLeft:
                return {{l, b}};
            case lyra::BoundsEdge::kCenter:
                return {{l, t}, {l, b}, {r, t}, {r, b}};
            default:
                return {};
        }
    }

    std::array<VPoint, 5> ToPointsWithCenter() const {
        VPoint tl = {l, t};
        VPoint tr = {r, t};
        VPoint br = {r, b};
        VPoint bl = {l, b};
        VPoint c = {l + (r - l) / 2.0f, t + (b - t) / 2.0f};
        return std::array<VPoint, 5>({tl, tr, br, bl, c});
    }

    bool operator==(const LTRBRect& other) const {
        return l == other.l && t == other.t && r == other.r && b == other.b;
    }
    // We need at least two points to create a proper bounding box
    bool IsIncomplete() const { return (l == fvalmax) || (r == -fvalmax) || (t == fvalmax) || (b == -fvalmax); }

    float& operator[](int i) {
        switch (i) {
            case 0:
                return l;
            case 1:
                return t;
            case 2:
                return r;
            case 3:
                return b;
        }

        std::cerr << "Tried to access 0 < index > 3 in ltrbrect" << std::endl;
        exit(1);
    }

    // Overlap of this rectangle with another
    LTRBRect Overlap(const LTRBRect& other) const {
        float overlapL = std::max(this->l, other.l);
        float overlapT = std::max(this->t, other.t);
        float overlapR = std::min(this->r, other.r);
        float overlapB = std::min(this->b, other.b);

        // Check if there's no overlap; if either dimension is inverted, return an indicator (e.g., a rectangle with no
        // area)
        if (overlapL > overlapR || overlapT > overlapB) {
            // Indicate no overlap; this could be adjusted based on how you want to handle this case
            return LTRBRect(0, 0, 0, 0);
        }

        return LTRBRect(overlapL, overlapT, overlapR, overlapB);
    }

    bool Contains(const VPoint& p) const { return p.x >= l && p.x <= r && p.y >= t && p.y <= b; }

    bool Contains(float x, float y) const { return Contains(VPoint::Make(x, y)); }

    LTRBRect Translated(const VPoint& offset) const { return {l + offset.x, t + offset.y, r + offset.x, b + offset.y}; }

    LTRBRect Scaled(const VPoint& scale) const { return {l, t, r * scale.x, b * scale.y}; }

    void Set(float l, float t, float r, float b) {
        this->l = l;
        this->r = r;
        this->t = t;
        this->b = b;
    }

    void Reset() {
        l = fvalmax;
        r = -fvalmax;
        b = -fvalmax;
        t = fvalmax;
    }

    Rect ToRect() const {
        if (IsIncomplete()) {
            return {};
        }
        return {.p0 = VPoint::Make(l, t), .p1 = VPoint::Make(r, t), .p2 = VPoint::Make(r, b), .p3 = VPoint::Make(l, b)};
    }

    LTRBRect Transform(const float* transform) {
        LTRBRect copy = *this;
        Reset();

        Expand(VPoint::Make(copy.l, copy.t).transform(transform));
        Expand(VPoint::Make(copy.r, copy.t).transform(transform));
        Expand(VPoint::Make(copy.l, copy.b).transform(transform));
        Expand(VPoint::Make(copy.r, copy.b).transform(transform));
        return *this;
    }

    LTRBRect InverseTransform(const float* transform) {
        LTRBRect copy = *this;
        Reset();

        Expand(VPoint::Make(copy.l, copy.t).inverseTransform(transform));
        Expand(VPoint::Make(copy.r, copy.t).inverseTransform(transform));
        Expand(VPoint::Make(copy.l, copy.b).inverseTransform(transform));
        Expand(VPoint::Make(copy.r, copy.b).inverseTransform(transform));
        return *this;
    }

    void Expand(const VPoint& p) { Expand(p.x, p.y); }

    void Expand(const LTRBRect& other) {
        l = std::min(other.l, l);
        r = std::max(other.r, r);
        t = std::min(other.t, t);
        b = std::max(other.b, b);
    }

    void Expand(float x, float y) {
        l = std::min(x, l);
        r = std::max(x, r);
        t = std::min(y, t);
        b = std::max(y, b);
    }

    const LTRBRect& ExpandSides(float v) {
        l -= v;
        t -= v;
        r += v;
        b += v;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& o, const LTRBRect& a);
std::ostream& operator<<(std::ostream& o, const Rect& a);
}  // namespace lyra
