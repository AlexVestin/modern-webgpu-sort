// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)

#pragma once

#include <inttypes.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

#include "VPaint.h"
#include "VPoint.h"
#include "kurbo_stroke.h"
#include "LTRBRect.h"
#include "linalg.h"

using namespace linalg::aliases;
using namespace linalg::ostream_overloads;

enum class VPathVerb : uint8_t { 
    kMove, 
    kLine, 
    kQuad, 
    kClose, 
    kCubic, 
};

struct SubPath {
    size_t startVerbIndex = 0;
    size_t startPointIndex = 0;
};

struct PathComponents {
    // Data
    std::vector<VPathVerb> pathVerbs;
    std::vector<VPoint> untransformedPoints;

    // Computed properties
    std::vector<VPoint> points;
    std::vector<uint32_t> encodedCommands;
    bool needsNewEncoding = true;
    uint32_t previousEncodedPathId = 999999999u;
    lyra::LTRBRect bounds;

    void Reset() {
        pathVerbs.clear();
        points.clear();
        untransformedPoints.clear();
        encodedCommands.clear();
        previousEncodedPathId = 999999999u;
        needsNewEncoding = true;
        bounds.Reset();
    }

    void FromVector(const VPathVerb* verbs, uint32_t numVerbs, const VPoint* points, uint32_t numPoints) {
        Reset();
        untransformedPoints = std::vector(points, points + numPoints);
        pathVerbs = std::vector(verbs, verbs + numVerbs); 

        for (auto& p: untransformedPoints) {
            bounds.Expand(p);
        }
        this->points.resize(numPoints);       
    }

    PathComponents Copy() const {
        PathComponents copy;
        copy.points = this->points;
        copy.encodedCommands = this->encodedCommands;
        copy.points.resize(copy.points.size());
        return copy;
    }
};

struct StrokeComponents : public PathComponents {
    kurbo::Path kurboPath;
    std::vector<SubPath> subPaths = {};
};


const lyra::LTRBRect invalidBoundingBox = {};

class VPath {
   public:
    VPath();
    VPath(const float* transform) : VPath() { SetTransform(transform); }
    void Clear();

    void AddSquare(float x, float y, float w, float h);
    void AddSquarePoint(float x, float y, float w);
    void AddRect(const lyra::Rect& r);

    VPath Copy() const {
        VPath copy;
        copy.SetTransform(this->transform);
        copy.fillComponents = this->fillComponents.Copy();
        return copy;
    }

    void Deserialize(const VPathVerb* verbs, uint32_t numVerbs, const VPoint* points, uint32_t numPoints) {
        fillComponents.FromVector(verbs, numVerbs, points, numPoints);
        strokeComponents.Reset();
    }

    void ExpandBoundingRect(PaintStyle paintStyle, float x, float y) {
        assert(!std::isnan(x) && std::isfinite(x));
        assert(!std::isnan(y) && std::isfinite(y));

        if (paintStyle == PaintStyle::kFill) {
            fillComponents.bounds.Expand(x, y);
        } else {
            strokeComponents.bounds.Expand(x, y);
        }
    }

    void ExpandBoundingRect(PaintStyle paintStyle, const lyra::LTRBRect& r) {
        assert(!std::isnan(r.l) && std::isfinite(r.l));
        assert(!std::isnan(r.t) && std::isfinite(r.t));
        assert(!std::isnan(r.r) && std::isfinite(r.r));
        assert(!std::isnan(r.b) && std::isfinite(r.b));

        if (paintStyle == PaintStyle::kFill) {
            fillComponents.bounds.Expand(r.l, r.t);
            fillComponents.bounds.Expand(r.r, r.b);
        } else {
            strokeComponents.bounds.Expand(r.l, r.t);
            strokeComponents.bounds.Expand(r.r, r.b);
        }
    }

     const std::vector<VPoint>& GetUntransformedPoints(PaintStyle paintStyle, uint32_t index = 0) const {
        if (paintStyle == PaintStyle::kFill) {
            return fillComponents.untransformedPoints;
        } else {
            return strokeComponents.untransformedPoints;
        }
    }

    const std::vector<VPoint>& GetPoints(PaintStyle paintStyle, uint32_t index = 0) const {
        if (paintStyle == PaintStyle::kFill) {
            return fillComponents.points;
        } else {
            return strokeComponents.points;
        }
    }
    const std::vector<VPathVerb>& GetVerbs(PaintStyle paintStyle, uint32_t index = 0) const {
        if (paintStyle == PaintStyle::kFill) {
            return fillComponents.pathVerbs;
        } else {
            return strokeComponents.pathVerbs;
        }
    }

    const std::vector<uint32_t>& GetEncoding(PaintStyle paintStyle, uint32_t index = 0) const {
        if (paintStyle == PaintStyle::kFill) {
            return fillComponents.encodedCommands;
        } else {
            return strokeComponents.encodedCommands;
        }
    }

    void SetBounds(PaintStyle style, const lyra::LTRBRect& b) {
        if (style == PaintStyle::kFill) {
            fillComponents.bounds = b;
        } else {
            strokeComponents.bounds = b;
        }
    }

    void Transform(const float* transform);

    const float* GetTransform() const { return transform; }
    void SetTransform(const float* transform) {
        memcpy(&this->transform[0], transform, 6 * sizeof(float));
    }

    uint32_t EncodeCommand(VPathVerb verb, uint32_t index) {
        return ((static_cast<uint32_t>(verb) & 0xffu) << 24u) | (index & 0x00FFFFFF);
    }

    void EnsureEncoding(uint32_t encodedPathId, PaintStyle paintStyle) {
        if (paintStyle == PaintStyle::kFill) {
            EnsureEncoding(fillComponents, encodedPathId);
        } else {
            EnsureEncoding(strokeComponents, encodedPathId);
        }
    }

    std::string dd(const VPoint& p) {
        std::stringstream ss;
        ss << std::setprecision(15);
        ss << p.x << " " << p.y;
        return ss.str();
    }

    // L 1.4013e-45 2.62043e-43 200 496 L200 496 142 496 L142 496 140 496 L140 496 140 494 L140 494 140 400 L140 400 140 398 L140 398 142 398 L142 398 200 398 L200 398 202 398 L202 398 202 400 L202 400 202 494 L202 494 202 496 Z L200 496 200 492 L200 492 198 494 L198 494 198 400 L198 400 200 402 L200 402 142 402 L142 402 144 400 L144 400 144 494 L144 494 142 492 Z 
    void DebugStoreToDisk() {
        uint32_t firstOffset = (strokeComponents.encodedCommands[0] & 0x00ffffffu);
        std::stringstream sstream;
        const std::vector<VPoint>& points = strokeComponents.points;

        
        for (int j = 0; j < strokeComponents.encodedCommands.size(); j += 2) {
            uint32_t v = strokeComponents.encodedCommands[j];
            uint32_t offset = ((v & 0x00ffffffu) - firstOffset) + 1u;
            uint32_t cmd = v >> 24u;
            switch(cmd) {
                case 0:
                    sstream << "M" << dd(points[offset]) << " ";
                    break;
                case 1:
                    sstream << "L" << dd(points[offset - 1]) << " " << dd(points[offset]) << " ";
                    break;
                case 3:
                    sstream << "Z ";
            }
        }
        
        std::cout << sstream.str() << std::endl;
    }

    const lyra::LTRBRect& GetLocalBounds(PaintStyle paintStyle) const {
        if (isInvalid) {
            return invalidBoundingBox;
        }

        if (paintStyle == PaintStyle::kFill) {
            return fillComponents.bounds;
        } else {
            return strokeComponents.bounds;
        }
    }

    void Retransform();
    bool IsExpandedStroke() const { return isExpandedStroke; }
    bool IsInvalid() const { return isInvalid; }

    // Path Commands
    void QuadTo(float cpx, float cpy, float x, float y);
    void CubicTo(float cpx1, float cpy1, float cpx2, float cpy2, float x, float y);
    void LineTo(float x, float y);
    void MoveTo(float x, float y);
    void Close();
    void ExpandStroke(const kurbo::StrokeStyle& style);

   private:
    void TransformInternal(const float* transform, PathComponents& pathComponents);

    void EnsureEncoding(PathComponents& c, uint32_t encodedPathId) {
        if (!c.needsNewEncoding) {
            if (c.previousEncodedPathId != encodedPathId) {
                for (int i = 0; i < c.encodedCommands.size(); i += 2) {
                    c.encodedCommands[i + 1] = encodedPathId;
                }
                c.previousEncodedPathId = encodedPathId;
            }
            return;
        }
        c.encodedCommands.clear();

        uint32_t index = 0u;
        uint32_t lastMoveOffset = 0u;

        const std::vector<VPathVerb>& verbs = c.pathVerbs;
        for (auto& verb : verbs) {
            uint32_t offset = index;
            switch (verb) {
                case VPathVerb::kClose:
                    offset = lastMoveOffset;
                    break;
                case VPathVerb::kMove:
                    lastMoveOffset = offset;
                    index += 1u;
                    break;
                case VPathVerb::kLine:
                    index += 1u;
                    break;
                case VPathVerb::kQuad:
                    index += 2u;
                    break;
                case VPathVerb::kCubic:
                    index += 3u;
                    break;
            }

            if (verb != VPathVerb::kMove) {
                c.encodedCommands.push_back(EncodeCommand(verb, offset));
                c.encodedCommands.push_back(encodedPathId);
            }
        }

        c.previousEncodedPathId = encodedPathId;
        c.needsNewEncoding = false;
    }
    bool isExpandedStroke = false;
    bool isInvalid = false;

    // Bounds before scaling is applied so we get the correct rotation in screen space
    void AddCommand(VPathVerb verb, float* vals, uint32_t numVals);

    PathComponents fillComponents;
    StrokeComponents strokeComponents;

    float transform[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
};
