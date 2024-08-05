// Copyright 2021, Alexander Vestin (alex.vestin@gmail.com)

#include "VPath.h"

#include <stdio.h>

#include <cassert>

VPath::VPath() {}

void VPath::MoveTo(float x, float y) {
    // subPaths.push_back({.startVerbIndex = pathVerbs.size(), .startPointIndex = untransformedPoints.size()});
    float vals[] = {x, y};
    AddCommand(VPathVerb::kMove, vals, 2);
}

void VPath::LineTo(float x, float y) {
    float vals[] = {x, y};
    AddCommand(VPathVerb::kLine, vals, 2);
}

void VPath::Close() {
    // Needs at least a move and line before close
    // assert(pathVerbs.size() >= 2);
    if (fillComponents.pathVerbs[fillComponents.pathVerbs.size() - 1] == VPathVerb::kClose) {
        return;
    }
    AddCommand(VPathVerb::kClose, nullptr, 0);
}

void VPath::QuadTo(float cx, float cy, float x, float y) {
    float vals[] = {cx, cy, x, y};
    AddCommand(VPathVerb::kQuad, vals, 4);
}

void VPath::CubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) {
    float vals[] = {cx1, cy1, cx2, cy2, x, y};
    AddCommand(VPathVerb::kCubic, vals, 6);
}

void VPath::Clear() {
    isInvalid = false;
    fillComponents.Reset();
    strokeComponents.Reset();
}

void VPath::Transform(const float* transform) {
    TransformInternal(transform, fillComponents);
    if (isExpandedStroke) {
        TransformInternal(transform, strokeComponents);
    }
}

void VPath::TransformInternal(const float* transform, PathComponents& pathComponents) {
    const std::vector<VPathVerb>& verbs = pathComponents.pathVerbs;
    const std::vector<VPoint>& srcPoints = pathComponents.untransformedPoints;
    std::vector<VPoint>& dstPoints =  pathComponents.points;

    assert(dstPoints.size() == srcPoints.size());

    size_t i = 0;
    for (auto& verb : verbs) {
        switch (verb) {
            case VPathVerb::kClose:
                break;
            case VPathVerb::kMove:
            case VPathVerb::kLine:

                if (i >= srcPoints.size()) {
                    std::cout << "i: " << i << " " << verbs.size() << " " << srcPoints.size() << " " << static_cast<uint32_t>(verb) << std::endl;        
                }
                assert(i < srcPoints.size());
                dstPoints[i] = srcPoints[i].transform(transform);
                i += 1;
                break;
            case VPathVerb::kQuad:
                assert(i + 1 < srcPoints.size());
                dstPoints[i] = srcPoints[i].transform(transform);
                dstPoints[i + 1] = srcPoints[i + 1].transform(transform);
                i += 2;
                break;
            case VPathVerb::kCubic:
                assert(i + 2 < srcPoints.size());
                dstPoints[i] = srcPoints[i].transform(transform);
                dstPoints[i + 1] = srcPoints[i + 1].transform(transform);
                dstPoints[i + 2] = srcPoints[i + 2].transform(transform);
                i += 3;
                break;
            default:
                std::cerr << "Unrecognized command: " << static_cast<uint32_t>(verb) << ":" << srcPoints[i] << "("
                          << verbs.size() << "," << i << ")" << std::endl;
                exit(1);
        }
    }
}

void VPath::AddRect(const lyra::Rect& r) {
    MoveTo(r.p0.x, r.p0.y);
    LineTo(r.p1.x, r.p1.y);
    LineTo(r.p2.x, r.p2.y);
    LineTo(r.p3.x, r.p3.y);
    Close();
}

void VPath::AddSquare(float x, float y, float w, float h) {
    MoveTo(x, y);
    LineTo(x + w, y);
    LineTo(x + w, y + h);
    LineTo(x, y + h);
    Close();
}

void VPath::AddSquarePoint(float x, float y, float w) {
    float hw = w / 2;
    AddSquare(x - hw, y - hw, w, w);
}

void VPath::Retransform() {
    isInvalid = false;
    if (std::fabs(transform[0] + transform[2]) < 0.00001f || std::fabs(transform[1] + transform[3]) < 0.00001f) {
        isInvalid = true;        
    } else {
        Transform(transform);    
    }
}

void VPath::ExpandStroke(const kurbo::StrokeStyle& styling) {
    assert(styling.tolerance > 0.0f);

    if (isExpandedStroke) {
        kurbo::free_arrays(strokeComponents.kurboPath);
    }

    const std::vector<VPoint>& srcPoints = fillComponents.untransformedPoints;
    const std::vector<VPathVerb>& srcVerbs = fillComponents.pathVerbs;

    if (styling.strokeWidth <= 0.0f || srcPoints.empty() || srcVerbs.empty()) {
        std::cout << "Not expanding " << styling.strokeWidth << " " << srcPoints.size() << " " << srcVerbs.size() << std::endl;
        return;
    }

    const float* srcData = reinterpret_cast<const float*>(srcPoints.data());
    const uint8_t* srcCmds = reinterpret_cast<const uint8_t*>(srcVerbs.data());
    kurbo::Path kurboPath =
        kurbo::kurbo_stroke(srcData, srcPoints.size() * 2,
                            srcCmds, srcVerbs.size(), styling);

    if (kurboPath.n_elements == 0 || kurboPath.n_points == 0) {
        std::cerr << "Kurbo failed to expand stroke" << std::endl;
        kurbo::free_arrays(kurboPath);
        isInvalid = true;
        return;
    }


    strokeComponents.pathVerbs = {reinterpret_cast<VPathVerb*>(kurboPath.types),
                   reinterpret_cast<VPathVerb*>(kurboPath.types + kurboPath.n_elements)};
    strokeComponents.untransformedPoints = {reinterpret_cast<VPoint*>(kurboPath.points),
                    reinterpret_cast<VPoint*>(kurboPath.points + kurboPath.n_points)};
    isExpandedStroke = true;
        
    strokeComponents.bounds.Reset();
    for (auto& v : strokeComponents.untransformedPoints) {
        strokeComponents.bounds.Expand(v);
    }
    //   for (auto& v : fillComponents.untransformedPoints) {
    //     strokeComponents.bounds.Expand(v);

    strokeComponents.kurboPath = kurboPath;
    strokeComponents.points.resize(strokeComponents.untransformedPoints.size());
    
    strokeComponents.needsNewEncoding = true;
    TransformInternal(transform, strokeComponents);
}

void VPath::AddCommand(VPathVerb verb, float* vals, uint32_t numVals) {
    if (isInvalid) {
        return;
    }
 

    for (uint32_t i = 0; i < numVals / 2; i++) {
        VPoint v = VPoint::Make(vals[i * 2], vals[i * 2 + 1]);
        if (!v.isCorrect()) {
            isInvalid = true;
            break;
        }

        fillComponents.bounds.Expand(v);
        fillComponents.untransformedPoints.push_back({v.x, v.y});
        fillComponents.points.push_back(v.transform(transform));
    }

    if(!isInvalid) {
        fillComponents.pathVerbs.push_back(verb);
    }

    fillComponents.needsNewEncoding = true;
}
