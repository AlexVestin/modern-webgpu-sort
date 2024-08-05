#pragma once

#include <string>
#include <vector>

#include "VPaint.h"
#include "VPath.h"
#include "../nanosvg.h"

namespace lyra::SVGUtil {
struct Element {
    VPath path;
    VPaint paint;
};
void SVGStringToPath(const std::string& svgPath, VPath& path);
void PrintPathAsSVG(const VPath& path);
std::vector<Element> ParseSVG(const NSVGimage* svgImage, const float* transform);
NSVGimage* ReadSVG(const std::string& svgContent, const std::string& label);
void NSVGPathToVPath(const NSVGpath* shape, VPath& path);
}  // namespace lyra::SVGUtil