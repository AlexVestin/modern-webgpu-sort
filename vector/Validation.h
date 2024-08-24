#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <list>

#include "Flatten.h"
#include "path/SVGUtil.h"

#include "../ComputeUtil.h"
#include "defs.h"

void RenderToAtlas2(
    const std::vector<DrawSpan>& spans, 
    const std::vector<uint32_t>& indices, 
    const std::vector<VPoint>& flatLinePoints,
    const std::vector<uint32_t>& atlasIndices);

void Render(
    uint32_t start,
    const std::vector<DrawSpan>& spans, 
    const std::vector<uint32_t>& indices, 
    const std::vector<VPoint>& flatLinePoints, 
    const std::vector<uint32_t>& col);


uint32_t EstimatedFullTileSize();
//  {
    // const auto& rect = el.path.GetLocalBounds(paintStyle);
    // int32_t t = RoundDownToTile(rect.t);
    // int32_t b = RoundDownToTile(rect.b) + TILE_SIZE;
    // int32_t l = static_cast<int32_t>(std::floor(rect.l));
    // int32_t r = static_cast<int32_t>(std::ceil(rect.r));
    // estimatedTileArea += std::abs((b - t) * (r - l));
// }


void WriteImages();
