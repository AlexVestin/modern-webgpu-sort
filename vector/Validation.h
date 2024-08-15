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

#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"
#include "defs.h"


void RenderToAtlas(
    const std::vector<DrawSpan>& spans, 
    const std::vector<uint32_t>& indices, 
    const std::vector<VPoint>& flatLinePoints, 
    std::array<float, IMAGE_WIDTH * IMAGE_HEIGHT>& image);

void Render(
    uint32_t start,
    const std::vector<DrawSpan>& spans, 
    const std::vector<uint32_t>& indices, 
    const std::vector<VPoint>& flatLinePoints, 
    std::array<uint32_t, IMAGE_WIDTH * IMAGE_HEIGHT>& image, 
    const std::vector<uint32_t>& col, 
    const std::array<float, IMAGE_WIDTH * IMAGE_HEIGHT>& atlas);