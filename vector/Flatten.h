#pragma once

#include <vector>

#include "path/VPoint.h"
#include "path/VPath.h"
#include "defs.h"

struct FlatCommand {
    VPoint point;
    VPathVerb verb;
} __attribute__ ((aligned (16)));

uint32_t FlattenCommands2(
    const std::vector<VPathVerb>& verbs, 
    const std::vector<VPoint>& points, 
    BitArray& outVerbs,
    std::vector<VPoint>& outPoints,
    float tolerance,
    uint32_t baseLineIndex);
