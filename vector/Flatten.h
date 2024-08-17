#pragma once

#include <vector>

#include "path/VPoint.h"
#include "path/VPath.h"

struct FlatCommand {
    VPoint point;
    VPathVerb verb;
} __attribute__ ((aligned (16)));

void FlattenCommands2(
    const std::vector<VPathVerb>& verbs, 
    const std::vector<VPoint>& points, 
    std::vector<VPathVerb>& outVerbs,
    std::vector<VPoint>& outPoints,
    float tolerance);