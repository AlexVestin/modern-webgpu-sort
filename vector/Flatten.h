#pragma once

#include <vector>

#include "path/VPoint.h"
#include "path/VPath.h"

struct FlatCommand {
    VPoint point;
    VPathVerb verb;
} __attribute__ ((aligned (16)));;

std::vector<FlatCommand> FlattenCommands(const std::vector<VPathVerb>& verbs, const std::vector<VPoint>& points, float tolerance);