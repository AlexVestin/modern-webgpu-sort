#pragma once

#include <vector>

#include "path/VPoint.h"
#include "path/VPath.h"

struct FlatCommand {
    VPathVerb verb;
    VPoint point;
};

std::vector<FlatCommand> FlattenCommands(const std::vector<VPathVerb>& verbs, const std::vector<VPoint>& points, float tolerance);