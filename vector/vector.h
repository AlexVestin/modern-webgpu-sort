#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#include "defs.h"
#include "Validation.h"

#include "Renderer.h"
#include "Flatten.h"
#include "path/SVGUtil.h"


struct VectorPathBuilder {
    void Load();
    void Process();
    void Zoom(float amount);
    void Move(float x, float y);

private:
    std::vector<lyra::SVGUtil::Element> elements;
    std::vector<Span> spans;
    BitArray flatVerbs;
    std::vector<uint32_t> atlasIndices;
    std::vector<VPoint> flatPointsGlobal;
    std::vector<uint32_t> indicesGlobal;
    std::vector<DrawSpan> drawSpansGlobal;
    std::vector<uint32_t> colors;
    Renderer renderer;
    float transform[6] = {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
};