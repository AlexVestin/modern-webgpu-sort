#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <list>

#include <thread>

#include "defs.h"
#include "Validation.h"
#include "Renderer.h"

#include "Flatten.h"
#include "path/SVGUtil.h"

const uint32_t flatPointsAllocation = 1 << 21u;
const uint32_t spansAllocation = 1 << 18;
const uint32_t drawSpansAllocation = 1 << 18;
const uint32_t indicesAllocation = 1 << 18;
const uint32_t atlasIndicesAllocation = 1 << 21u;

Renderer renderer(IMAGE_WIDTH, IMAGE_HEIGHT);

// y is in range [0, height - TILE_SIZE]
// x is in range [-inf, width - TILE_SIZE]
// TODO: clip lines more negative than -32767
static inline uint32_t PackPosition(int32_t x, int32_t y) {
    return (static_cast<uint32_t>(y) << 16u) | (static_cast<uint32_t>(x + 32767) & 0xffffu);
}

static inline int2 UnpackPosition(uint32_t v) {
  return {static_cast<int32_t>(v & 0xffffu) - 32767, static_cast<int32_t>(v >> 16u)};
}

static inline int32_t RoundDownToTile(float v) {
    return static_cast<int32_t>(std::floor(v * TILE_SIZE_DIV)) * TILE_SIZE;
}

static inline uint32_t Pack24And8(uint32_t lineEndIndex, uint32_t type) {
    return (type << 24u) | (lineEndIndex & 0xffffffu);
}

static inline int32_t SpanTypeToBackdrop(uint32_t direction) {
    if (direction == DIRECTION_UP) {
        return 1;
    } else if(direction == DIRECTION_DOWN) {
        return -1;
    }
    return 0;
}

struct AtlasManager {
    AtlasManager(uint32_t width, uint32_t height) : atlasWidth{width}, atlasHeight{height} { }

    bool Claim(uint32_t width, uint32_t height, uint2& pos) {
        uint32_t w = std::min(width, atlasWidth);
        allocation += w * height;

        for (std::list<uint2>::iterator it = freeRows.begin(); it != freeRows.end(); ++it){
            uint32_t spaceLeftInRow = atlasWidth - it->x;
            if (spaceLeftInRow > w) { 
                pos = *it;
                it->x += w;
                if (spaceLeftInRow - w < 4) {
                    freeRows.erase(it);
                }
                return true;
            }
        }

        if (freeRows.size() >= 4) {
            freeRows.pop_front();
        }
        
        uint32_t spaceLeftInRow = atlasWidth - counterX; // + 1?
        if (spaceLeftInRow < w) {
            if (spaceLeftInRow > 4)  {
                freeRows.push_back({counterX, counterY});
            }

            // Break into new row
            counterY += TILE_SIZE;
            // if (counterY >= atlasHeight) {
            //     // Fail
            //     // std::cerr << "Allocating outside" << std::endl;
            //     return false;
            // }

            counterX = 0u;  
        }

        // std::cout << "Allocating: " << w << " " << position() << std::endl;
        pos = uint2(counterX, counterY);
        counterX += w;
        return true;
    }   

    uint32_t UsedSpace() const {
        return (counterY * atlasWidth + counterX * TILE_SIZE);
    }

    uint32_t Allocation() const {
        return allocation;
    }

    uint2 position() const {
        return uint2(counterX, counterY);
    }

    uint32_t counterX = 0;
    uint32_t counterY = 0;
    uint32_t allocation = 0u;

    std::list<uint2> freeRows;
    const uint32_t atlasWidth;
    const uint32_t atlasHeight;
};

void MergeSpans(uint32_t spanStartId, const std::vector<Span>& spans, std::vector<uint32_t>& indices, std::vector<DrawSpan>& drawSpans, uint32_t pathId, AtlasManager& atlasManager, std::vector<uint32_t>& atlasIndices, const std::vector<VPoint>& flatPoints) {
    int32_t lastBackdrop = 0;
    auto EmitSpan = [&drawSpans, &lastBackdrop, pathId](uint32_t from, uint32_t to, int32_t x, int32_t y, int32_t maxX, int32_t backdrop) {
        if (maxX < 0 || y < 0 || y >= IMAGE_HEIGHT || x >= IMAGE_WIDTH) {
            return;
        }
        x = std::max(0, x);
        DrawSpan ds;
        ds.pathId = (static_cast<uint32_t>((maxX + 1u)) << 16u) | pathId;
        ds.position = PackPosition(x, y);
        ds.lineStartIndex = from;
        ds.lineEndIndex = Pack24And8(to,  static_cast<uint32_t>(lastBackdrop + 127));
        lastBackdrop = backdrop;
        drawSpans.push_back(ds);
    };

    const Span& span = spans[spanStartId];
    int32_t maxSpanX = span.spanMaxX;
    int32_t backdrop = SpanTypeToBackdrop(span.GetType());
    int2 spanPosition = UnpackPosition(span.key);
    uint32_t spanLineCount = span.NumLines();
    uint32_t lastIndexBase = indices.size();

    for (int k = 0; k <= spanLineCount; k += 256u) {
        indices.push_back(((span.lineStartIndex + k) & 0xffffffu) | (spanLineCount << 24u));
    }
    
    for (int i = spanStartId + 1u; i < spans.size(); i++) {
        const Span& newSpan = spans[i];
        int2 newSpanPosition = UnpackPosition(newSpan.key);

        bool canSplit = (newSpanPosition.x > maxSpanX) && (backdrop == 0);
        uint32_t lineCount = newSpan.NumLines();
        if ((newSpanPosition.y != spanPosition.y) || canSplit) {
            EmitSpan(lastIndexBase, indices.size(), spanPosition.x, spanPosition.y, maxSpanX, backdrop);
            lastIndexBase = indices.size();

            // set new span
            spanPosition = newSpanPosition;
            maxSpanX = newSpan.spanMaxX;
            backdrop = 0;
            lastBackdrop = 0;
        } else {
            if (newSpanPosition.x - maxSpanX > 1) {
                // previous
                EmitSpan(lastIndexBase, indices.size(), spanPosition.x, spanPosition.y, maxSpanX, backdrop);
                lastIndexBase = indices.size();
                // gap
                EmitSpan(lastIndexBase, lastIndexBase, maxSpanX + 1u, spanPosition.y, newSpanPosition.x - 1u, backdrop);
                spanPosition.x = newSpanPosition.x;
            }
            maxSpanX = std::max(maxSpanX, newSpan.spanMaxX);
        }

        backdrop += SpanTypeToBackdrop(newSpan.GetType());
        for (uint32_t k = 0; k <= lineCount; k += 256u) {
            indices.push_back(((newSpan.lineStartIndex + k) & 0xffffffu) | (lineCount << 24u));
        }
    }
    
    EmitSpan(lastIndexBase, indices.size(), spanPosition.x, spanPosition.y, maxSpanX, backdrop);
}

void TraverseGrid2(uint32_t workStartIndex, uint32_t workEndIndex, const BitArray& flatVerbs, const std::vector<VPoint>& flatPoints, std::vector<Span>& spans) {
    VPoint p0 = flatPoints[workStartIndex];

    int32_t spanTileY = RoundDownToTile(p0.y);
    uint32_t spanEntryDirection = 0u;

    float spanMinX = p0.x;
    float spanMaxX = p0.x;

    uint32_t spanLineStartIndex = workStartIndex + 1u;
    uint32_t contourId = spans.size();
    
    auto EmitClose = [&spans](uint32_t i, float spanMinX, float spanMaxX, int32_t spanTileY, uint32_t spanLineStartIndex, uint32_t contourId, uint32_t spanEntryDirection) {
        Span span;
        span.key = PackPosition(std::floor(spanMinX), spanTileY);
        span.lineStartIndex = spanLineStartIndex;
        span.PackTypeLineEndIndex(0, i - 1);
        span.spanMaxX = std::floor(spanMaxX);
        if (spanEntryDirection != 0u) {
            spans[contourId].SetType((spans[contourId].GetType() != spanEntryDirection) ? 0 : spanEntryDirection);
        }
        spans.push_back(span);
    };

    
    for (uint32_t i = workStartIndex + 1u; i < workEndIndex; i++) {
        const VPoint& p1 = flatPoints[i];
        int32_t y1 = RoundDownToTile(p1.y);

        if (!flatVerbs.IsBitSet(i)) {
            if (spanTileY == y1) {
                spanMaxX = std::max(spanMaxX, p1.x);
                spanMinX = std::min(spanMinX, p1.x);
                p0 = p1;    
                continue;
            }
            
            uint32_t lineDirection = (p1.y > p0.y) ? DIRECTION_DOWN : DIRECTION_UP;
            int32_t step = (lineDirection == DIRECTION_DOWN) ? TILE_SIZE : -TILE_SIZE;
            int32_t offset = (lineDirection == DIRECTION_DOWN) ? TILE_SIZE : 0;

            float slope = (p1.x - p0.x) / (p1.y - p0.y);            
            float xv0 = p0.x;
            float xv1 = p0.x + (static_cast<float>(spanTileY + offset) - p0.y) * slope;

            spanMaxX = std::max(spanMaxX, xv1);
            spanMinX = std::min(spanMinX, xv1);

            // if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                uint32_t type = (lineDirection == spanEntryDirection || spanEntryDirection == 0u) ? lineDirection : 0;
                spans.push_back(Span(PackPosition(std::floor(spanMinX), spanTileY), std::floor(spanMaxX), spanLineStartIndex, Pack24And8(i, type)));
            // }
            
            spanEntryDirection = lineDirection;
            spanLineStartIndex = i;
            spanTileY += step;

            while (spanTileY != y1) {
                xv0 = xv1;
                // xv1 += slope * step;
                xv1 = p0.x + (static_cast<float>(spanTileY + offset) - p0.y) * slope;
                spanMinX = std::min(xv0, xv1);
                spanMaxX = std::max(xv0, xv1);
                // if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                    spans.push_back(Span(PackPosition(std::floor(spanMinX), spanTileY), std::floor(spanMaxX), i, Pack24And8(i, lineDirection)));
                // }
                spanTileY += step;
            }

            spanMinX = std::min(xv1, p1.x);
            spanMaxX = std::max(xv1, p1.x);
        } else {
            EmitClose(i, spanMinX, spanMaxX, spanTileY, spanLineStartIndex, contourId, spanEntryDirection);
            spanLineStartIndex = i + 1;
            spanTileY = y1;
            spanEntryDirection = 0u;
            spanMaxX = p1.x;
            spanMinX = p1.x;
            contourId = spans.size();
        }

        p0 = p1;
    }

    EmitClose(workEndIndex, spanMinX, spanMaxX, spanTileY, spanLineStartIndex, contourId, spanEntryDirection);
}


std::vector<lyra::SVGUtil::Element> TestElements() {
    VPath p;
    p.MoveTo(100.0, 100.0);
    p.LineTo(200.0, 100.0);
    p.LineTo(200.0, 200.0);
    p.LineTo(100.0, 200.0);
    p.LineTo(100.0, 100.0);
    lyra::SVGUtil::Element e;
    e.path = p;
    return { e };
}

int main() {
    using std::chrono::milliseconds;
    std::ifstream t("paper-1.svg");
    
    if (t.fail()) {
        std::cerr << "Failed to find file" << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << t.rdbuf();
    auto* img = lyra::SVGUtil::ReadSVG(buffer.str(), "Label");

    const float transform[6] = {2.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    auto elements = lyra::SVGUtil::ParseSVG(img, transform);
    // auto elements = TestElements();

    std::chrono::high_resolution_clock::time_point h_start, h_end;
    std::vector<uint32_t> colors(elements.size());

    double avgTime = 0.0f;
    uint32_t iterations = 20000;

    std::vector<Span> spans;
    BitArray flatVerbs;

    std::vector<uint32_t> atlasIndices;
    std::vector<VPoint> flatPointsGlobal;
    std::vector<uint32_t> indicesGlobal;
    std::vector<DrawSpan> drawSpansGlobal;

    for (int j = 0; j < iterations; j++) {
        h_start = std::chrono::high_resolution_clock::now();
        flatPointsGlobal.clear();
        drawSpansGlobal.clear();
        indicesGlobal.clear();

        flatVerbs.clear();
        spans.clear();
        atlasIndices.clear();

        drawSpansGlobal.reserve(drawSpansAllocation);
        indicesGlobal.reserve(indicesAllocation);
        flatPointsGlobal.reserve(flatPointsAllocation);

        flatVerbs.reserve(flatPointsAllocation);
        spans.reserve(spansAllocation);
        atlasIndices.reserve(atlasIndicesAllocation);
    
        AtlasManager atlasManager(IMAGE_WIDTH, IMAGE_HEIGHT);
        uint32_t estimatedTileArea = 0u;        
        uint32_t lineBaseIndex = 0u;

        for (int i = 0; i < elements.size(); i++) {
            auto& el = elements[i];
            // const float transform[6] = {1.2, 0.0, 0.0, 1.2, 0.0, 0.0};
            // el.path.SetTransform(transform);
            // el.path.Retransform();

            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle); 
            uint32_t flatStartIndex = lineBaseIndex;
            lineBaseIndex = FlattenCommands2(verbs, points, flatVerbs, flatPointsGlobal, 0.01f, lineBaseIndex);
            uint32_t spanStartIndex = spans.size();
            TraverseGrid2(flatStartIndex, lineBaseIndex, flatVerbs, flatPointsGlobal, spans);
            std::sort(spans.begin() + spanStartIndex, spans.end(), [](const Span& s0, const Span& s1) {
                return s0.key < s1.key;
            });
            MergeSpans(spanStartIndex, spans, indicesGlobal, drawSpansGlobal, i, atlasManager, atlasIndices, flatPointsGlobal);
            colors[i] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
                    el.paint.GetFillColor().GetU8ABGR();

        }

        // uint32_t numDrawSpans = drawSpansGlobal.size();
        // uint32_t numIndices = indicesGlobal.size();        
        // renderer.Upload(colors, flatPointsGlobal, indicesGlobal, drawSpansGlobal, atlasIndices, lineBaseIndex, numIndices, numDrawSpans);
        // renderer.Render(atlasIndices.size(), numDrawSpans, lineBaseIndex);
        
        // std::cout << "Used space: " << atlasManager.UsedSpace() << " " << atlasManager.Allocation() << " " << atlasManager.position() << " estimated area: " << estimatedTileArea << " " << gaps * TILE_SIZE << std::endl;
        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
        avgTime += ms_double.count();
    }

    renderer.Dispose();
    std::cout << "avgTime: " << avgTime / iterations << std::endl;
    return 0;
}
