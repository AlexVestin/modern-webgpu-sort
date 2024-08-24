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

struct AtlasLine {
    uint32_t lineIndex;
    uint32_t atlasPosition;
};


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
    auto EmitSpan = [&](uint32_t from, uint32_t to, int32_t x, int32_t y, int32_t maxX, int32_t backdrop, uint32_t spanLineCount) {
        if (maxX < 0 || y < 0 || y >= IMAGE_HEIGHT || x >= IMAGE_WIDTH) {
            return;
        }
        x = std::max(0, x);
        DrawSpan ds;
        ds.pathId = (static_cast<uint32_t>((maxX + 1u)) << 16u) | pathId;
        ds.position = PackPosition(x, y);
        ds.lineStartIndex = indices.size();
        for (int j = from; j < to; j++) {
            const auto& s = spans[j];
            uint32_t numLines = s.NumLines();            
            for (int k = 0; k < ComputeUtil::div_up(numLines, 256); k++) {
                uint32_t index = s.lineStartIndex + k * 256u;
                indices.push_back((index & 0xffffffu) | (numLines << 24u));
            }
        }
        ds.lineEndIndex = Pack24And8(indices.size(),  static_cast<uint32_t>(lastBackdrop + 127));
        lastBackdrop = backdrop;

        // ds.lineIndices = uint2(~0u);
        // uint2 atlasPosition;
        // uint32_t atlasPositionPacked = ~0u;
        // if (spanLineCount > 3) {
        //     atlasManager.Claim((maxX + 1u) - x, TILE_SIZE, atlasPosition);
        //     atlasPositionPacked = PackPosition(atlasPosition.x, atlasPosition.y);
        // }
        // uint32_t bdPacked = static_cast<uint32_t>(lastBackdrop + 127);
        // lastBackdrop = backdrop;
        // if (spanLineCount == 0u) {
        //     ds.backdropLine0Packed = Pack24And8(0xffffffu, bdPacked);
        //     drawSpans.push_back(ds);
        //     return;
        // }
        // const Span& s0 = spans[from];
        // ds.backdropLine0Packed = Pack24And8(s0.lineStartIndex, bdPacked);
        // uint32_t firstOffset = 1u;
        // uint32_t lineIndex = 0u;
        // if (s0.NumLines() == 1u) {
        //     from++;
        // }
        // for (int j = from; j < to; j++) {
        //     const auto& s = spans[j];
        //     for (uint32_t k = s.lineStartIndex + firstOffset; k < s.GetLineEndIndex(); k++) {
        //         if (lineIndex < 2u) {
        //             ds.lineIndices[lineIndex++] = k;   
        //         } else {
        //             atlasIndices.push_back({k, atlasPositionPacked});
        //         }   
        //     }
        //     firstOffset = 0u;
        // } 

        drawSpans.push_back(ds);
    };

    
    uint32_t spanId = spanStartId;
    const Span& span = spans[spanId];
    int32_t maxSpanX = span.spanMaxX;
    uint32_t t = span.GetType();
    int32_t backdrop = 0;
    if (t == DIRECTION_UP) {
        backdrop++;
    } else if(t == DIRECTION_DOWN) {
        backdrop--;
    }

    int2 sp = UnpackPosition(span.key);
    int32_t currentSpanX = sp.x;
    int32_t currentSpanY = sp.y;

    uint32_t spanLineCount = span.NumLines();
    
    for (int i = spanStartId + 1u; i < spans.size(); i++) {
        const Span& newSpan = spans[i];

        int2 nsp = UnpackPosition(newSpan.key);
        int32_t newSpanY  = nsp.y;
        int32_t newSpanX  = nsp.x;

        bool canSplit = (newSpanX > maxSpanX) && (backdrop == 0) && spanLineCount > 1;
        uint32_t lineCount = newSpan.NumLines();
        if ((newSpanY != currentSpanY) || canSplit) {
           
            EmitSpan(spanId, i, currentSpanX, currentSpanY, maxSpanX, backdrop, spanLineCount);
            
            // set new span
            currentSpanY = newSpanY;
            currentSpanX = newSpanX;
            maxSpanX = newSpan.spanMaxX;
            backdrop = 0;
            lastBackdrop = 0;
            spanLineCount = lineCount;
            spanId = i;
        } else {
            if (maxSpanX < newSpanX && currentSpanX != newSpanX) {
                uint32_t gap = newSpanX - maxSpanX;
                if (gap > 2) {
                    // previous
                    EmitSpan(spanId, i, currentSpanX, currentSpanY, maxSpanX, backdrop, spanLineCount);
                    
                    // gap
                    EmitSpan(i, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u, backdrop, 0u);
                    
                    currentSpanX = newSpanX;
                    spanLineCount = 0u;
                    spanId = i;
                }   
            }
            spanLineCount += lineCount;
            maxSpanX = std::max(maxSpanX, newSpan.spanMaxX);
        }

        uint32_t t = newSpan.GetType();
        if (t == DIRECTION_UP) {
            backdrop++;
        } else if(t == DIRECTION_DOWN) {
            backdrop--;
        }
    }
    
    EmitSpan(spanId, spans.size(), currentSpanX, currentSpanY, maxSpanX, backdrop, spanLineCount);
}

void TraverseGrid2(uint32_t workStartIndex, uint32_t workEndIndex, const BitArray& flatVerbs, const std::vector<VPoint>& flatPoints, std::vector<Span>& spans) {
    VPoint p0 = flatPoints[workStartIndex];

    int32_t spanTileY = RoundDownToTile(p0.y);
    uint32_t spanEntryDirection = 0u;

    float spanMinX = p0.x;
    float spanMaxX = p0.x;

    uint32_t spanLineStartIndex = workStartIndex + 1u;
    uint32_t contourId = spans.size();
    
    auto EmitClose = [&spans, contourId, spanMinX, spanEntryDirection, spanTileY, spanLineStartIndex, spanMaxX](uint32_t i) {
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

    
    for (int i = workStartIndex + 1; i < workEndIndex; i++) {
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

            if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                uint32_t type = (lineDirection == spanEntryDirection || spanEntryDirection == 0u) ? lineDirection : 0;
                spans.push_back(Span(PackPosition(std::floor(spanMinX), spanTileY), std::floor(spanMaxX), spanLineStartIndex, Pack24And8(i, type)));
            }
            
            spanEntryDirection = lineDirection;
            spanLineStartIndex = i;
            spanTileY += step;

            while (spanTileY != y1) {
                xv0 = xv1;
                // xv1 += slope * step;
                xv1 = p0.x + (static_cast<float>(spanTileY + offset) - p0.y) * slope;
                spanMinX = std::min(xv0, xv1);
                spanMaxX = std::max(xv0, xv1);
                if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                    spans.push_back(Span(PackPosition(std::floor(spanMinX), spanTileY), std::floor(spanMaxX), i, Pack24And8(i, lineDirection)));
                }
                spanTileY += step;
            }

            spanMinX = std::min(xv1, p1.x);
            spanMaxX = std::max(xv1, p1.x);
        } else {
            EmitClose(i);
            spanLineStartIndex = i + 1;
            spanTileY = y1;
            spanEntryDirection = 0u;
            spanMaxX = p1.x;
            spanMinX = p1.x;
            contourId = spans.size();
        }

        p0 = p1;
    }

    EmitClose(workEndIndex);
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

    const float transform[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
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
            // const float transform[6] = {v, 0.0, 0.0, v, 0.0, 0.0};
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
        // renderer.Render(atlasIndices.size(), numDrawSpans);
        
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
