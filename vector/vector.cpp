

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <list>

#include "defs.h"
#include "Validation.h"
#include "Renderer.h"

#include "Flatten.h"
#include "path/SVGUtil.h"

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
    return static_cast<int32_t>(v * TILE_SIZE_DIV) * TILE_SIZE;
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
                lastEdgePosition = pos + uint2(w - 1u, 0u);
                it->x += w;
                if (spaceLeftInRow - w < 4) {
                    freeRows.erase(it);
                }
                return true;
            }
        }
        
        uint32_t spaceLeftInRow = atlasWidth - counterX; // + 1?
        if (spaceLeftInRow <= w) {
            if (spaceLeftInRow > 4)  {
                freeRows.push_back({counterX, counterY});
            }

            // Break into new row
            counterY += TILE_SIZE;
            if (counterY >= atlasHeight) {
                // Fail
                // std::cerr << "Allocating outside" << std::endl;
                return false;
            }

            counterX = 0u;  
        }

        pos = uint2(counterX, counterY);
        lastEdgePosition = pos + uint2(w - 1u, 0u);
        counterX += w;
        return true;
    }   

    const uint2& GetLastEdgePosition() const {
        return  lastEdgePosition;
    }

    uint32_t UsedSpace() const {
        return (counterY * atlasWidth + counterX * TILE_SIZE);
    }

    uint32_t counterX = 0;
    uint32_t counterY = 0;
    uint32_t allocation = 0u;

    uint2 lastEdgePosition;

    std::list<uint2> freeRows;
    const uint32_t atlasWidth;
    const uint32_t atlasHeight;
};

void MergeSpans(uint32_t spanStartId, const std::vector<Span>& spans, std::vector<uint32_t>& indices, std::vector<DrawSpan>& drawSpans, uint32_t pathId, AtlasManager& atlasManager, std::vector<uint32_t>& atlasIndices) {
    auto EmitSpan = [&](uint32_t from, uint32_t to, int32_t x, int32_t y, int32_t maxX) {
        if (maxX < 0 || y < 0 || y >= IMAGE_HEIGHT || x >= IMAGE_WIDTH) {
            return;
        }

        x = std::max(0, x);
        DrawSpan ds;
        ds.lineStartIndex = indices.size();
        
        for (int j = from; j < to; j++) {
            const auto& s = spans[j];
            uint32_t numLines = s.NumLines();            
            for (int k = 0; k < ComputeUtil::div_up(numLines, 256); k++) {
                uint32_t index = s.lineStartIndex + k * 256u;
                indices.push_back((index & 0xffffffu) | (numLines << 24u));
            }
            // for (int k = s.lineStartIndex; k <= s.GetLineEndIndex(); k++) {
            //         indices.push_back(k);
            // }
        } 
        ds.lineEndIndex = indices.size();
        ds.pathId = (static_cast<uint32_t>((maxX + 1u)) << 16u) | pathId;
        ds.position = PackPosition(x, y);

        uint32_t spanLineCount = ds.lineEndIndex - ds.lineStartIndex;
        
        if (spanLineCount > linesPerQuad) {
            uint2 atlasPosition;
            atlasManager.Claim((maxX + 1u) - x, TILE_SIZE, atlasPosition);
            ds.atlasPosition = PackPosition(atlasPosition.x, atlasPosition.y);   

            uint32_t lim = ComputeUtil::div_up(spanLineCount, linesPerQuad) - 1u;
            for (int i = 0; i < lim; i++) {
                atlasIndices.push_back(drawSpans.size() | (i << 24u));
            }
        } 
        // else if(spanLineCount == 0u) {
        //     const uint2& edgePosition = atlasManager.GetLastEdgePosition();
        //     ds.atlasPosition = (edgePosition.y << 16u) | edgePosition.x;
        // }

        drawSpans.push_back(ds);
    };

    int32_t backdrop = 0;
    uint32_t spanId = spanStartId;

    const Span& span = spans[spanId];
    int32_t maxSpanX = span.spanMaxX;

    int2 sp = UnpackPosition(span.key);
    int32_t currentSpanX = sp.x;
    int32_t currentSpanY = sp.y;

    uint32_t spanLineCount = span.NumLines();
    
    for (int i = spanStartId + 1u; i < spans.size(); i++) {
        const Span& newSpan = spans[i];

        int2 nsp = UnpackPosition(newSpan.key);
        int32_t newSpanY  = nsp.y;
        int32_t newSpanX  = nsp.x;

        // TODO: validate spanLineCount > 1 is correct
        bool canCommit = (newSpanX > maxSpanX) && (backdrop == 0) && spanLineCount > 1;
        bool isSplit = newSpanY == currentSpanY && canCommit;

        uint32_t lineCount = newSpan.NumLines();
        if ((newSpanY != currentSpanY) || canCommit) {
            EmitSpan(spanId, i, currentSpanX, currentSpanY, maxSpanX);
            // set new span
            currentSpanY = newSpanY;
            currentSpanX = newSpanX;
            maxSpanX = newSpan.spanMaxX;
            backdrop = 0;
            spanLineCount = lineCount;
            spanId = i;
        } else {
            // Merge
            
            // if (maxSpanX <newSpanX && currentSpanX !=newSpanX) {
            //     uint32_t gap =newSpanX - maxSpanX;
            //     if (gap > 1) {
            //         // Emit previous
            //         EmitSpan(spanId, i, currentSpanX, currentSpanY,  maxSpanX);
            //         EmitSpan(i, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u);
            //         // if (spanLineCount + lineCount > linesPerSpan) {
            //         //     // Emit empty from end of last to start of new
            //         //     EmitSpan(i, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u);
            //         // } else {
            //         //     EmitSpan(spanId, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u);
            //         // }  
            //         // update left, but not spanId so we get all the relevant lines for the future
            //         currentSpanX = newSpanX;
            //     }   
            // }
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
    
    EmitSpan(spanId, spans.size(), currentSpanX, currentSpanY, maxSpanX);
}

void TraverseGrid2(uint32_t workStartIndex, uint32_t workEndIndex, const std::vector<VPathVerb>& flatVerbs, const std::vector<VPoint>& flatPoints, std::vector<Span>& spans) {
    VPoint last = flatPoints[workStartIndex];

    int32_t spanTileY = RoundDownToTile(last.y);
    uint32_t spanEntryDirection = ~0u;

    float spanMinX = last.x;
    float spanMaxX = last.x;

    uint32_t spanLineStartIndex = workStartIndex + 1u;
    uint32_t contourId = spans.size();
    
    auto EmitClose = [&](uint32_t i) {
        Span span;
        span.key = PackPosition(spanMinX, spanTileY);
        span.lineStartIndex = spanLineStartIndex;
        span.PackTypeLineEndIndex(0, i - 1);
        span.spanMaxX = spanMaxX;

        // if we didn't exit the current span we dont need to update 
        // if (contourId < spans.size() && spans[contourId].GetType() == spanEntryDirection) {
        //     spans[contourId].SetType(spanEntryDirection);
        // }
        if (contourId < spans.size()) {
            uint32_t contourType = spans[contourId].GetType();  
            
            if ((contourType == 0 && (spanEntryDirection == 0u || spanEntryDirection == ~0u)) || (contourType != spanEntryDirection)) {
                spans[contourId].SetType(0);
            } else {
                spans[contourId].SetType(spanEntryDirection);
            }
        }
        
        spans.push_back(span);
    };


    for (int i = workStartIndex + 1; i < workEndIndex; i++) {
        const VPoint& p0 = last;
        const VPoint& p1 = flatPoints[i];
        
        if (flatVerbs[i] == VPathVerb::kLine) {
            // Horizontal lines
            if (p1.y >= spanTileY && p1.y < spanTileY + TILE_SIZE) {
                spanMaxX = std::max(spanMaxX, p1.x);
                spanMinX = std::min(spanMinX, p1.x);
                last = p1;
                continue;
            }
            
            bool downward = p1.y > p0.y;
            int32_t step = downward ? TILE_SIZE : -TILE_SIZE;
            int32_t offset = downward ? TILE_SIZE : 0;

            float slope = (p1.x - p0.x) / (p1.y - p0.y);
            float ymin = std::min(p0.y, p1.y);
            float ymax = std::max(p0.y, p1.y);

            float xv0 = p0.x;
            float xv1 = p0.x + (std::clamp(static_cast<float>(spanTileY + offset), ymin, ymax) - p0.y) * slope; 

            spanMaxX = std::max(spanMaxX, xv1);
            spanMinX = std::min(spanMinX, xv1);
            
            int32_t yc = spanTileY + step;       
            int32_t y1 = RoundDownToTile(p1.y);

            while (yc != y1 + step) {
                // Push span
                Span span;
                span.key = PackPosition(spanMinX, spanTileY);
                span.lineStartIndex = spanLineStartIndex;
                span.spanMaxX = spanMaxX;

                uint32_t type = downward ? DIRECTION_DOWN : DIRECTION_UP;
                if (type == spanEntryDirection || spanEntryDirection == ~0u) {
                    span.PackTypeLineEndIndex(type, i);
                } else {
                    span.PackTypeLineEndIndex(0, i);
                }
                spans.push_back(span);

                // Update counters
                spanLineStartIndex = i;                    
                spanEntryDirection = type;    
                spanTileY = yc;

                xv0 = xv1;
                xv1 = p0.x + (std::clamp(static_cast<float>(yc + offset), ymin, ymax) - p0.y) * slope; 

                spanMinX = std::min(xv0, xv1);
                spanMaxX = std::max(xv0, xv1);
                
                yc += step;
            }
        } else {
            EmitClose(i);

            spanLineStartIndex = i + 1;
            spanTileY = RoundDownToTile(p1.y);
            spanEntryDirection = ~0u;
            spanMaxX = p1.x;
            spanMinX = p1.x;
            contourId = spans.size();
        }

        last = p1;
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
    std::ifstream t("/Users/alexandervestin/prog/modern-webgpu-sort/boston.svg");
    
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
    uint32_t iterations = 10000;

    uint32_t flatPointsAllocation = 1 << 20;
    uint32_t spansAllocation = 1 << 19;
    uint32_t drawSpansAllocation = 1 << 19;
    uint32_t indicesAllocation = 1 << 19;
    uint32_t atlasIndicesAllocation = 1 << 19;

    // CPU Local 
    std::vector<VPathVerb> flatVerbs;
    std::vector<Span> spans;

    // GPU buffers
    std::vector<VPoint> flatPoints;
    std::vector<uint32_t> indices;
    std::vector<DrawSpan> drawSpans;
    std::vector<uint32_t> atlasIndices;

    for (int j = 0; j < iterations; j++) {
        flatPoints.clear();
        flatVerbs.clear();
        spans.clear();
        drawSpans.clear();
        indices.clear();
        atlasIndices.clear();

        if (flatPointsAllocation > flatPoints.capacity()) {
            flatPoints.reserve(flatPointsAllocation);
            flatVerbs.reserve(flatPointsAllocation);
        }
        
        if (spansAllocation > spans.capacity()) {
            spans.reserve(spansAllocation);
        }
        
        if (drawSpansAllocation > drawSpans.capacity()) {
            drawSpans.reserve(drawSpansAllocation);
        }

        if (indicesAllocation > indices.capacity()) {
            indices.reserve(indicesAllocation);
        }

        atlasIndices.reserve(atlasIndicesAllocation);
    
        AtlasManager atlasManager(IMAGE_WIDTH, IMAGE_HEIGHT);
        h_start = std::chrono::high_resolution_clock::now();
        
        uint32_t baseLineIndex = 0u;
        for (int i = 0; i < elements.size(); i++) {
            const auto& el = elements[i];
            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle);
            
            uint32_t flatStartIndex = baseLineIndex;
            baseLineIndex = FlattenCommands2(verbs, points, flatVerbs, flatPoints, 0.1f, baseLineIndex);
            uint32_t spanStartIndex = spans.size();
            TraverseGrid2(flatStartIndex, baseLineIndex, flatVerbs, flatPoints, spans);
            std::sort(spans.begin() + spanStartIndex, spans.end(), [](const Span& s0, const Span& s1) {
                return s0.key < s1.key;
            });
            uint32_t drawSpansStartIndex = drawSpans.size();
            MergeSpans(spanStartIndex, spans, indices, drawSpans, i, atlasManager, atlasIndices);                   
            colors[i] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
                    el.paint.GetFillColor().GetU8ABGR();
        }

        renderer.Upload(colors, flatPoints, indices, drawSpans, atlasIndices, baseLineIndex);
        renderer.Render(atlasIndices.size(), drawSpans.size());

        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
        avgTime += ms_double.count();
        // std::cout << flatPoints.size() << " " << spans.size() << std::endl;
        // std::cout << atlasManager.UsedSpace() << std::endl;
        // std::cout << flatPoints.size() * 8 << " b" << std::endl;
    }

    renderer.Dispose();

    std::cout << "avgTime: " << avgTime / iterations << std::endl;
    return 0;
}
