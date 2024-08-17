

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

uint32_t MergeSpans(uint32_t spanStartId, const std::vector<Span>& spans, std::vector<uint32_t>& indices, std::vector<DrawSpan>& drawSpans, uint32_t pathId, AtlasManager& atlasManager) {

    uint32_t over = 0u;
    auto EmitSpan = [&indices, &pathId, &drawSpans, &atlasManager, &spans, &over](uint32_t from, uint32_t to, uint32_t x, uint32_t y, uint32_t maxX) {
        DrawSpan ds;
        ds.lineStartIndex = indices.size();
        for (int j = from; j < to; j++) {
            const auto& s = spans[j];
            for (int k = s.lineStartIndex; k <= s.lineEndIndex; k++) {
                indices.push_back(k);
            }
        } 
        ds.lineEndIndex = indices.size();
        ds.pathId = (static_cast<uint32_t>((maxX + 1u)) << 16u) | pathId;
        ds.position = (y << 16u) | x;

        uint32_t spanLineCount = ds.lineEndIndex - ds.lineStartIndex;
        
        if (spanLineCount > linesPerQuad) {
            uint2 atlasPosition;
            atlasManager.Claim((maxX + 1u) - x, TILE_SIZE, atlasPosition);
            ds.atlasPosition = (atlasPosition.y << 16u) | atlasPosition.x;   
            over += ((spanLineCount + (linesPerQuad - 1u)) / linesPerQuad) * linesPerQuad;
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
    uint32_t currentSpanY = span.key >> 16u;
    uint32_t currentSpanX = (span.key & 0xffffu);
    uint32_t spanLineCount = (span.lineEndIndex - span.lineStartIndex) + 1u;
    
    for (int i = spanStartId + 1u; i < spans.size(); i++) {

        const Span& newSpan = spans[i];
        uint32_t newSpanY  = newSpan.key >> 16u;
        uint32_t newSpanX  = (newSpan.key & 0xffffu);

        // TODO: validate spanLineCount > 1 is correct
        bool canCommit = (newSpanX > maxSpanX) && (backdrop == 0) && spanLineCount > 1;
        bool isSplit = newSpanY == currentSpanY && canCommit;

        if ((newSpanY != currentSpanY) || canCommit) {
       

            EmitSpan(spanId, i, currentSpanX, currentSpanY, maxSpanX);
            // set new span
            currentSpanY = newSpanY;
            currentSpanX = newSpanX;
            maxSpanX = newSpan.spanMaxX;
            backdrop = 0;
            spanLineCount = (newSpan.lineEndIndex - newSpan.lineStartIndex) + 1u;
            spanId = i;
        } else {
            // Merge
            uint32_t lineCount = (newSpan.lineEndIndex - newSpan.lineStartIndex) + 1u;

            // if (maxSpanX < newSpanX && currentSpanX != newSpanX && backdrop != 0) {
            //     uint32_t gap = newSpanX - maxSpanX;
            //     if (gap > 32) {
            //         // Emit previous
            //         EmitSpan(spanId, i, currentSpanX, currentSpanY, maxSpanX);
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

        if (newSpan.type == DIRECTION_UP) {
            backdrop++;
        } else if(newSpan.type == DIRECTION_DOWN) {
            backdrop--;
        }
    }
    
    EmitSpan(spanId, spans.size(), currentSpanX, currentSpanY, maxSpanX);
    return over;
}

void TraverseGrid(uint32_t workStartIndex, const std::vector<VPathVerb>& flatVerbs, const std::vector<VPoint>& flatPoints, std::vector<Span>& spans) {
    VPoint last;
    
    // Counters
    uint32_t lastTileY = ~0u;
    uint32_t lastTileX = ~0u;
    uint32_t spanEntryDirection = ~0u;

    // Ids
    uint32_t startLineIndex = flatVerbs.size();
    uint32_t contourId = spans.size();
    
    float spanMaxX = 0u;

    uint32_t spanStartSize = spans.size();

    for (int i = workStartIndex; i <= flatVerbs.size(); i++) {
        if (i < flatVerbs.size() && flatVerbs[i] == VPathVerb::kLine) {
            const VPoint& p0 = last;
            const VPoint& p1 = flatPoints[i];

            if (p0 == p1) {
                continue;
            }

            if (std::abs(p0.y - p1.y) < 1.0e-6f) {
                // TODO: why cant we remove these
                spanMaxX = std::max(std::max(p0.x, p1.x), spanMaxX);
                lastTileX = std::min(lastTileX, static_cast<uint32_t>(std::min(p0.x, p1.x)));
                // -- 

                last = p1;
                continue;
            }
            
            int32_t y0 = static_cast<int32_t>(p0.y * TILE_SIZE_DIV) * TILE_SIZE;
            int32_t y1 = static_cast<int32_t>(p1.y * TILE_SIZE_DIV) * TILE_SIZE;
            int32_t dir = (y1 > y0) ? TILE_SIZE : -TILE_SIZE;

            float slope = (p1.x - p0.x) / (p1.y - p0.y);
            float miny = std::min(p0.y, p1.y);
            float maxy = std::max(p0.y, p1.y);

            for (int yc = y0; (dir < 0 && yc >= y1) || (dir > 0 && yc <= y1); yc += dir) {                
                float yv0 = std::clamp(static_cast<float>(yc), miny, maxy);
                float xv0 = p0.x + (yv0 - p0.y) * slope;

                float yv1 = std::clamp(static_cast<float>(yc + TILE_SIZE), miny, maxy);
                float xv1 = p0.x + (yv1 - p0.y) * slope;

                float px = std::min(xv0, xv1);
                if (i > workStartIndex && lastTileY != yc) {
                    // Commit on entering new span
                    Span span;
                    span.key = (lastTileY  << 16u) | (lastTileX & 0xffffu);
                    span.lineStartIndex = startLineIndex;
                    span.lineEndIndex = i;
                    span.spanMaxX = spanMaxX;

                    uint32_t type = p0.y > p1.y ? DIRECTION_UP : DIRECTION_DOWN; // 2 = down
                    if (type == spanEntryDirection || spanEntryDirection == ~0u) {
                        span.type = type;
                    } else {
                        span.type = 0;
                    }

                    spans.push_back(span);

                    // Update counters
                    lastTileX = -1000000;
                    startLineIndex = i;                    
                    spanEntryDirection = type;
                    spanMaxX = -10000;
                }


                spanMaxX = std::max(std::max(xv0, xv1), spanMaxX);
                lastTileX = std::min(lastTileX, static_cast<uint32_t>(px));
                lastTileY = yc;
            }
        } else {
            if (i > workStartIndex) {
                // Commit close
                Span span;
                span.key = (lastTileY << 16u) | (lastTileX & 0xffffu);
                span.lineStartIndex = startLineIndex;
                span.lineEndIndex = i - 1;
                span.spanMaxX = spanMaxX;
                span.type = 0;

                // if we didn't exit the current span we dont need to update 
                if (contourId < spans.size()) {
                    uint32_t contourType = spans[contourId].type;
                                        
                    if ((contourType == 0 && (spanEntryDirection == 0u || spanEntryDirection == ~0u)) || (contourType != spanEntryDirection)) {
                        spans[contourId].type = 0;
                    } else {
                        spans[contourId].type = spanEntryDirection;
                    }
                }
               
                spans.push_back(span);
            }

            // Update counters
            if (i < flatVerbs.size()) {
                startLineIndex = i + 1;

                const auto& p = flatPoints[i];
                lastTileY = static_cast<int32_t>(p.y * TILE_SIZE_DIV) * TILE_SIZE;
                spanEntryDirection = ~0u;
                
                spanMaxX = p.x;
                lastTileX = static_cast<uint32_t>(spanMaxX);
                
                // Next span is the start of the contour
                contourId = spans.size();
            } else {
                return;
            }
        }

        last = flatPoints[i];
    }
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
    std::ifstream t("ghost.svg");
    
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
    uint32_t iterations = 10000;


    uint32_t flatPointsAllocation = 1 << 20;
    uint32_t spansAllocation = 1 << 19;
    uint32_t drawSpansAllocation = 1 << 19;
    uint32_t indicesAllocation = 1 << 19;

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
    

        AtlasManager atlasManager(IMAGE_WIDTH, IMAGE_HEIGHT);

        h_start = std::chrono::high_resolution_clock::now();
        
        uint32_t over = 0;
        for (int i = 0; i < elements.size(); i++) {
            const auto& el = elements[i];
            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle);

            uint32_t flatStartIndex = flatVerbs.size();
            FlattenCommands2(verbs, points, flatVerbs, flatPoints, 0.1f);
 
            colors[i] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
                    el.paint.GetFillColor().GetU8ABGR();
            
            uint32_t spanStartIndex = spans.size();
            TraverseGrid(flatStartIndex, flatVerbs, flatPoints, spans);
            std::sort(spans.begin() + spanStartIndex, spans.end(), [](const Span& s0, const Span& s1) {
                return s0.key < s1.key;
            });

            uint32_t drawSpansStartIndex = drawSpans.size();
            over += MergeSpans(spanStartIndex, spans, indices, drawSpans, i, atlasManager);
        }

        // atlasIndices.reserve(over);
        // uint32_t index = 0;
        // for (int j = 0; j < drawSpans.size(); j++) {
        //     const auto& span = drawSpans[j];
        //     uint32_t lineCount = span.lineEndIndex - span.lineStartIndex;
        //     if (lineCount > linesPerQuad) {
        //         uint32_t lim = ComputeUtil::div_up(lineCount, linesPerQuad) - 1u;
        //         for (int i = 0; i < lim; i++) {
        //             atlasIndices.push_back((j | (i << 24u)));
        //         }
        //     }
        // }
        // RenderToAtlas2(drawSpans, indices, flatPoints, atlasIndices);
        // Render(0, drawSpans, indices, flatPoints, colors);
        // WriteImages();
        // renderer.Upload(colors, flatPoints, indices, drawSpans, atlasIndices);
        // renderer.Render(atlasIndices.size(), drawSpans.size());

        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
        avgTime += ms_double.count();
        std::cout << flatPoints.size() << " " << drawSpans.size() << " " << indices.size() << " " << atlasIndices.size() << std::endl;
    }

    renderer.Dispose();

    std::cout << "avgTime: " << avgTime / iterations << std::endl;
    return 0;
}
