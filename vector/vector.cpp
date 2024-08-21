

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
const uint32_t atlasIndicesAllocation = 1 << 18;

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

std::mutex doneMtx;
std::mutex mtx;
std::condition_variable cv_start, cv_done;
std::atomic<int> counter = 0;
uint32_t pointSum = 0u;
std::atomic<bool> killThreads = false;
std::atomic<int64_t> startIteration = -1;
std::atomic<int64_t> copyIteration = -1;

std::vector<std::atomic<uint32_t>> flatPointCounts(32);
std::vector<std::atomic<uint32_t>> drawSpansCounts(32);
std::vector<std::atomic<uint32_t>> lineIndicesCounts(32);

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
        
        // if (spanLineCount > linesPerQuad) {
        //     uint2 atlasPosition;
        //     atlasManager.Claim((maxX + 1u) - x, TILE_SIZE, atlasPosition);
        //     ds.atlasPosition = PackPosition(atlasPosition.x, atlasPosition.y);   

        //     uint32_t lim = ComputeUtil::div_up(spanLineCount, linesPerQuad) - 1u;
        //     for (int i = 0; i < lim; i++) {
        //         atlasIndices.push_back(drawSpans.size() | (i << 24u));
        //     }
        // } 
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

void TraverseGrid2(uint32_t workStartIndex, uint32_t workEndIndex, const BitArray& flatVerbs, const std::vector<VPoint>& flatPoints, std::vector<Span>& spans) {
    VPoint p0 = flatPoints[workStartIndex];

    int32_t spanTileY = RoundDownToTile(p0.y);
    uint32_t spanEntryDirection = ~0u;

    float spanMinX = p0.x;
    float spanMaxX = p0.x;

    uint32_t spanLineStartIndex = workStartIndex + 1u;
    uint32_t contourId = spans.size();
    
    auto EmitClose = [&](uint32_t i) {
        Span span;
        span.key = PackPosition(spanMinX, spanTileY);
        span.lineStartIndex = spanLineStartIndex;
        span.PackTypeLineEndIndex(0, i - 1);
        span.spanMaxX = spanMaxX;
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
            int32_t nextY = (lineDirection == DIRECTION_DOWN) ? spanTileY + TILE_SIZE : spanTileY;

            // Cant be zero because of the same tile check from earlier
            float slope = (p1.x - p0.x) / (p1.y - p0.y);            
            float xv0 = p0.x;
            float xv1 = p0.x + (static_cast<float>(nextY) - p0.y) * slope;

            spanMaxX = std::max(spanMaxX, xv1);
            spanMinX = std::min(spanMinX, xv1);

            uint32_t type = (lineDirection == spanEntryDirection || spanEntryDirection == ~0u) ? lineDirection : 0;
            if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                spans.push_back(Span(PackPosition(spanMinX, spanTileY), spanMaxX, spanLineStartIndex, Pack24And8(i, type)));
            }
            
            spanEntryDirection = lineDirection;
            spanLineStartIndex = i;
            spanTileY += step;

            while (spanTileY != y1) {
                xv0 = xv1;
                xv1 += slope * step;

                spanMinX = std::min(xv0, xv1);
                spanMaxX = std::max(xv0, xv1);

                if (spanTileY >= 0 && spanTileY < IMAGE_HEIGHT && spanMinX < IMAGE_WIDTH && spanMaxX >= 0.0f) {
                    spans.push_back(Span(PackPosition(spanMinX, spanTileY), spanMaxX, i, Pack24And8(i, lineDirection)));
                }
              
                spanTileY += step;
            }

            spanMinX = std::min(spanMinX, p1.x);
            spanMaxX = std::max(spanMaxX, p1.x);

        } else {
            EmitClose(i);
            spanLineStartIndex = i + 1;
            spanTileY = y1;
            spanEntryDirection = ~0u;
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

void ThreadRun(uint32_t tid, uint32_t n, const std::vector<lyra::SVGUtil::Element>& elements) {
    std::vector<VPoint> flatPoints;
    std::vector<uint32_t> indices;
    std::vector<DrawSpan> drawSpans;
    std::vector<uint32_t> atlasIndices;
    std::vector<Span> spans;
    BitArray flatVerbs;

    // TODO: retain
    flatPoints.reserve(1u << 18u);
    flatVerbs.reserve(1u << 18u);
    spans.reserve(1u << 14u);
    drawSpans.reserve(1u << 14u);
    indices.reserve(1u << 14u);
    // atlasIndices.reserve(1u << 1);

    AtlasManager atlasManager(IMAGE_WIDTH, IMAGE_HEIGHT);

    std::chrono::high_resolution_clock::time_point h_start, h_end;

    uint32_t myIteration = 0u;
    while (true) {
        // std::cout << "Waiting" << std::endl;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_start.wait(lk, [myIteration] { return myIteration == startIteration; });
        }
    
        if (killThreads) {
            break;
        }

        uint32_t slice = ComputeUtil::div_up(pointSum, n);
        flatPoints.clear();
        flatVerbs.clear();
        spans.clear();
        drawSpans.clear();
        indices.clear();
        atlasIndices.clear();


        h_start = std::chrono::high_resolution_clock::now();
        uint32_t lineBaseIndex = 0u;
        uint32_t numPoints = 0u;
        for (int i = 0; i < elements.size(); i++) {
            const auto& el = elements[i];
            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            if (numPoints < slice * tid) {
                numPoints += points.size();
                continue;
            }

            

            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle);
            numPoints += points.size();
            
            uint32_t flatStartIndex = lineBaseIndex;
            lineBaseIndex = FlattenCommands2(verbs, points, flatVerbs, flatPoints, 0.1f, lineBaseIndex);
            uint32_t spanStartIndex = spans.size();
            // TraverseGrid2(flatStartIndex, lineBaseIndex, flatVerbs, flatPoints, spans);
            std::sort(spans.begin() + spanStartIndex, spans.end(), [](const Span& s0, const Span& s1) {
                return s0.key < s1.key;
            });
            uint32_t drawSpansStartIndex = drawSpans.size();
            MergeSpans(spanStartIndex, spans, indices, drawSpans, i, atlasManager, atlasIndices);                   
            
            if (numPoints >= (tid + 1) * slice) {
                break;
            } 
        }

        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;

        for (int i = 0; i < n; i++) {
            if (tid <= i) {
                flatPointCounts[i].fetch_add(lineBaseIndex);
                drawSpansCounts[i].fetch_add(drawSpans.size());
                lineIndicesCounts[i].fetch_add(indices.size());
            }
        }

        // std::cout << "tid: " << tid << " " << ms_double.count() << " " << numPoints << std::endl;
        counter++;
        if (counter.load() == n) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_done.notify_one();
            }
        }

        {
            std::unique_lock<std::mutex> lk(mtx);
            cv_start.wait(lk, [myIteration] { return myIteration == copyIteration; });
        }

        // Copy time

        // if (tid == 0) {
        //     for (int i = 0; i < flatPointCounts[0]; i++) flatPointsGlobal[i] = flatPoints[i];
        //     for (int i = 0; i < lineIndicesCounts[0]; i++) indicesGlobal[i] = indices[i];
        //     for (int i = 0; i < drawSpansCounts[0]; i++) drawSpansGlobal[i] = drawSpans[i];
        // } else {
        //     for (int i = flatPointCounts[tid - 1]; i < flatPointCounts[tid]; i++) flatPointsGlobal[i] = flatPoints[i - flatPointCounts[tid - 1]];
        //     for (int i = lineIndicesCounts[tid - 1]; i < lineIndicesCounts[tid]; i++) indicesGlobal[i] = indices[i - lineIndicesCounts[tid - 1]];
        //     for (int i = drawSpansCounts[tid - 1]; i < drawSpansCounts[tid]; i++) drawSpansGlobal[i] = drawSpans[i - drawSpansCounts[tid - 1]];
        // }

        counter++;
        if (counter.load() == n) {
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_done.notify_one();
            }
        }
        
        myIteration++;
    }
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

    // CPU Local 
    std::vector<Span> spans;
    BitArray flatVerbs;
    std::vector<uint32_t> atlasIndices;


    std::vector<VPoint> flatPointsGlobal;
    std::vector<uint32_t> indicesGlobal;
    std::vector<DrawSpan> drawSpansGlobal;

    // std::vector<std::thread> threads(11);        
    // for (uint32_t i = 0; i < threads.size(); i++) {
    //     threads[i] = std::thread(&ThreadRun, i, threads.size(), std::cref(elements)); 
    // }


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
        

        // for (int i = 0; i < threads.size(); i++) {
        //     flatPointCounts[i] = 0u;
        //     drawSpansCounts[i] = 0u;
        //     lineIndicesCounts[i] = 0u;
        // }

        // pointSum = 0u;
        // uint32_t ii = 0u;
        // for (auto& el: elements) {
        //     auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
        //     pointSum += el.path.GetPoints(paintStyle).size();
        //     colors[ii++] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
        //             el.paint.GetFillColor().GetU8ABGR();
        // }
        // {
        //     std::lock_guard<std::mutex> lk(mtx);
        //     startIteration = j;
        //     counter = 0;  
        // }
        // cv_start.notify_all();
        // {
        //     std::unique_lock<std::mutex> lk(mtx);
        //     cv_done.wait(lk, [n = threads.size()] { return counter == n; });
        //     counter = 0;
        //     // Make sure we have space
        //     flatPointsGlobal.reserve(flatPointCounts[threads.size() - 1u]);
        //     indicesGlobal.reserve(lineIndicesCounts[threads.size() - 1u]);
        //     drawSpansGlobal.reserve(drawSpansCounts[threads.size() - 1u]);
        //     copyIteration = j;
        // }
        // // start copy
        // cv_start.notify_all();
        // {
        //     std::unique_lock<std::mutex> lk(mtx);
        //     cv_done.wait(lk, [n = threads.size()] { return counter == n; });
        //     // Copy should be done
        // }

        // uint32_t lineBaseIndex = flatPointCounts[threads.size() - 1u];
        // uint32_t numDrawSpans = drawSpansCounts[threads.size() - 1u];
        // uint32_t numIndices = lineIndicesCounts[threads.size() - 1u];
        
        uint32_t lineBaseIndex = 0u;
        for (int i = 0; i < elements.size(); i++) {
            const auto& el = elements[i];
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
            MergeSpans(spanStartIndex, spans, indicesGlobal, drawSpansGlobal, i, atlasManager, atlasIndices);                   
            colors[i] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
                    el.paint.GetFillColor().GetU8ABGR();
        }
        uint32_t numDrawSpans = drawSpansGlobal.size();
        uint32_t numIndices = indicesGlobal.size();        
        
        std::cout << lineBaseIndex << " " << numDrawSpans << " " << spans.size() << std::endl;
        renderer.Upload(colors, flatPointsGlobal, indicesGlobal, drawSpansGlobal, atlasIndices, lineBaseIndex, numIndices, numDrawSpans);
        renderer.Render(atlasIndices.size(), numDrawSpans);

        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
        avgTime += ms_double.count();


        // for (auto& drawSpan: drawSpansGlobal) {
        //     for (int i = drawSpan.lineStartIndex; i < drawSpan.lineEndIndex; i++) {
        //         uint32_t ind = indicesGlobal[i] & 0xffffffu;
        //         uint32_t cnt = indicesGlobal[i] >> 24u;
        //         std::cout << ind << " " << cnt << std::endl;
        //     }
        // }
    }

    // Notify all threads to start
    // {
    //     std::lock_guard<std::mutex> lk(mtx);
    //     killThreads = true;
    //     startIteration = iterations;
    // }
    // cv_start.notify_all();
    // for (uint32_t i = 0; i < threads.size(); i++) {
    //     threads[i].join();
    // }

    renderer.Dispose();

    std::cout << "avgTime: " << avgTime / iterations << std::endl;
    return 0;
}
