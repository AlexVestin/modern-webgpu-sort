

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Flatten.h"
#include "path/SVGUtil.h"

#include <chrono>

const uint32_t TILE_SIZE = 4u;
const float TILE_SIZE_DIV = 1.0f / static_cast<float>(TILE_SIZE);
uint32_t under = 0u;
uint32_t over = 0u;
uint32_t totalSpanCount = 0u;
const uint32_t linesPerSpan = 8u;
double totalSpanArea = 0.0f;
double totalLookupArea = 0.0f;

struct Span {
    uint32_t key;
    uint32_t lineStartIndex;
    uint32_t lineEndIndex;
    int32_t spanMaxX;
    uint32_t type;
};


bool aabbContainsSegment(const VPoint& p1, const VPoint& p2, const VPoint& min, const VPoint& max) {
    // Completely outside.
    if ((p1.x < min.x && p2.x < min.x) || (p1.y < min.y && p2.y < min.y) ||
        (p1.x >= max.x && p2.x >= max.x) || (p1.y >= max.y && p2.y >= max.y)) {
        return false;
    }
    
    // TODO: handle in traversal?
    if (p1 == p2) {
        return true;
    }

    float m = (p2.y - p1.y) / (p2.x - p1.x);

    float y = m * (min.x - p1.x) + p1.y;
    if (y >= min.y && y < max.y) {
        return true;
    }

    y = m * (max.x - p1.x) + p1.y;
    if (y >= min.y && y < max.y) {
        return true;
    }

    float x = (min.y - p1.y) / m + p1.x;
    if (x >= min.x && x < max.x) {
        return true;
    }

    x = (max.y - p1.y) / m + p1.x;
    if (x >= min.x && x < max.x) {
        return true;
    }

    return false;
}

float FindYIntersection(const VPoint& p1, const VPoint& p2, float y) {
    return p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
}


uint32_t ToTileIndex(float v) {
    return std::floor(v / static_cast<float>(TILE_SIZE));
} 


bool LOG = false;
void ValidateSpanLine(const Span& span, const std::vector<FlatCommand>& flatLines, float y) {
    int32_t ty = ((span.key >> 16u) & 0xffffu);
    int32_t tx = (span.key & 0xffffu);

    if (LOG) {
        std::cout << y  << " tilex: " << (span.key & 0xffff) << "  maxx: " << span.spanMaxX << " Type: " << span.type << std::endl;
    }
    

    int32_t mx = -1111;

    if (span.lineStartIndex > span.lineEndIndex & 0xffffffu) {
        std::cerr << "Empty span" << std::endl;
        exit(1);
    }

    for (int i = span.lineStartIndex; i <= span.lineEndIndex & 0xffffffu; i++) {
        const VPoint& p0 = flatLines[i - 1].point;
        const VPoint& p1 = flatLines[i].point;

        if (LOG) {
            std::cout << (flatLines[i].verb == VPathVerb::kMove ? "m" : "l") << " " << p0 << p1 << " i: " << i << std::endl;
        }
        
        if ((p0.y >= y + TILE_SIZE && p1.y >= y + TILE_SIZE) || (p0.y < y && p1.y < y)) {
            std::cerr << "Faulty line: " << p0 << p1 << " y: " << y << std::endl;
            exit(1);
        }

        if (flatLines[i].verb == VPathVerb::kMove) {
            std::cerr << "MoveTo mixed into wrong place" << std::endl;
            exit(1);
        }


        float miny = std::min(p0.y, p1.y);
        float maxy = std::max(p0.y, p1.y);
        float minx = std::min(p0.x, p1.x);
        float maxx = std::max(p0.x, p1.x);

        float yv0 = std::clamp(static_cast<float>(ty), miny, maxy);
        float yv1 = std::clamp(static_cast<float>(ty + TILE_SIZE), miny, maxy);

        float xv0, xv1;
        if (std::abs(p0.y - p1.y) < 1.0e-6f) {
            xv0 = p0.x;
            xv1 = p1.x;
        } else {
            float dx = (p1.x - p0.x) / (p1.y - p0.y);
            xv0 = p0.x + (yv0 - p0.y) * dx;
            xv1 = p0.x + (yv1 - p0.y) * dx;
        }

        float pxmax = std::clamp(std::max(xv0, xv1), minx, maxx);
        float pxmin = std::clamp(std::min(xv0, xv1), minx, maxx);

        if (static_cast<int32_t>(pxmin) < tx) {
            std::cerr << "Line to the left of span" << std::endl;
            exit(1);
        }

        if (static_cast<int32_t>(pxmax) > span.spanMaxX) {
            std::cerr << "Faulty span max x: " << pxmax << " " << span.spanMaxX << " y: (" << yv0 << " " << yv1 << ") minx: " << minx << " " << maxx << p0 << p1 << " " << xv0 << " " << xv1 << std::endl;
            std::cerr << pxmin << " " << pxmax << " " << tx  << std::endl;
            exit(1);
        }

        mx = std::max(mx, static_cast<int32_t>(pxmax));

    }

    if (mx != span.spanMaxX) {
        std::cerr << "Faulty max span x total check: " << mx << " " << span.spanMaxX << " (" << span.lineStartIndex << " " << span.lineEndIndex << ")" << std::endl;
        exit(1);
    }
}


void TestAllLines(const std::vector<FlatCommand>& flatLines, const std::vector<uint32_t>& mergedSpanLines, const VPoint& tl, int32_t mx) {

    VPoint last;
    int i = 0;

    VPoint br = VPoint::Make(mx + 1, tl.y + TILE_SIZE);
    std::vector<uint32_t> hits;
    for (const auto& l: flatLines) {
        // std::cout << (l.verb == VPathVerb::kMove ? "m" : "l") << " " << last << l.point << " i: " << i << std::endl;
        if (l.verb == VPathVerb::kLine) {
           if (aabbContainsSegment(last, l.point, tl, br)) {
                hits.push_back(i);
                if (std::find(mergedSpanLines.begin(), mergedSpanLines.end(), i) == mergedSpanLines.end()) {
                    std::cerr << "Line that should be in span is not: " << last << l.point << " for rect: " << tl << br << " at index: " << i << std::endl;

                    for(auto& v: mergedSpanLines) {
                        std::cerr << "  " << v << std::endl;
                    }
                    exit(1);
                }
           }
        }

        if (l.point.x < 0.0f || l.point.y < 0.0f) {
            std::cerr << "Negative values" << std::endl;
            exit(1);
        }
        last = l.point;
        i++;
    }

    if (hits.size() != mergedSpanLines.size()) {
        std::cerr << "Extraneus lines in span" << std::endl;
        std::cerr << " hits: " << hits.size() << " " << mergedSpanLines.size() << std::endl;
        for (auto& i: mergedSpanLines) {
            if (std::find(hits.begin(), hits.end(), i) == hits.end()) {
                auto& p0 = flatLines[i - 1].point;
                auto& p1 = flatLines[i].point;
                std::cerr << "Line: " << i << " is extra " << p0 << p1 << " for " << tl << br << std::endl;
            }
        }

        exit(1);
    }
}


uint32_t MergeSpans(const std::vector<Span>& spans, const std::vector<FlatCommand>& flatLines) {
    int32_t backdrop = 0;
    uint32_t spanId = 0u;
    Span span = spans[0];

    int32_t maxSpanX = span.spanMaxX;
    uint32_t currentSpanY = span.key >> 16u;
    uint32_t currentSpanX = (span.key & 0xffffu);
    uint32_t spanLineCount = (span.lineEndIndex - span.lineStartIndex) + 1u;

    uint32_t spanCount = 0u;

    
    for (int i = 1; i < spans.size(); i++) {
        const Span& newSpan = spans[i];
       
        uint32_t newSpanY  = newSpan.key >> 16u;
        uint32_t newSpanX  = (newSpan.key & 0xffffu);

        // TODO: validate spanLineCount > 1 is correct
        bool canCommit = (newSpanX > maxSpanX) && ((backdrop % 2) == 0) && spanLineCount > 1;


        bool isSplit = newSpanY == currentSpanY && canCommit;

        if ((newSpanY != currentSpanY) || canCommit) {
            // Commit
            spanCount++;

            // uint32_t area =  ((maxSpanX + 1) - currentSpanX) * TILE_SIZE;
            // std::cout << "-ms: " << currentSpanY * TILE_SIZE << "x" << currentSpanX << "->" <<  maxSpanX << "(" << area  << ") " << spanLineCount  << std::endl;
            // std::vector<uint32_t> allLines;
            // for (int j = spanId; j < i; j++) {
            //     const auto& s = spans[j];
            //     ValidateSpanLine(s, flatLines, currentSpanY);
            //     for (int k = s.lineStartIndex; k <= s.lineEndIndex; k++) {
            //         allLines.push_back(k);
            //     }
            // }            
            // TestAllLines(flatLines, allLines, VPoint::Make(currentSpanX, currentSpanY), maxSpanX);    
            // if (spanLineCount <= 1) {
            //     std::cerr << "Faulty line count: " << spanLineCount << " " << currentSpanY << std::endl;
            //     exit(1);
            // }

            // if (spanLineCount >= linesPerSpan) {
            //     over++;
            //     totalLookupArea += ((maxSpanX + 1) - currentSpanX) * TILE_SIZE;
            // } else {
            //     under++;
            // }
            // totalSpanArea += area;
            // maxLineCount = std::max(spanLineCount, maxLineCount);
            
            // set new span
            currentSpanY = newSpanY;
            currentSpanX = newSpanX;
            maxSpanX = newSpan.spanMaxX;
            
            backdrop = 0;
            if (newSpan.type == 1) {
                backdrop++;
            } else if(newSpan.type == 2) {
                backdrop--;
            }

            spanLineCount = (newSpan.lineEndIndex - newSpan.lineStartIndex) + 1u;
            spanId = i;
        } else {
            // Merge
            spanLineCount += (newSpan.lineEndIndex - newSpan.lineStartIndex) + 1u;
            maxSpanX = std::max(maxSpanX, newSpan.spanMaxX);

            // std::cout << "-n: " << currentSpanY * TILE_SIZE << "x" << currentSpanX << " " << (newSpan.lineEndIndex - newSpan.lineStartIndex) + 1u << " " << span.spanMaxX  << std::endl;

            if (newSpan.type == 1) {
                backdrop++;
            } else if(newSpan.type == 2) {
                backdrop--;
            }
        }
    }

    // if (spanLineCount >= linesPerSpan) {
    //     over++;
    //     totalLookupArea += ((maxSpanX + 1) - currentSpanX) * TILE_SIZE;
    // } else {
    //     under++;
    // }
    // totalSpanArea += ((maxSpanX + 1) - currentSpanX) * TILE_SIZE;


    spanCount++;
    totalSpanCount += spanCount;
    return 0u;
}

std::vector<Span> TraverseGrid(const std::vector<FlatCommand>& flatLines) {
    std::vector<Span> spans;

    spans.reserve(flatLines.size() / 2);
    
    VPoint last;
    uint32_t lastTileY = ~0u;
    uint32_t lastTileX = ~0u;
    uint32_t startIndex = 0u;
    uint32_t spanEntryDirection = ~0u;
    uint32_t contourId = 0u;
    float spanMaxX = 0u;

    for (int i = 0; i <= flatLines.size(); i++) {
        // std::cout << "i: " << i << " flags: " << (i != flatLines.size()) << " " << static_cast<uint32_t>(flatLines[i].verb) << std::endl;
        if (i < flatLines.size() && flatLines[i].verb == VPathVerb::kLine) {
            const auto& line = flatLines[i];
            const VPoint& p0 = last;
            const VPoint& p1 = line.point;

            float miny = std::min(p0.y, p1.y);
            float maxy = std::max(p0.y, p1.y);
      
            int32_t y0 = static_cast<int32_t>(p0.y * TILE_SIZE_DIV) * TILE_SIZE;
            int32_t y1 = static_cast<int32_t>(p1.y * TILE_SIZE_DIV) * TILE_SIZE;
            int32_t dir = y1 > y0 ? TILE_SIZE : -TILE_SIZE;


            float slope = (p1.x - p0.x) / (p1.y - p0.y);
            float xv0, xv1;

            bool isHorizontal = std::abs(p0.y - p1.y) < 1.0e-6f;
            if (isHorizontal) {
                xv0 = p0.x;
                xv1 = p1.x;
            }

            for (int yc = y0; (dir < 0 && yc >= y1) || (dir > 0 && yc <= y1); yc += dir) {                
                if (!isHorizontal) {
                    float yv0 = std::clamp(static_cast<float>(yc), miny, maxy);
                    float yv1 = std::clamp(static_cast<float>(yc + TILE_SIZE), miny, maxy);
                    xv0 = p0.x + (yv0 - p0.y) * slope;
                    xv1 = p0.x + (yv1 - p0.y) * slope;
                }

                float px = std::min(xv0, xv1);
                if (i > 0 && lastTileY != yc) {
                    if (lastTileX == ~0u || lastTileY == ~0u) {
                        std::cerr << "LastTile not set: " << lastTileX << " " << lastTileY << " " << spanEntryDirection << std::endl;
                        exit(1);
                    }

                    // Commit on entering new span
                    Span span;
                    span.key = (lastTileY  << 16u) | (lastTileX & 0xffffu);
                    span.lineStartIndex = startIndex;
                    span.lineEndIndex = i;
                    span.spanMaxX = spanMaxX;

                    uint32_t type = p0.y > p1.y ? 1 : 2;
                    if (type == spanEntryDirection || spanEntryDirection == ~0u) {
                        span.type = type;
                    } else {
                        span.type = 0;
                    }

                    spans.push_back(span);

                    // Update counters
                    lastTileX = -1000000;
                    startIndex = i;                    
                    spanEntryDirection = type;
                    spanMaxX = -10000;
                }


                spanMaxX = std::max(std::max(xv0, xv1), spanMaxX);
                lastTileX = std::min(lastTileX, static_cast<uint32_t>(px));
                lastTileY = yc;
            }
        } else {
            if (i > 0) {
                // Commit close
                Span span;
                span.key = (lastTileY << 16u) | (lastTileX & 0xffffu);
                span.lineStartIndex = startIndex;
                span.lineEndIndex = i - 1;
                span.spanMaxX = spanMaxX;
                span.type = 0;

                // if we didn't exit the current span we dont need to update 
                if (contourId < spans.size()) {
                    assert(contourId < spans.size());
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
            if (i < flatLines.size()) {
                startIndex = i + 1;

                lastTileY = static_cast<int32_t>(flatLines[i].point.y * TILE_SIZE_DIV) * TILE_SIZE;
                spanEntryDirection = ~0u;
                
                spanMaxX = flatLines[i].point.x;
                lastTileX = static_cast<uint32_t>(spanMaxX);
                
                // Next span is the start of the contour
                contourId = spans.size() ;
            } else {
                return spans;
            }
        }

        last = flatLines[i].point;
    }

    return spans;
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
    return {e};
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
    auto elements = lyra::SVGUtil::ParseSVG(img, nullptr);

    // auto elements = TestElements();
    uint32_t numFlatLines = 0u;
    uint32_t numSpans  = 0u;
    uint32_t numSplits = 0u;
    std::chrono::high_resolution_clock::time_point h_start, h_end;
    for (int j= 0 ; j < 1000; j++) {    
        
        h_start = std::chrono::high_resolution_clock::now();

   
        for (int i = 0; i < elements.size(); i++) {
            auto& el = elements[i];
            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle);
            auto flatLines = FlattenCommands(verbs, points, 0.2f);
            numFlatLines += flatLines.size();

            std::vector<Span> spans = TraverseGrid(flatLines);
            numSpans += spans.size();
            // std::sort(spans.begin(), spans.end(), [](const Span& s0, const Span& s1) {
            //     return s0.key < s1.key;
            // });
            // numSplits += MergeSpans(spans, flatLines);
        }

        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
    }

    std::cout << "Spans: " << numSpans << " lines: " << numFlatLines << " numSplits: " << numSplits << std::endl;
    std::cout << "Total: " << totalSpanCount << " over: " << over << " under: " << under << " area: " << (totalSpanArea / totalSpanCount) << " lua: " << static_cast<uint32_t>(totalLookupArea) / 10 << std::endl;
    return 0;
}
