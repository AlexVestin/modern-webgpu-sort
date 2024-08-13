

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"

#include "Flatten.h"
#include "path/SVGUtil.h"

#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"



static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;


const uint32_t DIRECTION_UP  = 1u;
const uint32_t DIRECTION_DOWN = 2u;

static std::unique_ptr<wgpu::Instance> instance;

const uint32_t TILE_SIZE = 4u;
const uint32_t linesPerSpan = 12u;

const float TILE_SIZE_DIV = 1.0f / static_cast<float>(TILE_SIZE);
uint32_t under = 0u;
uint32_t over = 0u;
uint32_t totalSpanCount = 0u;

double totalSpanArea = 0.0f;
double totalLookupArea = 0.0f;

struct Span {
    uint32_t key; // y 16 bits, x 16 bits 
    uint32_t lineStartIndex; // 
    uint32_t lineEndIndex; // 
    int32_t spanMaxX; // in the range of [0, width] + left stuff
    uint32_t type;
};

struct DrawSpan {
    uint32_t position;
    uint32_t lineStartIndex;
    uint32_t lineEndIndex;
    uint32_t pathId;
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

    if (span.lineStartIndex > span.lineEndIndex) {
        std::cerr << "Empty span" << std::endl;
        exit(1);
    }

    for (int i = span.lineStartIndex; i <= span.lineEndIndex; i++) {
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
        for(int i = span.lineStartIndex; i < span.lineEndIndex; i++) {
            std::cout << flatLines[i].point << std::endl;
        }
        exit(1);
    }
}


const uint32_t IMAGE_WIDTH = 1920;
const uint32_t IMAGE_HEIGHT = 1080;

float4 unpack(uint32_t v) {
    uint32_t a = (v >> 24u) & 0xffu;
    uint32_t b = (v >> 16u) & 0xffu;
    uint32_t g = (v >> 8u) & 0xffu;
    uint32_t r = v & 0xffu;
    return float4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

float4 unpackRGBA(uint32_t v) {
    uint32_t r = (v >> 24u) & 0xffu;
    uint32_t g = (v >> 16u) & 0xffu;
    uint32_t b = (v >> 8u) & 0xffu;
    uint32_t a = v & 0xffu;
    return float4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

uint32_t pack(const float4& c) {
    uint8_t a = static_cast<uint8_t>(std::clamp(c.x, 0.0f, 1.0f) * 255.0f);
    uint8_t b = static_cast<uint8_t>(std::clamp(c.y, 0.0f, 1.0f) * 255.0f);
    uint8_t g = static_cast<uint8_t>(std::clamp(c.z, 0.0f, 1.0f) * 255.0f);
    uint8_t r = static_cast<uint8_t>(std::clamp(c.w, 0.0f, 1.0f) * 255.0f);
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

float Area(const VPoint& p0, const VPoint& p1, const VPoint& xy) {
    VPoint delta = p1 - p0;
    float y = p0.y - xy.y;
    float y0 = std::clamp(y, 0.0f, 1.0f);
    float y1 = std::clamp(y + delta.y, 0.0f, 1.0f);
    float dy = y0 - y1;

    if (dy != 0.0f) {
        float vec_y_recip = 1.0 / delta.y;

        float t0 = (y0 - y) * vec_y_recip;
        float t1 = (y1 - y) * vec_y_recip;

        float startx = p0.x - xy.x;
        float x0 = startx + t0 * delta.x;
        float x1 = startx + t1 * delta.x;
        float xmin0 = std::min(x0, x1);
        float xmax0 = std::max(x0, x1); 

        float xmin = std::min(xmin0, 1.0f) - 1.0e-3;
        float xmax = xmax0;
        float b = std::min(xmax, 1.0f);
        float c = std::max(b, 0.0f);
        float d = std::max(xmin, 0.0f);
        float a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
        return a * dy;
    }
    return 0.0f;
}


void Render(const std::vector<DrawSpan>& spans, const std::vector<uint32_t>& indices, const std::vector<VPoint>& flatLinePoints, std::array<uint32_t, IMAGE_WIDTH * IMAGE_HEIGHT>& image, uint32_t col) {
    float4 color = unpack(col);
    for (auto& span: spans) {
        uint32_t mx = span.pathId >> 16u;
        uint32_t pid = span.pathId & 0xffffu;

        uint32_t px = span.position & 0xffffu;
        uint32_t py = span.position >> 16u;

        for (int y = py; y < py + TILE_SIZE; y++) {
            if (y >= IMAGE_HEIGHT) {
                break;
            }
            assert(index > 0);
            for (int x = px; x < mx; x++) {
                if (x >= IMAGE_WIDTH) {
                    break;
                }
                float area = 0.0f;
                uint32_t pixelIndex = y * IMAGE_WIDTH + x;
                for (int i = span.lineStartIndex; i < span.lineEndIndex; i++) {
                    uint32_t index = indices[i];
                    const VPoint& p0 = flatLinePoints[index - 1u];
                    const VPoint& p1 = flatLinePoints[index];
                    // Draw line          
                    area += Area(p0, p1, VPoint::Make(x, y));
                }


                float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f); 
                float4 dst = unpack(image[pixelIndex]);
                float4 src = color * a;
                float4 res = dst * float4(1.0 - src.w) + src;
                image[pixelIndex] = pack(res);
            }        
        }
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


uint32_t MergeSpans(const std::vector<Span>& spans, const std::vector<FlatCommand>& flatLines, std::vector<uint32_t>& indices, std::vector<DrawSpan>& drawSpans, uint32_t pathId) {
    int32_t backdrop = 0;
    uint32_t spanId = 0u;
    Span span = spans[0];

    int32_t maxSpanX = span.spanMaxX;
    uint32_t currentSpanY = span.key >> 16u;
    uint32_t currentSpanX = (span.key & 0xffffu);
    uint32_t spanLineCount = (span.lineEndIndex - span.lineStartIndex) + 1u;

    uint32_t downers = 0u;

    // std::cout << "------------" << std::endl;
    
    for (int i = 1; i < spans.size(); i++) {
        const Span& newSpan = spans[i];
       
        uint32_t newSpanY  = newSpan.key >> 16u;
        uint32_t newSpanX  = (newSpan.key & 0xffffu);

        // TODO: validate spanLineCount > 1 is correct
        bool canCommit = (newSpanX > maxSpanX) && (backdrop  == 0) && spanLineCount > 1;
        bool isSplit = newSpanY == currentSpanY && canCommit;

        if ((newSpanY != currentSpanY) || canCommit) {
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

            // std::cout << currentSpanX << "x" << currentSpanY << " " << downers << " " << spanLineCount << " [";
            // for (int j = spanId; j < i; j++) {
            //     const auto& s = spans[j];

                
            //     for (int k = s.lineStartIndex; k <= s.lineEndIndex; k++) {
            //         std::cout << k << " ";
            //     }

            //     if (s.type == DIRECTION_DOWN) {
            //         std::cout << "v ";
            //     } else if(s.type == DIRECTION_UP) {
            //         std::cout << "^ ";
            //     } else {
            //         std::cout << "= ";
            //     }                
            // } 

            // std::cout << "]" << std::endl;


            downers = 0u;

            DrawSpan ds;
            ds.lineStartIndex = indices.size();
            for (int j = spanId; j < i; j++) {
                const auto& s = spans[j];
                for (int k = s.lineStartIndex; k <= s.lineEndIndex; k++) {
                    indices.push_back(k);
                }
            } 
            ds.lineEndIndex = indices.size();
            ds.pathId = (static_cast<uint32_t>((maxSpanX + 1u)) << 16u) | pathId;
            ds.position = (currentSpanY << 16u) | currentSpanX;
            drawSpans.push_back(ds);
            
            // set new span
            currentSpanY = newSpanY;
            currentSpanX = newSpanX;
            maxSpanX = newSpan.spanMaxX;
            
            backdrop = 0;
            if (newSpan.type == DIRECTION_UP) {
                backdrop++;
            } else if(newSpan.type == DIRECTION_DOWN) {
                backdrop--;
                downers++;
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
                downers++;
            }
        }
    }

    DrawSpan ds;
    ds.lineStartIndex = indices.size();
    for (int j = spanId; j < spans.size(); j++) {
        const auto& s = spans[j];
        for (int k = s.lineStartIndex; k <= s.lineEndIndex; k++) {
            indices.push_back(k);
        }
    } 
    ds.lineEndIndex = indices.size();
    ds.pathId = (static_cast<uint32_t>((maxSpanX + 1u)) << 16u) | pathId;
    ds.position = (currentSpanY << 16u) | currentSpanX;
    drawSpans.push_back(ds);
    return 0u;
}

std::vector<Span> TraverseGrid(const std::vector<FlatCommand>& flatLines, uint32_t& hits) {
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

            if (p0 == p1) {
                continue;
            }
      
            if (std::abs(p0.y - p1.y) < 1.0e-6f) {
                spanMaxX = std::max(std::max(p0.x, p1.x), spanMaxX);
                lastTileX = std::min(lastTileX, static_cast<uint32_t>(std::min(p0.x, p1.x)));
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


                // std::cout << "Line: " << i << " " << p0 << p1 << " " << xv0 << " " << xv1 << " " << min << std::endl;

                float px = std::min(xv0, xv1);
                if (i > 0 && lastTileY != yc) {
                    // Commit on entering new span
                    Span span;
                    span.key = (lastTileY  << 16u) | (lastTileX & 0xffffu);
                    span.lineStartIndex = startIndex;
                    span.lineEndIndex = i;
                    span.spanMaxX = spanMaxX;

                    hits += (i - startIndex) + 1u;

                    uint32_t type = p0.y > p1.y ? DIRECTION_UP : DIRECTION_DOWN; // 2 = down
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

                hits += (i - startIndex);

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
                contourId = spans.size();
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
    return { e };
}

int main() {
    dawnProcSetProcs(&dawn::native::GetProcs());

    std::vector<const char*> enableToggleNames = {"allow_unsafe_apis", "dump_shaders"};
    std::vector<const char*> disabledToggleNames = {};

    wgpu::DawnTogglesDescriptor toggles;
    toggles.enabledToggles = enableToggleNames.data();
    toggles.enabledToggleCount = enableToggleNames.size();
    toggles.disabledToggles = disabledToggleNames.data();
    toggles.disabledToggleCount = disabledToggleNames.size();

    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.nextInChain = &toggles;
    instanceDescriptor.features.timedWaitAnyEnable = true;
    instance = std::make_unique<wgpu::Instance>(wgpu::CreateInstance(&instanceDescriptor));

    if (instance == nullptr) {
        std::cerr << "Failed to create instance" << std::endl;
        exit(1);
    }

    wgpu::Adapter adapter = NativeUtils::SetupAdapter(instance);
    wgpu::Device device = NativeUtils::SetupDevice(instance, adapter);


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
    uint32_t numFlatLines = 0u;
    uint32_t numSpans  = 0u;
    uint32_t numSplits = 0u;
    uint32_t numDrawSpans = 0u;
    uint32_t numIndices = 0u;

    wgpu::Buffer pathInfoBuffer;
    wgpu::Buffer lineIndexBuffer;
    wgpu::Buffer flatLinePointBuffer;
    wgpu::Buffer drawSpansBuffer;
    wgpu::BindGroup drawBindGroup;

    std::chrono::high_resolution_clock::time_point h_start, h_end;
    for (int j = 0; j < 1; j++) {
        h_start = std::chrono::high_resolution_clock::now();
        std::vector<uint32_t> colors(elements.size());
        std::array<uint32_t, IMAGE_WIDTH * IMAGE_HEIGHT> image = {};

        for (int i = 0; i < elements.size(); i++) {
            std::vector<uint32_t> indices;
            std::vector<DrawSpan> drawSpans;
            std::vector<VPoint> flatLinePoints;

            indices.reserve(1 << 20);
            drawSpans.reserve(1 << 20);


            auto& el = elements[i];
            auto paintStyle = el.path.IsExpandedStroke() ? PaintStyle::kStroke : PaintStyle::kFill;
            const std::vector<VPoint>& points = el.path.GetPoints(paintStyle);
            const std::vector<VPathVerb>& verbs = el.path.GetVerbs(paintStyle);
            auto flatLines = FlattenCommands(verbs, points, 0.20f);

            // for(auto& fl: flatLines) {
            //     std::cout << fl.point << std::endl;
            // }
            
            flatLinePoints.reserve(flatLines.size());
            for (auto& fl: flatLines) {
                flatLinePoints.push_back(fl.point);
            }


         
            numFlatLines += flatLines.size();

            colors[i] = el.path.IsExpandedStroke() ? el.paint.GetStrokeColor().GetU8ABGR() : 
                    el.paint.GetFillColor().GetU8ABGR();
        

            uint32_t hits = 0u;
            std::vector<Span> spans = TraverseGrid(flatLines, hits);
            numSpans += spans.size();
            std::sort(spans.begin(), spans.end(), [](const Span& s0, const Span& s1) {
                return s0.key < s1.key;
            });

            // for(auto& s: spans) {
            //     std::cout << (s.key >> 16u) << " " << (s.key & 0xffffu) << " " << s.spanMaxX << std::endl;
            // }
            numSplits += MergeSpans(spans, flatLines, indices, drawSpans, i);
            Render(drawSpans, indices, flatLinePoints, image, colors[i]);
            // break;

            numDrawSpans += drawSpans.size();
            numIndices += indices.size();
        }

        
        uint32_t bpr = (IMAGE_WIDTH * sizeof(uint32_t));
        stbi_write_png("image.png", IMAGE_WIDTH, IMAGE_HEIGHT, 4, static_cast<const void*>(image.data()), bpr);
        // Save image to disk
        // pathInfoBuffer =
        //     utils::CreateBufferFromData(device, colors.data(), colors.size() * sizeof(uint32_t), copyDstUsage, "PathInformation");
        
        // lineIndexBuffer =
        //     utils::CreateBufferFromData(device, indices.data(), indices.size() * sizeof(uint32_t), copyDstUsage, "LineIndices");
        
        // flatLinePointBuffer = 
        //     utils::CreateBufferFromData(device, flatLinePoints.data(), flatLinePoints.size() * sizeof(VPoint), copyDstUsage, "flatLinePoints");;

        // drawSpansBuffer = 
        //     utils::CreateBufferFromData(device, drawSpans.data(), drawSpans.size() * sizeof(DrawSpan), copyDstUsage, "DrawSpans");;


        // // 
        // wgpu::BindGroupLayout drawLayout = utils::MakeBindGroupLayout(device, "DrawBindGroupLayout", {
        //     {0, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {1, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {2, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        //     {3, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment, wgpu::BufferBindingType::ReadOnlyStorage},
        // });

        // wgpu::BindGroup bindGroup = utils::MakeBindGroup(device, drawLayout, {
        //     {0, pathInfoBuffer},
        //     {1, lineIndexBuffer},
        //     {2, flatLinePointBuffer},
        //     {3, drawSpansBuffer},
        // });

        // wgpu::PipelineLayoutDescriptor descriptor;


        h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = h_end - h_start;
        std::cout << ms_double.count() << std::endl;
        

        // uint32_t lookupArea = 0u;
        // for (const auto& span: drawSpans) {
        //     uint32_t c = span.lineEndIndex - span.lineStartIndex;
        //     if (c <= linesPerSpan) {
        //         under++;
        //     } else {
        //         over++;
        //         uint32_t mx = span.pathId >> 16u;
        //         uint32_t pid = span.pathId & 0xffffu;
        //         uint32_t tx = span.position & 0xffffu;                
        //         lookupArea += (mx - tx)  * TILE_SIZE;
        //     }
        // }
        
        // std::cout << " LookupArea: " << lookupArea << std::endl;
    }

    uint32_t total = numFlatLines * 16;

    std::cout << "Totalbytes: "<< total << std::endl; 
    std::cout << "Spans: " << numSpans << " lines: " << numFlatLines << " numDrawSpans: " << numDrawSpans << " numIndices: " << numIndices << std::endl;
    std::cout << "Total: " << totalSpanCount << " over: " << over << " under: " << under << " area: " << (totalSpanArea / totalSpanCount) << " lua: " << static_cast<uint32_t>(totalLookupArea) / 10 << std::endl;


    device.Destroy();
    return 0;
}
