
#include "Validation.h"

// #pragma GCC diagnostic push 
// #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
// #pragma GCC diagnostic ignored "-Wdeprecated-declarations" 
// #pragma GCC diagnostic ignored "-Wmissing-field-initializers"
// #pragma GCC diagnostic ignored "-Wextra-semi-stmt"
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"
// #include "stb_image.h"
// #pragma GCC diagnostic pop


static std::array<uint32_t, IMAGE_WIDTH * IMAGE_HEIGHT> image = {};
static std::array<float, IMAGE_WIDTH * IMAGE_HEIGHT> atlas = {};
static std::array<uint8_t, IMAGE_WIDTH * IMAGE_HEIGHT> outAtlas = {};

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
// std::cout << "]" << std::endl;

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



bool LOG = false;
void ValidateSpanLine(const Span& span, const std::vector<FlatCommand>& flatLines, float y) {
    int32_t ty = ((span.key >> 16u) & 0xffffu);
    int32_t tx = (span.key & 0xffffu);

    if (LOG) {
        std::cout << y  << " tilex: " << (span.key & 0xffff) << "  maxx: " << span.spanMaxX << " Type: " << span.GetType() << std::endl;
    }
    int32_t mx = -1111;

    if (span.lineStartIndex > span.GetLineEndIndex()) {
        std::cerr << "Empty span" << std::endl;
        exit(1);
    }

    for (int i = span.lineStartIndex; i <= span.GetLineEndIndex(); i++) {
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
        std::cerr << "Faulty max span x total check: " << mx << " " << span.spanMaxX << " (" << span.lineStartIndex << " " << span.GetLineEndIndex() << ")" << std::endl;
        for(int i = span.lineStartIndex; i < span.GetLineEndIndex(); i++) {
            std::cout << flatLines[i].point << std::endl;
        }
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

void RenderToAtlas2(const std::vector<DrawSpan>& spans, const std::vector<uint32_t>& indices, const std::vector<VPoint>& flatLinePoints, const std::vector<uint32_t>& atlasIndices) {

    // for (auto& atlasIndex: atlasIndices) {
    //     uint32_t it = atlasIndex >> 24u;
    //     uint32_t spanIndex = atlasIndex & 0xffffffu;
    //     const auto& span = spans[spanIndex];
        
    //     uint32_t mx = span.pathId >> 16u;
        
    //     uint32_t tl_x = span.position & 0xffffu;
    //     uint32_t tl_y = span.position >> 16u;
        
    //     uint32_t atlasX = span.atlasPosition & 0xffffu;
    //     uint32_t atlasY = span.atlasPosition >> 16u;

    //     uint32_t width = mx - tl_x;


    //     uint32_t offset = (it + 1u) * linesPerQuad;
    //     uint32_t startIndex = span.lineStartIndex + offset;
    //     uint32_t endIndex = std::min(startIndex + linesPerQuad, span.lineEndIndex);

    //     for (int y = 0; y < TILE_SIZE; y++) {
    //         if (y + atlasY >= IMAGE_HEIGHT) {
    //             break;
    //         }
    //         for (int x = 0; x < width; x++) {
    //             if (x + atlasX >= IMAGE_WIDTH) {
    //                 break;
    //             }
    //             float area = 0.0f;
    //             uint32_t pixelIndex = (atlasY + y) * IMAGE_WIDTH + (atlasX + x);
    //             for (int i = startIndex; i < endIndex; i++) {
    //                 uint32_t index = indices[i];
    //                 const VPoint& p0 = flatLinePoints[index - 1u];
    //                 const VPoint& p1 = flatLinePoints[index];
    //                 area += Area(p0, p1, VPoint::Make(tl_x + x, tl_y + y));
    //             }

    //             atlas[pixelIndex] = area;
    //         }        
    //     }
    // }
}


void Render(uint32_t start, const std::vector<DrawSpan>& spans, const std::vector<uint32_t>& indices, const std::vector<VPoint>& flatLinePoints, const std::vector<uint32_t>& colors) {
    
    for (int i = start; i < spans.size(); i++) {
        const DrawSpan& span = spans[i]; 
        uint32_t pathId = span.pathId & 0xffffu;
        float4 color = unpack(colors[pathId]);
        
        uint32_t mx = span.pathId >> 16u;
        uint32_t pid = span.pathId & 0xffffu;
        uint32_t tl_x = span.position & 0xffffu;
        uint32_t tl_y = span.position >> 16u;

        // uint32_t atl_tl_x = span.atlasPosition & 0xffffu;
        // uint32_t atl_tl_y = span.atlasPosition >> 16u;

        uint32_t width = mx - tl_x;

        // for (int cy = 0; cy < TILE_SIZE; cy++) {
        //     uint32_t y = tl_y + cy;
        //     if (y >= IMAGE_HEIGHT) {
        //         break;
        //     }
        //     for (int cx = 0; cx < width; cx++) {
        //         uint32_t x = tl_x + cx;
        //         if (x >= IMAGE_WIDTH) {
        //             break;
        //         }
                
        //         float area = 0.0f;
        //         uint32_t count = span.lineEndIndex - span.lineStartIndex;
        //         // if (count > linesPerQuad) {
        //         //     uint32_t atlasIndex = (atl_tl_y + cy) * IMAGE_WIDTH + atl_tl_x + cx;
        //         //     area += atlas[atlasIndex];
        //         // } 

        //         for (int i = 0; i < std::min(count, linesPerQuad); i++) {
        //             uint32_t lineIndex = indices[span.lineStartIndex + i];
        //             const VPoint& p0 = flatLinePoints[lineIndex - 1u];
        //             const VPoint& p1 = flatLinePoints[lineIndex];
        //             // Draw line          
        //             area += Area(p0, p1, VPoint::Make(x, y));
        //         }

        //         float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f); 
        //         uint32_t pixelIndex = y * IMAGE_WIDTH + x;
        //         float4 dst = unpack(image[pixelIndex]);
        //         float4 src = color * a;
        //         float4 res = dst * float4(1.0 - src.w) + src;
        //         image[pixelIndex] = pack(res);
        //     }        
        // }
    }
}

void WriteImages() {
    uint32_t channels = 4u;
    uint32_t bpr = IMAGE_WIDTH * channels;
    // stbi_write_png("image.png", IMAGE_WIDTH, IMAGE_HEIGHT, channels, static_cast<const void*>(image.data()), bpr);
    for (int i = 0; i < atlas.size(); i++) {
        float area = atlas[i];
        float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f);
        outAtlas[i] = static_cast<uint8_t>(a * 255.0f); 
    }
    channels = 1u;
    bpr = IMAGE_WIDTH * channels;
    // stbi_write_png("image_atlas.png", IMAGE_WIDTH, IMAGE_HEIGHT, channels, static_cast<const void*>(outAtlas.data()), bpr);       
}
