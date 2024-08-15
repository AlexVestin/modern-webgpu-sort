

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <list>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"

#include "defs.h"
#include "Validation.h"

#include "Flatten.h"
#include "path/SVGUtil.h"

#include "../wgpu/NativeUtils.h"
#include "../ComputeUtil.h"


static const wgpu::BufferUsage storageUsage = wgpu::BufferUsage::Storage;
static const wgpu::BufferUsage copyDstUsage = storageUsage | wgpu::BufferUsage::CopyDst;
static const wgpu::BufferUsage copySrcUsage = storageUsage | wgpu::BufferUsage::CopySrc;
static const wgpu::BufferUsage copyAllUsage = copySrcUsage | copyDstUsage;

static std::unique_ptr<wgpu::Instance> instance;


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
                std::cerr << "Allocating outside" << std::endl;
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
        // return atlasWidth - counterX + (atlasHeight - counterY) * atlasWidth; 

        return (counterY * atlasWidth + counterX * TILE_SIZE);
        // return t - allocation;
    }

    uint32_t counterX = 0;
    uint32_t counterY = 0;
    uint32_t allocation = 0u;

    uint2 lastEdgePosition;

    std::list<uint2> freeRows;
    const uint32_t atlasWidth;
    const uint32_t atlasHeight;
};


uint32_t over = 0u;
uint32_t totalSpanCount = 0u;

double totalSpanArea = 0.0f;
double totalLookupArea = 0.0f;

uint32_t ToTileIndex(float v) {
    return std::floor(v / static_cast<float>(TILE_SIZE));
} 


uint32_t MergeSpans(const std::vector<Span>& spans, const std::vector<FlatCommand>& flatLines, std::vector<uint32_t>& indices, std::vector<DrawSpan>& drawSpans, uint32_t pathId, AtlasManager& atlasManager) {

    auto EmitSpan = [&indices, &pathId, &drawSpans, &atlasManager, &spans](uint32_t from, uint32_t to, uint32_t x, uint32_t y, uint32_t maxX) {
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
        
        if (spanLineCount > linesPerSpan) {
            uint2 atlasPosition;
            atlasManager.Claim((maxX + 1u) - x, TILE_SIZE, atlasPosition);
            ds.atlasPosition = (atlasPosition.y << 16u) | atlasPosition.x;   
        } else if(spanLineCount == 0u) {
            const uint2& edgePosition = atlasManager.GetLastEdgePosition();
            ds.atlasPosition = (edgePosition.y << 16u) | edgePosition.x;
        }

        drawSpans.push_back(ds);
    };

    int32_t backdrop = 0;
    uint32_t spanId = 0u;

    const Span& span = spans[0];
    int32_t maxSpanX = span.spanMaxX;
    uint32_t currentSpanY = span.key >> 16u;
    uint32_t currentSpanX = (span.key & 0xffffu);
    uint32_t spanLineCount = (span.lineEndIndex - span.lineStartIndex) + 1u;
    
    for (int i = 1; i < spans.size(); i++) {
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
            //         if (spanLineCount + lineCount > linesPerSpan) {
            //             // Emit empty from end of last to start of new
            //             EmitSpan(i, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u);
            //         } else {
            //             EmitSpan(spanId, i, maxSpanX + 1u, currentSpanY, newSpanX - 1u);
            //         }  
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

            // Ignore horizontal lines
            if (std::abs(p0.y - p1.y) < 1.0e-6f) {
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


std::array<uint32_t, IMAGE_WIDTH * IMAGE_HEIGHT> image = {};
std::array<float, IMAGE_WIDTH * IMAGE_HEIGHT> atlas = {};
std::array<uint8_t, IMAGE_WIDTH * IMAGE_HEIGHT> outAtlas = {};

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
        AtlasManager atlasManager(IMAGE_WIDTH, IMAGE_HEIGHT);

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
            numSplits += MergeSpans(spans, flatLines, indices, drawSpans, i, atlasManager);
            RenderToAtlas(drawSpans, indices, flatLinePoints, atlas);
            Render(drawSpans, indices, flatLinePoints, image, colors[i], atlas);
            // break;

            numDrawSpans += drawSpans.size();
            numIndices += indices.size();
        }

        std::cout << "Used atlas space: " << atlasManager.UsedSpace() << std::endl;

        
        uint32_t channels = 4u;
        uint32_t bpr = IMAGE_WIDTH * channels;
        stbi_write_png("image.png", IMAGE_WIDTH, IMAGE_HEIGHT, channels, static_cast<const void*>(image.data()), bpr);
        
        for (int i = 0; i < atlas.size(); i++) {
            float area = atlas[i];
            float a = std::min(std::abs(area - 2.0f * std::round(0.5f * area)), 1.0f);
            outAtlas[i] = static_cast<uint8_t>(a * 255.0f); 
        }

        channels = 1u;
        bpr = IMAGE_WIDTH * channels;
        stbi_write_png("image_atlas.png", IMAGE_WIDTH, IMAGE_HEIGHT, channels, static_cast<const void*>(outAtlas.data()), bpr);
        

        
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
    }

    uint32_t total = numFlatLines * 16;

    std::cout << "Totalbytes: "<< total << std::endl; 
    std::cout << "Spans: " << numSpans << " lines: " << numFlatLines << " numDrawSpans: " << numDrawSpans << " numIndices: " << numIndices << std::endl;
    device.Destroy();
    return 0;
}
