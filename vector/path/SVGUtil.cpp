
#define NANOSVG_IMPLEMENTATION

#include "SVGUtil.h"

#include <cassert>
#include <sstream>

namespace lyra::SVGUtil {

struct PathCommand {
    char type;
    std::vector<double> params;
};

std::vector<PathCommand> parseSVGPath(const std::string& dString) {
    std::vector<PathCommand> pathCommands;
    std::istringstream iss(dString);
    char command;

    while (iss >> command) {
        PathCommand pathCmd;
        pathCmd.type = command;

        while (iss.peek() == ' ' || iss.peek() == ',') {
            iss.ignore();
        }

        double param;
        while (iss >> param) {
            pathCmd.params.push_back(param);

            while (iss.peek() == ' ' || iss.peek() == ',') {
                iss.ignore();
            }

            if (iss.peek() == '-' || iss.peek() == '+' || std::isdigit(iss.peek())) {
                continue;
            } else {
                break;
            }
        }

        pathCommands.push_back(pathCmd);
    }

    return pathCommands;
}

std::string trimAndRemoveExtraWhitespaces(const std::string& input) {
    std::string result;
    bool prevIsWhitespace = false;

    for (char c : input) {
        if (std::isspace(c)) {
            if (!prevIsWhitespace) {
                result += ' ';  // Replace consecutive whitespaces with a single space
                prevIsWhitespace = true;
            }
        } else {
            result += c;
            prevIsWhitespace = false;
        }
    }

    // Remove leading and trailing whitespaces
    size_t firstNonSpace = result.find_first_not_of(' ');
    size_t lastNonSpace = result.find_last_not_of(' ');

    if (firstNonSpace != std::string::npos && lastNonSpace != std::string::npos) {
        result = result.substr(firstNonSpace, lastNonSpace - firstNonSpace + 1);
    } else {
        result.clear();  // The string is either empty or contains only whitespaces
    }

    return result;
}

void SVGStringToPath(const std::string& svgPath, VPath& path) {
    std::string trimmedContent = trimAndRemoveExtraWhitespaces(svgPath);
    std::vector<PathCommand> commands = parseSVGPath(trimmedContent);
    
    for (auto& c : commands) {
        switch (c.type) {
            case 'M': {
                assert(c.params.size() == 2);
                path.MoveTo(c.params[0], c.params[1]);
                break;
            };
            case 'c':
            case 'C': {
                assert(c.params.size() % 6 == 0);
                for (int i = 0; i < c.params.size(); i += 6) {
                    path.CubicTo(c.params[i + 0], c.params[i + 1], c.params[i + 2], c.params[i + 3], c.params[i + 4], c.params[i + 5]);
                }
                
                break;
            }
            case 'q':
            case 'Q': {
                assert(c.params.size() % 4 == 0);
                for (int i = 0; i < c.params.size(); i += 4) {
                    path.QuadTo(c.params[i + 0], c.params[i + 1], c.params[i + 2], c.params[i + 3]);
                }
                break;
            }
            case 'l':
            case 'L': {
                assert(c.params.size() % 2 == 0);
                for (int i = 0; i < c.params.size(); i += 2) {
                    path.LineTo(c.params[i + 0], c.params[i + 1]);
                }
                break;
            }
            case 'z':
            case 'Z': {
                assert(c.params.size() == 0);
                path.Close();
                break;
            }

            default:
                std::cerr << "Unknown command: " << c.type << std::endl;
                exit(1);
        }
    }
}

void PrintPathAsSVG(const VPath& path) {
    const std::vector<VPathVerb>& verbs = path.GetVerbs(PaintStyle::kFill);
    const std::vector<VPoint>& points = path.GetPoints(PaintStyle::kFill);
    std::stringstream ss;

    uint32_t index = 0;
    for (auto& v : verbs) {
        switch (v) {
            case VPathVerb::kClose:
                ss << " Z ";
                break;
            case VPathVerb::kMove:
                ss << " M " << points[index].x << " " << points[index].y;
                index += 1;
                break;
            case VPathVerb::kLine:
                ss << " L " << points[index].x << " " << points[index].y;
                index += 1;
                break;
            case VPathVerb::kQuad:
                ss << " Q " << points[index].x << " " << points[index].y << " " << points[index + 1].x << " "
                   << points[index + 1].y;
                index += 2;
                break;
            case VPathVerb::kCubic:
                ss << " C " << points[index].x << " " << points[index].y << " " << points[index + 1].x << " "
                   << points[index + 1].y << " " << points[index + 2].x << " " << points[index + 2].y;
                index += 3;
                break;
            default:
                std::cerr << "Unknown type in Svg pring...." << std::endl;
        }
    }

    std::cerr << ss.str() << std::endl;
}

uint32_t nsvgColorToLyra(uint32_t c) {
    uint8_t r = (c >> 0u) & 0xffu;
    uint8_t g = (c >> 8u) & 0xffu;
    uint8_t b = (c >> 16u) & 0xffu;
    uint8_t a = (c >> 24u) & 0xffu;

    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

std::vector<Element> ParseSVG(const NSVGimage* svgImage, const float* transform) {
    std::vector<Element> elements = {};

    uint32_t p_counter = 0;
    for (NSVGshape* shape = svgImage->shapes; shape != NULL; shape = shape->next) {
        VPaint fp;
        VPaint sp;

        bool isFill = shape->fill.type != NSVG_PAINT_NONE;
        bool isStroke = shape->stroke.type != NSVG_PAINT_NONE;

        if (isStroke) {
            VPath strokedPath;
            if (transform != nullptr) {
                strokedPath.SetTransform(transform);
            }
            
            for (NSVGpath* path = shape->paths; path != NULL; path = path->next) {
                NSVGPathToVPath(path, strokedPath);
            }

            uint32_t dashCount = static_cast<uint32_t>(shape->strokeDashCount);
            float dashOffset = shape->strokeDashOffset;

            if (dashCount > 0) {
                sp.SetStrokeDashArray({&shape->strokeDashArray[0], &shape->strokeDashArray[dashCount]});
                sp.SetStrokeDashOffset(dashOffset);
            }

            switch (shape->strokeLineJoin) {
                case NSVG_JOIN_MITER: {
                    sp.SetLineJoin(LineJoin::Miter);
                    break;
                }

                case NSVG_JOIN_ROUND: {
                    sp.SetLineJoin(LineJoin::Round);
                    break;
                }

                case NSVG_JOIN_BEVEL: {
                    sp.SetLineJoin(LineJoin::Bevel);
                    break;
                }
            }

            switch (shape->strokeLineCap) {
                case NSVG_CAP_BUTT: {
                    sp.SetLineCap(LineCap::Butt);
                    break;
                }

                case NSVG_CAP_ROUND: {
                    sp.SetLineCap(LineCap::Round);
                    break;
                }

                case NSVG_CAP_SQUARE: {
                    sp.SetLineCap(LineCap::Square);
                    break;
                }
            }

            switch (shape->fillRule) {
                case NSVG_FILLRULE_NONZERO: 
                    sp.SetFillRule(FillRule::NonZero);
                    break;
                case NSVG_FILLRULE_EVENODD:
                    sp.SetFillRule(FillRule::EvenOdd);
                    break;
                default:
                    std::cout << "Invalid fill rule: " << shape->fillRule << std::endl;
            }

            uint32_t c = shape->stroke.color;
            sp.SetStrokeColor(nsvgColorToLyra(c));
            sp.SetStrokeOpacity(shape->opacity);
            sp.SetStyle(PaintStyle::kStroke);
            sp.SetStrokeWidth(shape->strokeWidth);

            strokedPath.ExpandStroke(sp.GetKurboStrokeStyling());
            elements.push_back({strokedPath, sp});
        }

        if (isFill) {
            VPath filledPath;
            if (transform != nullptr) {
                filledPath.SetTransform(transform);
            }
            
            for (NSVGpath* path = shape->paths; path != NULL; path = path->next) {
                NSVGPathToVPath(path, filledPath);
            }

            uint32_t c = shape->fill.color;
            fp.SetFillColor(nsvgColorToLyra(c));
            fp.SetFillOpacity(shape->opacity);
            fp.SetStyle(PaintStyle::kFill);
            elements.push_back({filledPath, fp});
        }
    }

    return elements;
}

NSVGimage* ReadSVG(const std::string& svgContent, const std::string& label) {
    char* content = new char[svgContent.length() + 1];
    strcpy(content, svgContent.c_str());
    NSVGimage* image = nsvgParse(content, "px", 96);  // nsvgParseFromFile("assets/ghost.svg", "px", 96);
    delete[] content;

    if (image) {
        printf("Parsed svg file %s\n", label.c_str());
    } else {
        fprintf(stderr, "Failed to load svg\n");
        return nullptr;
    }

    return image;
}

void NSVGPathToVPath(const NSVGpath* path, VPath& pp) {
    uint32_t point_index = 0;
    for (int i = 0; i < path->ncmds; i++) {
        switch (path->cmds[i]) {
            case NSVG_CMD_QUAD: {
                const float* p = &path->pts[point_index];
                pp.QuadTo(p[0], p[1], p[2], p[3]);
                point_index += 4;
                break;
            }
            case NSVG_CMD_MOVE: {
                const float* p = &path->pts[point_index];
                pp.MoveTo(p[0], p[1]);
                point_index += 2;
                break;
            }
            case NSVG_CMD_LINE: {
                const float* p = &path->pts[point_index];
                pp.LineTo(p[0], p[1]);
                point_index += 2;
                break;
            }
            case NSVG_CMD_CUBIC: {
                const float* p = &path->pts[point_index];
                pp.CubicTo(p[0], p[1], p[2], p[3], p[4], p[5]);
                point_index += 6;
                break;
            }
            case NSVG_CMD_CLOSE: {
                pp.Close();
                break;
            }
            default: {
                printf("SVG: Unknown command: %d\n", path->cmds[i]);
                exit(1);
            }
        }
    }
}
}  // namespace lyra::SVGUtil
