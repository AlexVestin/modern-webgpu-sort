
#include "Flatten.h"


std::vector<FlatCommand> FlattenCommands(const std::vector<VPathVerb>& verbs, const std::vector<VPoint>& points, float tolerance) {
    VPoint _last{0, 0};
    VPoint first{0, 0};
    std::vector<FlatCommand> output{};

    size_t i = 0;
    const float TOO_LONG_LINE = 100000.f;
    const float tolerance4  = tolerance * 4.0f;
    const float sqrt_of_8 = 2.82842712475f;
    
    output.reserve(points.size() * 4);
    for (auto& verb: verbs) {
        switch (verb) {
            case VPathVerb::kClose:
                output.push_back({first.x, first.y, VPathVerb::kLine});
                break;

            case VPathVerb::kMove:
                assert(i < points.size());
                first = points[i];
                // intentional fall through
                [[fallthrough]];
            case VPathVerb::kLine:  {
                
                assert(i < points.size());
                const VPoint& line = points[i];
    
                output.push_back({line.x, line.y, verb});
                _last = line;
                i += 1;
                break;
            }
            case VPathVerb::kQuad: {
                const VPoint& control = points[i];
                const VPoint& point = points[i + 1];

                float l = (_last - 2.0f * control + point).Length();
                float dt = std::sqrt(tolerance4 / l);

                float t = 0.0;
                VPoint p01, p12, line, _lastLine = _last;
                while (t < 1.0) {
                    t = std::min(t + dt, 1.0f);
                    p01 = _last.Lerp(control, t);
                    p12 = control.Lerp(point, t);
                    line = p01.Lerp(p12, t);
                    output.push_back({line.x, line.y, VPathVerb::kLine});
                }
                _last = point;
                i += 2;
                break;
            }
            case VPathVerb::kCubic: {
                const VPoint& control1 = points[i];
                const VPoint& control2 = points[i + 1];
                const VPoint& point = points[i + 2];

                // if (!control1.isCorrect() || !control2.isCorrect() || !point.isCorrect()) {
                //     std::cerr << "Incorrect line or point" << std::endl;
                //     return {};
                // }

                VPoint a = -1.0f * _last + 3.0f * control1 - 3.0f * control2 + point;
                VPoint b = 3.0f * (_last - 2.0f * control1 + control2);
                float conc = std::max(b.Length(), (a + b).Length());
                float dt = std::sqrt((sqrt_of_8 * tolerance) / conc);
                int cnt = std::ceil(1.0f / dt);
                dt = 1.0f / static_cast<float>(cnt);

                float t = 0.0;
                VPoint _lastLine = _last;
                for (int j = 0; j < cnt; j++) {
                    t = std::min(t + dt, 1.0f);
                    VPoint p01 = _last.Lerp(control1, t);
                    VPoint p12 = control1.Lerp(control2, t);
                    VPoint p23 = control2.Lerp(point, t);
                    VPoint p012 = p01.Lerp(p12, t);
                    VPoint p123 = p12.Lerp(p23, t);
                    VPoint line = p012.Lerp(p123, t);

                    output.push_back({line.x, line.y, VPathVerb::kLine});
                    _lastLine = line;
                }

                _last = point;
                i += 3;
                break;
            }

            default: {
                std::cerr << "Verb was: " << static_cast<uint32_t>(verbs[i]) << " At position: " << i << " buffer size: " << points.size()
                          << std::endl;
                exit(1);
            }
        }
    }

    return output;
}
