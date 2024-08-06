
#include "Flatten.h"


std::vector<FlatCommand> FlattenCommands(const std::vector<VPathVerb>& verbs, const std::vector<VPoint>& points, float tolerance) {
    VPoint _last{0, 0};
    VPoint first{0, 0};
    std::vector<FlatCommand> output;

    size_t i = 0;
    const float TOO_LONG_LINE = 100000.f;

    for (auto& verb: verbs) {
        switch (verb) {
            case VPathVerb::kClose:
                output.push_back({VPathVerb::kLine, first.x, first.y});
                break;

            case VPathVerb::kMove:
                first = points[i];
                // intentional fall through
            case VPathVerb::kLine: {
                VPoint line = points[i];
                // if (!line.isCorrect()) {
                //     std::cerr << "Incorrect line: " << line << " " << i << std::endl;
                //     return {};
                // }
                output.push_back({verb, line.x, line.y});
                _last = line;
                i += 1;
                break;
            }
            case VPathVerb::kQuad: {
                VPoint control = points[i];
                VPoint point = points[i + 1];

                // if (!control.isCorrect() || !point.isCorrect()) {
                //     std::cerr << "Incorrect line or point" << std::endl;
                //     return {};
                // }
                // if ((point - _last).Length() > TOO_LONG_LINE || (control - _last).Length() > TOO_LONG_LINE) {
                //     std::cerr << "TOO LONG" << std::endl;
                //     return {};
                // }

                float l = (_last - 2.0f * control + point).Length();

                float dt = std::sqrt((4.0f * tolerance) / l);

                float t = 0.0;
                VPoint p01, p12, line, _lastLine = _last;
                while (t < 1.0) {
                    t = std::min(t + dt, 1.0f);
                    p01 = _last.Lerp(control, t);
                    p12 = control.Lerp(point, t);
                    line = p01.Lerp(p12, t);
                    output.push_back({VPathVerb::kLine, line.x, line.y});
                }
                _last = point;
                i += 2;
                break;
            }
            case VPathVerb::kCubic: {
                VPoint control1 = points[i];
                VPoint control2 = points[i + 1];
                VPoint point = points[i + 2];

                // if (!control1.isCorrect() || !control2.isCorrect() || !point.isCorrect()) {
                //     std::cerr << "Incorrect line or point" << std::endl;
                //     return {};
                // }

                VPoint a = -1.0f * _last + 3.0f * control1 - 3.0f * control2 + point;
                VPoint b = 3.0f * (_last - 2.0f * control1 + control2);
                float conc = std::max(b.Length(), (a + b).Length());

                float sqrt_of_8 = 2.82842712475f;

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

                    output.push_back({VPathVerb::kLine, line.x, line.y});
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