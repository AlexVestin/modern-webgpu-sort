#include "LTRBRect.h"

namespace lyra {
std::ostream& operator<<(std::ostream& o, const LTRBRect& a) {
    o << "(" << a.l << "," << a.t << "," << a.r << "," << a.b << ")";
    return o;
}

std::ostream& operator<<(std::ostream& o, const Rect& a) {
    o << "(" << a.p0.x << "," << a.p0.y << ") (" << a.p1.x << "," << a.p1.y << ")";
    o << " (" << a.p2.x << "," << a.p2.y << ") (" << a.p3.x << "," << a.p3.y << ")";
    return o;
}
}  // namespace lyra
