const HALF_PI = 1.5707963268;
const PI = 3.14159265359;
const TWO_PI = 6.28318530718;
const TWO_PI_QUANT = (1.0 / TWO_PI) * 255.0;
const TILE_SIZE = 16.0;
const LINES_PER_QUAD = 4u;

const view_step = vec2f(1.0) / vec2f(1920.0, 1080.0);
const sqrt_of_half = 0.7071067811865475;
const sqrt_of_half_256 = 181.01933598375615;
const num_verts = 6u;

struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(linear) dist0: vec4f, // distance from corner to line interpolated per line
    @location(1) @interpolate(linear) dist1: vec4f,
    @location(2) @interpolate(flat) heights0: vec2u, // 8 bits per height point, 2 points per line -> 16 bits per line
    @location(3) @interpolate(flat) heights1: vec2u, 
    @location(4) @interpolate(flat) angles: vec2u, // 8 bits per angle
    @location(5) @interpolate(flat) info: vec2u, // pathId 20 bytes, height baseline 12 bytes, 32 bits for buffer index
};

struct DrawSpan {
    position: u32,
    line_start_index: u32,
    line_end_index: u32,
    path_id: u32,
    atlas_position: u32,
    padding: u32,
};

fn signed_distance(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let dir = b - a;
    let perp = vec2(dir.y, -dir.x);
    let dir_to_p1 = b - p;
    return dot(normalize(perp), dir_to_p1);
}

fn unpack_position(v: u32) -> vec2f {
  let x = f32(v & 0xffffu) - 32767.0;
  return vec2(x, f32(v >> 16u));
}

fn area(p0: vec2f, p1: vec2f, xy: vec2f) -> f32 {
    let delta = p1 - p0;
    let y = p0.y - xy.y;
    let y0 = clamp(y, 0.0, 1.0);
    let y1 = clamp(y + delta.y, 0.0, 1.0);
    let dy = y0 - y1;

    if dy != 0.0 {
        let vec_y_recip = 1.0 / delta.y;

        let t0 = (y0 - y) * vec_y_recip;
        let t1 = (y1 - y) * vec_y_recip;

        // 
        let startx = p0.x - xy.x;
        let x0 = startx + t0 * delta.x;
        let x1 = startx + t1 * delta.x;
        let xmin0 = min(x0, x1);
        let xmax0 = max(x0, x1); 

        let xmin = min(xmin0, 1.0) - 1.0e-3;
        let xmax = xmax0;
        let b = min(xmax, 1.0);
        let c = max(b, 0.0);
        let d = max(xmin, 0.0);
        let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
        return a * dy;
    }
    return 0.0;
}

@group(0) @binding(0) var<storage, read> colors: array<u32>;
@group(0) @binding(1) var<storage, read> line_indices: array<u32>;
@group(0) @binding(2) var<storage, read> points: array<vec2f>;
@group(0) @binding(3) var<storage, read> draw_spans: array<DrawSpan>;
@group(0) @binding(4) var<storage, read> atlas_indices: array<u32>;


@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VSOutput {
    var out: VSOutput;
    let quad_id = VertexIndex / num_verts;
    
    let span_info = atlas_indices[quad_id];
    let span_index = span_info & 0xffffffu;

    let draw_span = draw_spans[span_index];
    let tl = unpack_position(draw_span.position);
    let min_x  = tl.x;
    let min_y  = tl.y;
    let max_x  = f32(draw_span.path_id >> 16u);
    let max_y  = min_y + TILE_SIZE;

    let width  = max_x - min_x;


    let vertex_id = VertexIndex % 6u; 
    var pos = array(
        vec2(min_x, min_y), // tr0 tl
        vec2(max_x, min_y), // tr0 tr
        vec2(max_x, max_y), // tr0 br
        vec2(min_x, min_y), // tr1 tl
        vec2(max_x, max_y), // tr1 br
        vec2(min_x, max_y), // tr1 bl
    );

    var v_pos = pos[vertex_id];

    // -----Read lines -----
    let offset = ((span_info >> 24u) + 1u) * LINES_PER_QUAD;     

    // Loop 1
    // for (var i = 0u; i < 4u; i++) {
    //     let index = line_indices[start_index + i];
    //     let p0 = points[index - 1u];
    //     let p1 = points[index];
    //     // Dists
    //     out.dist0[i] = signed_distance(v_pos, p0, p1);
    //     // Angles
    //     let line = p1 - p0;
    //     let a = u32((atan2(line.y, line.x) + PI) * TWO_PI_QUANT);
    //     out.angles[0u] |= (a << (i * 8u));
    //     // Heights // TODO 
    //     let h = 0u;
    //     let line_min_y = clamp(min(p0.y, p1.y), min_y, max_y) - min_y;
    //     let line_max_y = clamp(max(p0.y, p1.y), min_y, max_y) - min_y;
    //     // 
    //     out.heights0[i >> 1u] = (h << ((i & 1u) * 16u));
    // }
    // // Loop 1
    // start_index += 4u;
    // for (var i = 0u; i < 4u; i++) {
    //     let index = line_indices[start_index + i];
    //     let p0 = points[index - 1u];
    //     let p1 = points[index];
    //     // Dists
    //     out.dist1[i] = signed_distance(v_pos, p0, p1);
    //     // Angles
    //     let line = p1 - p0;
    //     let a = u32((atan2(line.y, line.x) + PI) * TWO_PI_QUANT);
    //     out.angles[1u] |= (a << (i * 8u));
    //     // Heights // TODO 
    //     let h = 0u; 
    //     let line_min_y = clamp(min(p0.y, p1.y), min_y, max_y) - min_y;
    //     let line_max_y = clamp(max(p0.y, p1.y), min_y, max_y) - min_y;
    //     // 
    //     out.heights1[i >> 1u] = (h << ((i & 1u) * 16u));
    // }

    let atl_tl = unpack_position(draw_span.atlas_position);
    let atl_min_x  = atl_tl.x;
    let atl_min_y  = atl_tl.y;
    let atl_max_x  = atl_min_x + width;
    let atl_max_y  = atl_min_y + TILE_SIZE;

    var atlas_pos = array(
        vec2(atl_min_x, atl_min_y), // tr0 tl
        vec2(atl_max_x, atl_min_y), // tr0 tr
        vec2(atl_max_x, atl_max_y), // tr0 br
        vec2(atl_min_x, atl_min_y), // tr1 tl
        vec2(atl_max_x, atl_max_y), // tr1 br
        vec2(atl_min_x, atl_max_y), // tr1 bl
    );


    var count = 0u;
    var index = 0u;
    var start_index = ~0u;
    var number_lines = ~0u;
    var first_span_offset = 0u;
    // find start
    let num_spans = draw_span.line_end_index - draw_span.line_start_index; // + 1?
    while (index < num_spans) {
        let index_data = line_indices[draw_span.line_start_index + index];
        count += (index_data >> 24u);
        
        if (count >= offset && start_index == ~0u) {
            start_index = draw_span.line_start_index + index;
        }

        if (count >= offset + LINES_PER_QUAD) {
            count = LINES_PER_QUAD;
            break;
        }
        index++;
    }

    if (number_lines == ~0u) {
        number_lines = count - offset;
    }


    out.info.x = start_index;
    out.info.y = (first_span_offset << 16u) | number_lines;
    out.dist0 = vec4f(v_pos, out.dist0.zw);

    var p = atlas_pos[vertex_id] * view_step * 2.0 - 1.0;    
    out.position = vec4<f32>(p.x, -p.y, 0.0, 1.0);
    return out;
}


@fragment
fn frag_main(
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(linear) dist0: vec4f, // distance from corner to line interpolated per line
    @location(1) @interpolate(linear) dist1: vec4f,
    @location(2) @interpolate(flat) heights0: vec2u, // 8 bits per height point, 2 points per line -> 16 bits per line
    @location(3) @interpolate(flat) heights1: vec2u, 
    @location(4) @interpolate(flat) angles: vec2u, // 8 bits per angle
    @location(5) @interpolate(flat) info: vec2u, // pa
    ) -> @location(0) vec4<f32> {
    var a = 0.0;
    let xy = dist0.xy - vec2f(0.5);

    let start_span_index = info.x;
    var first_span_offset = info.y >> 16u;
    let number_lines = info.y & 0xffffu;

    var count = 0u;
    var i = 0u;

    while (count < LINES_PER_QUAD) {
        let index_data = line_indices[start_span_index + i];
        let span_start_index = index_data & 0xffffffu;
        let span_line_count = index_data >> 24u;
        
        for (var j = first_span_offset; j < min(span_line_count, first_span_offset + LINES_PER_QUAD); j++) {
            let p0 = points[span_start_index + j - 1u];
            let p1 = points[span_start_index + j];
            a += area(p0, p1, xy) * f32(i < number_lines);
            count++;
        }

        first_span_offset = 0u;
        i++;
    }
    
    return vec4f(a);
}