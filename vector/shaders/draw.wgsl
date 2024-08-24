struct VSOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) info: vec4u,
};

struct Uniforms {
    view_size: vec2u,
    tile_size: u32,
    padding: u32
};

struct DrawSpan {
    position: u32,
    path_id: u32,
    line_start_index: u32,
    line_end_index: u32,
};

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
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var atlas_texture: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler : sampler;


@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VSOutput {
    var out: VSOutput;
    let quad_id = VertexIndex / 6u;

    let draw_span = draw_spans[quad_id];
    let tl = unpack_position(draw_span.position);
    let min_x  = tl.x;
    let min_y  = tl.y;
    let max_x  = f32(draw_span.path_id >> 16u);
    let max_y  = min_y + f32(uniforms.tile_size);
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

    
    let end_index = draw_span.line_end_index & 0xffffffu;
    let backdrop = i32(draw_span.line_end_index >> 24u) - 127;
    let count = min(end_index - draw_span.line_start_index, 255u);
    out.info.x = draw_span.line_start_index | (count  << 24u);
    out.info.y = colors[draw_span.path_id & 0xffffu];
    out.info.z = bitcast<u32>(backdrop);

    var v_pos = pos[vertex_id];
    var p = v_pos * vec2f(2.0) / vec2f(uniforms.view_size) - 1.0;
    out.position = vec4<f32>(p.x, -p.y, 0.0, 1.0);
    return out;
}

@fragment
fn frag_main(@builtin(position) pos: vec4f, @location(0) @interpolate(flat) info: vec4u) -> @location(0) vec4<f32> {
    let xy = pos.xy - vec2f(0.5);
    let start = info.x & 0xffffffu;
    let cnt = info.x >> 24u;
    var a = f32(bitcast<i32>(info.z));
    for (var i = 0u; i < cnt; i++) {
        let index_data = line_indices[start + i];
        let span_start_index = index_data & 0xffffffu;
        let span_line_count = index_data >> 24u;
        for (var j = 0u; j < span_line_count; j++) {
            let p0 = points[span_start_index + j - 1u];
            let p1 = points[span_start_index + j];
            a += area(p0, p1, xy);
        }
    }
    a = min(abs(a - 2.0 * round(0.5 * a)), 1.0); 
    return unpack4x8unorm(info.y) * a;// * 0.8 + vec4f(0.2, 0.0, 0.0, 0.3);
}