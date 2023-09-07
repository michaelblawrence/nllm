use std::rc::Rc;
use std::{cell::Cell, f64};
use wasm_bindgen::{__rt::WasmRefCell, prelude::*};
use web_sys::CanvasRenderingContext2d;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math)]
    fn random() -> f64;
}

#[derive(Clone, Copy)]
struct CanvasDrawConfig {
    x: f64,
    y: f64,
}

fn main() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let canvas_config: Rc<Cell<CanvasDrawConfig>> =
        Rc::new(Cell::new(CanvasDrawConfig { x: 0.0, y: 0.0 }));
    let mousedown_pos: Rc<Cell<Option<(i32, i32)>>> = Rc::new(Cell::new(None));
    let animate_closure_ref: Rc<WasmRefCell<Option<JsValue>>> = Rc::new(WasmRefCell::new(None));

    let mouse_xy = mousedown_pos.clone();
    let closure = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
        if e.button() == 0 {
            mouse_xy.set(Some((e.offset_x(), e.offset_y())));
            e.prevent_default();
        }
    });
    canvas
        .add_event_listener_with_callback("mousedown", closure.into_js_value().unchecked_ref())
        .unwrap();

    let mouse_xy = mousedown_pos.clone();
    let redraw_frame = animate_closure_ref.clone();
    let canvas_config_clone = canvas_config.clone();
    let closure = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
        if let Some(source) = mouse_xy.get() {
            let target = (e.offset_x(), e.offset_y());
            canvas_config_clone.set(CanvasDrawConfig {
                x: (target.0 - source.0) as f64,
                y: (target.1 - source.1) as f64,
            });
            request_animation_frame(&redraw_frame);
        }
    });
    canvas
        .add_event_listener_with_callback("mousemove", closure.into_js_value().unchecked_ref())
        .unwrap();

    let mouse_xy = mousedown_pos.clone();
    let closure = Closure::<dyn FnMut(_)>::new(move |_: web_sys::MouseEvent| {
        mouse_xy.set(None);
    });
    canvas
        .add_event_listener_with_callback("mouseup", closure.into_js_value().unchecked_ref())
        .unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let animate_closure: Closure<dyn FnMut()> = Closure::wrap({
        let animate_closure_ref = animate_closure_ref.clone();
        let canvas_config = canvas_config.clone();
        Box::new(move || {
            context.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
            draw_canvas(&context, canvas_config.get().clone());

            let window = &web_sys::window().unwrap();
            let animate_closure_ref = animate_closure_ref.borrow();

            let callback: &js_sys::Function = animate_closure_ref.as_ref().unwrap().unchecked_ref();
            // window.request_animation_frame(callback).unwrap();
        }) as Box<dyn FnMut()>
    });

    {
        *animate_closure_ref.borrow_mut() = Some(animate_closure.into_js_value());
    }

    {
        animate_closure_ref
            .borrow()
            .as_ref()
            .unwrap()
            .unchecked_ref::<js_sys::Function>()
            .call0(&JsValue::NULL)
            .unwrap();
    }
}

fn request_animation_frame(animate_closure_ref: &WasmRefCell<Option<JsValue>>) {
    let window = &web_sys::window().unwrap();
    let animate_closure_ref = animate_closure_ref.borrow();

    let callback: &js_sys::Function = animate_closure_ref.as_ref().unwrap().unchecked_ref();
    window.request_animation_frame(callback).unwrap();
}

fn draw_canvas(context: &web_sys::CanvasRenderingContext2d, canvas_config: CanvasDrawConfig) {
    // free_draw_mlp(context);
    let mut block = CanvasBlockConfig::new(context, (10, 10));
    block.set_origin(canvas_config.x, canvas_config.y);
    // draw_mlp(context, block.child_block((1, 1), (2, 4)));
    draw_mhsa(context, block.child_block((1, 1), (8, 8)));
}

fn draw_mhsa(context: &web_sys::CanvasRenderingContext2d, block: CanvasBlockConfig) {
    let embedding_dimension = 32;
    let sequence_len = 4;
    let head_count = 3;
    let thirds = block.reslice(5, 1);
    let input_block = thirds.child_column(0);
    let token_block = thirds.child_column(1);
    let feed_forward_block = thirds.child_column_span(2, 3);

    let node_radius = 4.0;
    let token_node_radius = 16.0;

    let input_block = input_block.reslice(1, embedding_dimension);
    fill_block_nodes(context, input_block, node_radius);

    let token_block = token_block.reslice(1, sequence_len);
    fill_block_nodes(context, token_block, token_node_radius);
    draw_text_block_nodes(context, token_block, &["the", "quick", "brown", "fox"]);

    let feed_forward_block = feed_forward_block.reslice(3, embedding_dimension);
    let ffwd_input_block = feed_forward_block.child_column(0);
    let ffwd_hidden_block = feed_forward_block.child_column(2);
    fill_block_nodes(context, ffwd_input_block, node_radius);
    fill_block_nodes(context, ffwd_hidden_block, node_radius);

    join_block_nodes(context, input_block, token_block, token_node_radius);
    join_block_nodes(context, token_block, ffwd_input_block, token_node_radius);
    join_block_nodes(
        context,
        ffwd_input_block,
        ffwd_hidden_block,
        token_node_radius,
    );

    // fill_block_frame(context, input_block);
    // fill_block_frame(context, input_block);
    // fill_block_frame(context, token_block);
    // fill_block_frame(context, feed_forward_block);
}

#[derive(Clone, Copy)]
struct CanvasBlockConfig {
    x: f64,
    y: f64,
    w: f64,
    h: f64,

    x_blocks: usize,
    y_blocks: usize,
}

impl CanvasBlockConfig {
    fn new(context: &web_sys::CanvasRenderingContext2d, blocks: (usize, usize)) -> Self {
        let canvas = &context.canvas().unwrap();
        let w = canvas.width() as f64;
        let h = canvas.height() as f64;
        Self::new_xy(0.0, 0.0, w, h, blocks)
    }
    fn new_xy(x: f64, y: f64, w: f64, h: f64, blocks: (usize, usize)) -> Self {
        let (x_blocks, y_blocks) = blocks;

        Self {
            x,
            y,
            w,
            h,
            x_blocks,
            y_blocks,
        }
    }
    fn set_origin(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }
    fn child(&self, x_block: usize, y_block: usize) -> Self {
        self.child_block((x_block, y_block), (1, 1))
    }
    fn child_column(&self, x_block: usize) -> Self {
        self.child_block((x_block, 0), (1, self.y_blocks))
    }
    fn child_column_span(&self, x_block: usize, span_length: usize) -> Self {
        self.child_block((x_block, 0), (span_length, self.y_blocks))
    }
    fn child_row(&self, y_block: usize) -> Self {
        self.child_block((0, y_block), (self.x_blocks, 1))
    }
    fn child_block(&self, origin: (usize, usize), size: (usize, usize)) -> Self {
        let (x, y) = self.px_position(origin.0, origin.1);
        let (block_w, block_h) = self.block_size();
        Self {
            x,
            y,
            w: block_w * size.0 as f64,
            h: block_h * size.1 as f64,
            x_blocks: size.0,
            y_blocks: size.1,
        }
    }
    fn reslice(self, new_x_blocks: usize, new_y_blocks: usize) -> Self {
        Self {
            x_blocks: new_x_blocks,
            y_blocks: new_y_blocks,
            ..self
        }
    }
    fn px_position(&self, x_block: usize, y_block: usize) -> (f64, f64) {
        let (x_idx_f64, y_idx_f64) = (x_block as f64, y_block as f64);
        let (block_w, block_h) = self.block_size();
        (self.x + block_w * x_idx_f64, self.y + block_h * y_idx_f64)
    }
    fn block_size(&self) -> (f64, f64) {
        (
            self.w / (self.x_blocks) as f64,
            self.h / self.y_blocks as f64,
        )
    }
    fn blocks(&self) -> (usize, usize) {
        (self.x_blocks, self.y_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_blocks() {
        let block = CanvasBlockConfig::new_xy(0.0, 0.0, 100.0, 100.0, (10, 10));
        assert_eq!((10.0, 10.0), block.block_size());
        assert_eq!((0.0, 0.0), block.px_position(0, 0));
        assert_eq!((10.0, 0.0), block.px_position(1, 0));

        let child = block.child_block((1, 1), (2, 2));
        assert_eq!((10.0, 10.0), child.block_size());
        assert_eq!((10.0, 10.0), child.px_position(0, 0));
        assert_eq!((20.0, 10.0), child.px_position(1, 0));

        let sliced = child.reslice(10, 10);
        assert_eq!((2.0, 2.0), sliced.block_size());
        assert_eq!((10.0, 10.0), sliced.px_position(0, 0));
        assert_eq!((20.0, 14.0), sliced.px_position(5, 2));
    }
}

#[derive(Clone, Copy)]
struct CanvasFeedForwardConfig {
    input_neurons: usize,
    hidden_neurons: usize,
    output_neurons: usize,
    neuron_radius: f64,
    neuron_y_spacing: usize,
    layer_x_spacing: usize,
}

fn draw_text_block_nodes(
    context: &web_sys::CanvasRenderingContext2d,
    block: CanvasBlockConfig,
    labels: &[&str],
) {
    let (_, y_blocks) = block.blocks();
    let (max_width, _) = block.block_size();
    let font_px = 10.0;
    context.set_font(&format!("{font_px}px sans-serif"));
    context.set_text_align("center");

    for (i, label_text) in (0..y_blocks).zip(labels) {
        let (x, y) = block.px_position(0, i);
        context
            .fill_text_with_max_width(label_text, x, y, max_width)
            .unwrap();
    }
}

fn fill_block_frame(context: &web_sys::CanvasRenderingContext2d, block: CanvasBlockConfig) {
    let (x_blocks, y_blocks) = block.blocks();
    let (left, top) = block.px_position(0, 0);
    let (right, bottom) = block.px_position(x_blocks, y_blocks);
    let w = right - left;
    let h = bottom - top;
    set_stroke_brightness(context, 0);
    context.stroke_rect(left, top, w, h);
}

fn fill_block_nodes(
    context: &web_sys::CanvasRenderingContext2d,
    block: CanvasBlockConfig,
    node_radius: f64,
) {
    let (_, y_blocks) = block.blocks();
    for i in 0..y_blocks {
        context.begin_path();
        let (input_x, input_y) = block.px_position(0, i);
        context
            .arc(input_x, input_y, node_radius, 0.0, 2.0 * f64::consts::PI)
            .unwrap();
        // context.set_fill_style(&"white".into());
        // context.fill();
        set_stroke_brightness(context, 0);
        context.stroke();
    }
}

fn join_block_nodes(
    context: &web_sys::CanvasRenderingContext2d,
    source_block: CanvasBlockConfig,
    dest_block: CanvasBlockConfig,
    node_radius: f64,
) {
    let (_, src_blocks) = source_block.blocks();
    let (_, dest_blocks) = dest_block.blocks();

    for i in 0..src_blocks {
        for j in 0..dest_blocks {
            context.begin_path();
            let (src_x, src_y) = source_block.px_position(0, i);
            let (dest_x, dest_y) = dest_block.px_position(0, j);
            context.move_to(src_x + node_radius, src_y);
            context.line_to(dest_x - node_radius, dest_y);
            randomize_stroke_brightness(context);
            context.stroke();
        }
    }
}

fn draw_feed_forward(
    context: &web_sys::CanvasRenderingContext2d,
    config: CanvasFeedForwardConfig,
    block: CanvasBlockConfig,
) {
    // Define MLP architecture
    let CanvasFeedForwardConfig {
        input_neurons,
        hidden_neurons,
        output_neurons,
        neuron_radius,
        neuron_y_spacing,
        layer_x_spacing,
    } = config;

    let max_layer_size: usize = [input_neurons, hidden_neurons, output_neurons]
        .into_iter()
        .max()
        .unwrap();

    // Configure layer format

    let center_y_offset = |neurons: usize| (max_layer_size - neurons) / 2;

    // Init context for drawing
    set_stroke_brightness(context, 0);

    // Draw input layer
    let block_x = 0;
    let input_y_spacing = neuron_y_spacing;
    let input_y_offset = center_y_offset(input_neurons);
    let calc_input_px = |i: usize| block.px_position(block_x, input_y_spacing * i + input_y_offset);
    for i in 0..input_neurons {
        context.begin_path();
        let (input_x, input_y) = calc_input_px(i);
        context
            .arc(input_x, input_y, neuron_radius, 0.0, 2.0 * f64::consts::PI)
            .unwrap();
        context.stroke();
    }

    // Draw hidden layer
    let block_x = block_x + layer_x_spacing;
    let hidden_y_spacing = neuron_y_spacing;
    let hidden_y_offset = center_y_offset(hidden_neurons);
    let calc_hidden_px =
        |i: usize| block.px_position(block_x, hidden_y_spacing * i + hidden_y_offset);
    for i in 0..hidden_neurons {
        context.begin_path();
        let (hidden_x, hidden_y) = calc_hidden_px(i);
        context
            .arc(
                hidden_x,
                hidden_y,
                neuron_radius,
                0.0,
                2.0 * f64::consts::PI,
            )
            .unwrap();
        context.stroke();
    }

    // Draw output layer
    let block_x = block_x + layer_x_spacing;
    let output_y_spacing = neuron_y_spacing;
    let output_y_offset = center_y_offset(output_neurons);
    let calc_output_px =
        |i: usize| block.px_position(block_x, output_y_spacing * i + output_y_offset);
    for i in 0..output_neurons {
        context.begin_path();
        let (output_x, output_y) = calc_output_px(i);
        context
            .arc(
                output_x,
                output_y,
                neuron_radius,
                0.0,
                2.0 * f64::consts::PI,
            )
            .unwrap();
        context.stroke();
    }

    // Draw connections
    for i in 0..input_neurons {
        for j in 0..hidden_neurons {
            context.begin_path();
            let (input_x, input_y) = calc_input_px(i);
            let (hidden_x, hidden_y) = calc_hidden_px(j);
            context.move_to(input_x + neuron_radius, input_y);
            context.line_to(hidden_x - neuron_radius, hidden_y);
            randomize_stroke_brightness(context);
            context.stroke();
        }
    }

    for i in 0..hidden_neurons {
        for j in 0..output_neurons {
            context.begin_path();
            let (hidden_x, hidden_y) = calc_hidden_px(i);
            let (output_x, output_y) = calc_output_px(j);
            context.move_to(hidden_x + neuron_radius, hidden_y);
            context.line_to(output_x - neuron_radius, output_y);
            randomize_stroke_brightness(context);
            context.stroke();
        }
    }
}

fn draw_mlp(context: &web_sys::CanvasRenderingContext2d, block: CanvasBlockConfig) {
    let config = CanvasFeedForwardConfig {
        input_neurons: 16,
        hidden_neurons: 32,
        output_neurons: 16,
        neuron_radius: 3.0,
        neuron_y_spacing: 1,
        layer_x_spacing: 20,
    };
    let block = block.reslice(64, 64);
    draw_feed_forward(context, config, block)
}

fn randomize_stroke_brightness(context: &CanvasRenderingContext2d) {
    let random = random();
    // let random = random.powf(2.0);
    let amt = (random * 255.0) as u8;
    set_stroke_brightness(context, amt)
}

fn set_stroke_brightness(context: &CanvasRenderingContext2d, brightness: u8) {
    let stroke_color = format!("#000000{:02X}", 255 - brightness);
    context.set_stroke_style(&JsValue::from_str(&stroke_color));
}
