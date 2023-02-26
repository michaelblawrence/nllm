use std::rc::Rc;

use wasm_bindgen::{prelude::Closure, JsCast, __rt::WasmRefCell};
use web_sys::{CanvasRenderingContext2d, EventTarget, HtmlCanvasElement};
use yew::prelude::*;

use crate::{ml, components::PlotComponent};

#[function_component]
pub(crate) fn Tester1() -> Html {
    const INC: usize = 4;
    const CANVAS_SIZE: (usize, usize) = (400, 400);
    let counter = use_state(|| 0);
    let onclick = {
        let counter = counter.clone();
        move |_| {
            let value = *counter + INC;
            counter.set(value);
        }
    };

    let node = use_node_ref();

    use_effect_with_deps(
        |node| {
            let canvas = node
                .cast::<HtmlCanvasElement>()
                .expect("node not attached to canvas elem");

            let listener = Closure::<dyn Fn(Event)>::wrap(Box::new(|_| {
                web_sys::console::log_1(&"Clicked!".into());
            }));

            canvas
                .add_event_listener_with_callback("click", listener.as_ref().unchecked_ref())
                .unwrap();

            listener.forget();

            let ctx: CanvasRenderingContext2d = canvas
                .get_context("2d")
                .expect("error in ctx")
                .expect("no 2d ctx")
                .dyn_into()
                .expect("not 2d context");
            fun_name(ctx, &canvas, CANVAS_SIZE);

            move || {
                // cleanup();
            }
        },
        node.clone(),
    );

    html! {
        <div>
            <PlotComponent points={vec![(0.1, 0.2), (1.1, 2.2), (2.1, 0.4)]} />
            <button {onclick}>{ format!("+{INC}") }</button>
            <p>{ *counter }</p>
            <canvas width={CANVAS_SIZE.0.to_string()} height={CANVAS_SIZE.1.to_string()} ref={node}/>
        </div>
    }
}

pub(crate) fn fun_name(
    ctx: CanvasRenderingContext2d,
    canvas: &HtmlCanvasElement,
    canvas_size: (usize, usize),
) {
    use ml::*;
    // canvas.add_event_listener_with_callback("pointermove", Func|x| {});

    let shape = NetworkShape::new(2, 2, vec![30, 6]);
    let rng = Rc::new(JsRng::default());
    let network_input = [0.1, 0.9];
    let target_output = [0.25, 0.75];
    let mut network = Network::new(&shape, LayerInitStrategy::Random(rng.clone()));
    network.set_activation_mode(NetworkActivationMode::Linear);

    let output_1 = network.compute(network_input.iter().into()).unwrap();
    let mut round = WasmRefCell::new(0);

    let mut network = WasmRefCell::new(network);
    let mut handle = add_closure(canvas, "pointermove", move |evt| {
        web_sys::console::log_1(&format!("here").into());
        let mut network = network.borrow_mut();
        network
            .learn(NetworkLearnStrategy::GradientDecent {
                learn_rate: 0.001,
                inputs: network_input.iter().into(),
                target_outputs: target_output.iter().into(),
            })
            .unwrap();

        let error = network
            .compute_error(network_input.iter().into(), &target_output.iter().into())
            .unwrap()
            .iter()
            .sum::<NodeValue>();

        let prev_round = *round.borrow();
        let next_round = prev_round + 1;
        *round.borrow_mut() = next_round;

        web_sys::console::log_1(&format!("round = {next_round:?}, network_cost = {error}").into());
    });
    ctx.set_line_width(10.0);
    ctx.stroke_rect(75.0, 20.0, 40.0, 40.0);
    ctx.fill_rect(80.0, 150.0, 25.0, 25.0);
    handle.2.take().unwrap().forget();
}

pub(crate) struct EventClosureHandle<'a>(&'a EventTarget, &'a str, Option<Closure<dyn Fn(Event)>>);

impl<'a> Drop for EventClosureHandle<'a> {
    fn drop(&mut self) {
        if let Some(closure) = self.2.as_ref() {
            self.0
                .remove_event_listener_with_callback(self.1, closure.as_ref().unchecked_ref())
                .unwrap();
        }
    }
}

pub(crate) fn add_closure<'a>(
    elem: &'a EventTarget,
    event_name: &'a str,
    fn_closure: impl Fn(Event) -> () + 'static,
) -> EventClosureHandle<'a> {
    let listener = Closure::<dyn Fn(Event)>::wrap(Box::new(fn_closure));

    elem.add_event_listener_with_callback(event_name, listener.as_ref().as_ref().unchecked_ref())
        .unwrap();

    EventClosureHandle(elem, event_name, Some(listener))
}
