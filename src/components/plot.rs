use wasm_bindgen::__rt::WasmRefCell;
use yew::prelude::*;

use crate::ml;

#[derive(Properties, PartialEq, Clone)]
pub struct PlotComponentProps {
    pub points: Vec<(ml::NodeValue, ml::NodeValue)>,
    #[prop_or_default]
    pub x_axis_log: bool,
}

impl PlotComponentProps {
    pub fn x_points(&self) -> Vec<ml::NodeValue> {
        self.points.iter().map(|(x, _)| *x).collect()
    }
    pub fn y_points(&self) -> Vec<ml::NodeValue> {
        self.points.iter().map(|(_, y)| *y).collect()
    }
}

static mut PLOT_COUNTER: usize = 0;

#[function_component(PlotComponent)]
pub fn plot_component(props: &PlotComponentProps) -> Html {
    let plot_id = use_state(|| {
        let next_counter = get_next_counter();
        format!("plot-div-{}", next_counter)
    });
    let plot = use_state(|| construct_plot(props));

    let p = yew_hooks::use_async::<_, _, ()>({
        let plot = plot.clone();
        let plot_id = plot_id.clone();

        async move {
            plotly::bindings::react(plot_id.as_str(), &*plot).await;
            Ok(())
        }
    });

    use_effect_with_deps(
        {
            let plot = plot.clone();
            move |props: &PlotComponentProps| {
                plot.set(construct_plot(props));
                p.run();
                move || {}
            }
        },
        props.clone(),
    );

    html! {
        <div id={(*plot_id).clone()}></div>
    }
}

// TODO: use guid etc. rather than counter for id creation
fn get_next_counter() -> usize {
    unsafe {
        PLOT_COUNTER += 1;
    }
    unsafe { PLOT_COUNTER }
}

fn construct_plot(props: &PlotComponentProps) -> plotly::Plot {
    use plotly::{
        layout::{Axis, AxisType},
        Plot, Scatter,
    };

    let mut plot = Plot::new();
    if props.x_axis_log {
        let layout = plot.layout().clone();
        plot.set_layout(layout.y_axis(Axis::default().type_(AxisType::Log)));
    }

    let trace = Scatter::new(props.x_points(), props.y_points());
    plot.add_trace(trace);

    plot
}
