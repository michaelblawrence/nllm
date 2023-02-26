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

#[function_component(PlotComponent)]
pub fn plot_component(props: &PlotComponentProps) -> Html {
    let plot_id = "plot-div";
    let plot = use_state(|| construct_plot(props));

    let p = yew_hooks::use_async::<_, _, ()>({
        let plot = plot.clone();

        async move {
            plotly::bindings::react(plot_id, &*plot).await;
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
        <div id={plot_id}></div>
    }
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