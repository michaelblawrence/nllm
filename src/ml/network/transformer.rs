use std::iter;

use anyhow::{anyhow, Result};
use itertools::Itertools;

use crate::ml::RngStrategy;

use self::{dense::Dense, linear::Linear};

use super::{layer::LayerInitStrategy, LayerValues, NodeValue};

pub mod blocks {
    use anyhow::Result;

    use super::{
        layers::{DropoutLayer, FeedForwardLayer, LayerNormalization, MultiHeadSelfAttentionLayer},
        linear::Linear,
    };

    pub struct Encoder {
        attention: MultiHeadSelfAttentionLayer,
        network: FeedForwardLayer,
        dropout: (DropoutLayer, DropoutLayer),
        layer_norm: (LayerNormalization, LayerNormalization),
    }

    impl Encoder {
        pub fn new(sequence_len: usize, embedding_dimension: usize, head_count: usize) -> Encoder {
            Self::new_builder(sequence_len, embedding_dimension, head_count).build()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
        ) -> builder::EncoderBuilder {
            builder::EncoderBuilder::new(sequence_len, embedding_dimension, head_count)
        }

        pub fn forward_training(&self, inputs: &Linear) -> Result<Linear> {
            let attention_output = self.attention.forward(&inputs)?;
            let attention_output = self.dropout.0.forward(&attention_output)?;

            let skip_attention_output = attention_output.iter().add(inputs.iter()).collect();
            let skip_attention_output = self.layer_norm.0.forward(&skip_attention_output)?;

            let network_inputs = &skip_attention_output;
            let network_output = self.network.forward(&network_inputs)?;
            let network_output = self.dropout.1.forward(&network_output)?;

            let skip_network_output = network_output.iter().add(network_inputs.iter()).collect();
            let skip_network_output = self.layer_norm.0.forward(&skip_network_output)?;

            let encoder_output = self.layer_norm.1.forward(&skip_network_output)?;
            Ok(encoder_output)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{layer::LayerInitStrategy, RngStrategy};

        use super::*;

        #[test]
        fn encoder_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;

            let rng = RngStrategy::testable(1234);

            let inputs = new_inputs(seq_len, embed_dim, &rng);
            let encoder_block = Encoder::new_builder(seq_len, embed_dim, head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build();

            let output = encoder_block.forward_training(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        fn new_inputs(
            sequence_len: usize,
            embedding_dimension: usize,
            rng: &RngStrategy,
        ) -> Linear {
            let init_strategy = LayerInitStrategy::Kaiming;
            let mut inputs = Linear::new(sequence_len, embedding_dimension);
            inputs.initialize_as_layer(&init_strategy, rng);
            inputs
        }
    }

    mod builder {
        use super::{
            super::layers::{DropoutLayer, FeedForwardLayer, LayerNormalization},
            Encoder,
        };

        use crate::ml::{
            layer::LayerInitStrategy, transformer::layers::MultiHeadSelfAttentionLayer, NodeValue,
            RngStrategy,
        };

        pub struct EncoderBuilder {
            pub(crate) sequence_len: usize,
            pub(crate) model_dimension: usize,
            pub(crate) head_count: usize,
            pub(crate) rng: RngStrategy,
            pub(crate) feed_forward_init_strategy: LayerInitStrategy,
            pub(crate) feed_forward_hidden_dimension: usize,
            pub(crate) dropout_rate: NodeValue,
        }

        impl EncoderBuilder {
            pub fn new(sequence_len: usize, model_dimension: usize, head_count: usize) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count,
                    rng: Default::default(),
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> Encoder {
                let attention = MultiHeadSelfAttentionLayer::new(
                    self.sequence_len,
                    self.model_dimension,
                    self.head_count,
                    &self.rng,
                );
                let network = FeedForwardLayer::new(
                    self.model_dimension,
                    self.feed_forward_hidden_dimension,
                    &self.feed_forward_init_strategy,
                    &self.rng,
                );
                let dropout1 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let dropout2 = DropoutLayer::new(self.dropout_rate, &self.rng);
                let layer_norm1 = LayerNormalization::new();
                let layer_norm2 = LayerNormalization::new();

                Encoder {
                    attention,
                    network,
                    dropout: (dropout1, dropout2),
                    layer_norm: (layer_norm1, layer_norm2),
                }
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_feed_forward_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.feed_forward_init_strategy = init_strategy;
                self
            }

            pub fn with_feed_forward_hidden_dimension(mut self, hidden_dimension: usize) -> Self {
                self.feed_forward_hidden_dimension = hidden_dimension;
                self
            }

            pub fn with_dropout_rate(mut self, dropout_rate: NodeValue) -> Self {
                self.dropout_rate = dropout_rate;
                self
            }
        }
    }
}

pub mod layers {
    use anyhow::Result;

    use crate::ml::NetworkActivationMode;

    use super::{*, attention::SelfAttention};

    pub struct MultiHeadSelfAttentionLayer {
        heads: Vec<SelfAttention>,
        dense_layer: Dense,
    }

    impl MultiHeadSelfAttentionLayer {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            rng: &RngStrategy,
        ) -> Self {
            let heads = iter::repeat_with(|| {
                SelfAttention::new_head(sequence_len, embedding_dimension, head_count, rng)
            })
            .take(head_count)
            .collect();

            let layer_strategy = LayerInitStrategy::KaimingZeroBias;
            let dense_layer = Dense::new(
                embedding_dimension,
                embedding_dimension,
                &layer_strategy,
                &rng,
            );

            Self { heads, dense_layer }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            let mut head_outputs = vec![];
            for head in &self.heads {
                let head_output = head.forward(inputs)?;
                head_outputs.push(head_output);
            }

            let attention_output = head_outputs
                .into_iter()
                .fold(None, |concat: Option<Linear>, head| match concat {
                    Some(iter) => Some(iter.concat(&head).collect()),
                    None => Some(head),
                })
                .unwrap();

            let mut dense_layer_outputs = vec![];
            for row in attention_output.rows_iter() {
                let dense_output = self.dense_layer.forward(&LayerValues::new(row.to_vec()))?;
                dense_layer_outputs.push(dense_output);
            }

            Linear::from_values(&dense_layer_outputs)
        }

        pub fn backward(&self, output_gradients: &Linear) -> Result<Linear> {
            let mut dense_layer_input_gradients = vec![];

            for row in output_gradients.rows_iter() {
                let dense_output = self.dense_layer.backward(LayerValues::new(row.to_vec()))?;
                dense_layer_input_gradients.push(dense_output);
            }

            let dense_layer_input_gradient = Linear::from_values(&dense_layer_input_gradients);

            todo!()
        }
    }

    pub struct FeedForwardLayer {
        hidden_layer: Dense,
        output_layer: Dense,
    }

    impl FeedForwardLayer {
        pub fn new(
            model_dimensions: usize,
            hidden_dimension: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut hidden_layer = Dense::new(model_dimensions, hidden_dimension, strategy, rng);
            hidden_layer.set_activation(NetworkActivationMode::RelU);

            let output_layer = Dense::new(hidden_dimension, model_dimensions, strategy, rng);
            Self {
                hidden_layer,
                output_layer,
            }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            let mut dense_layer_outputs = vec![];

            for row in inputs.rows_iter() {
                let layer_input = LayerValues::new(row.to_vec());
                let hidden_layer = self.hidden_layer.forward(&layer_input)?;
                let dense_output = self.output_layer.forward(&hidden_layer)?;
                dense_layer_outputs.push(dense_output);
            }

            Linear::from_values(&dense_layer_outputs)
        }
    }

    pub struct DropoutLayer {
        dropout_rate: NodeValue,
        rng: RngStrategy,
    }

    impl DropoutLayer {
        pub fn new(dropout_rate: NodeValue, rng: &RngStrategy) -> Self {
            let rng = rng.clone().upgrade();
            Self { dropout_rate, rng }
        }

        pub fn forward(&self, input: &Linear) -> Result<Linear> {
            Ok(input
                .iter()
                .dropout(self.dropout_rate, self.rng.clone())
                .collect())
        }
    }

    pub struct LayerNormalization;

    impl LayerNormalization {
        pub fn new() -> Self {
            Self
        }

        pub fn forward(&self, input: &Linear) -> Result<Linear> {
            let mean = input.iter().flatten_mean();
            let std_dev = input.iter().flatten_stddev(mean.iter());

            // TODO: optimise norm calc + add scale + shift params
            let norm = input
                .iter()
                .add(mean.iter().multiply_scalar(-1.0).grow(input.stride()))
                .dot_product(std_dev.iter().powf_scalar(-1.0).grow(input.stride()))
                .collect();

            Ok(norm)
        }
    }
}

pub mod attention {
    use anyhow::{anyhow, Result};
    
    use crate::ml::{RngStrategy, NodeValue, layer::LayerInitStrategy};

    use super::linear::Linear;

    pub struct SelfAttention {
        keys: Linear,
        values: Linear,
        queries: Linear,
        mask: Option<Linear>,
        embedding_dimension: usize,
        sequence_len: usize,
    }

    impl SelfAttention {
        pub fn new(sequence_len: usize, embedding_dimension: usize, rng: &RngStrategy) -> Self {
            Self::new_head(sequence_len, embedding_dimension, 1, rng)
        }

        pub fn new_head(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            rng: &RngStrategy,
        ) -> Self {
            let head_dimension = embedding_dimension / head_count;
            Self {
                keys: Self::new_kvq_linear(embedding_dimension, head_dimension, rng),
                values: Self::new_kvq_linear(embedding_dimension, head_dimension, rng),
                queries: Self::new_kvq_linear(embedding_dimension, head_dimension, rng),
                mask: None,
                embedding_dimension,
                sequence_len,
            }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            if inputs.count() == 0 {
                Err(anyhow!("no values provided for input vectors"))?;
            }
            if inputs.count() != self.sequence_len {
                Err(anyhow!("mismatched number of input vectors"))?;
            }
            if inputs.stride() != self.embedding_dimension {
                Err(anyhow!("mismatched input vector size"))?;
            };

            self.forward_advanced(&inputs, &inputs, &inputs)
        }

        pub fn forward_advanced(
            &self,
            key_inputs: &Linear,
            value_inputs: &Linear,
            query_inputs: &Linear,
        ) -> Result<Linear> {
            let keys = key_inputs.matrix_product(&self.keys);
            let values = value_inputs.matrix_product(&self.values);
            let queries = query_inputs.matrix_product(&self.queries);
            let mask = self.mask.as_ref();

            let output = SelfAttention::scaled_self_attention(&queries, &keys, &values, mask);

            Ok(output)
        }

        pub fn backward(&self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            if output_gradients.stride() != self.values.stride() {
                Err(anyhow!("mismatched ouput vector size"))?;
            }

            let keys = inputs.matrix_product(&self.keys);
            let values = inputs.matrix_product(&self.values);
            let queries = inputs.matrix_product(&self.queries);
            let (dkeys, dvalues, dqueries) =
                Self::scaled_self_attention_d(&queries, &keys, &values, &output_gradients);

            todo!()
        }

        fn scaled_self_attention(
            queries: &Linear,
            keys: &Linear,
            values: &Linear,
            mask: Option<&Linear>,
        ) -> Linear {
            let attention_scores = queries.iter().matrix_transpose_product(keys.iter());

            let scaled_attention_scores = attention_scores
                .iter()
                .multiply_scalar(1.0 / keys.count() as NodeValue);

            let masked_attention_scores = match mask {
                Some(mask) => scaled_attention_scores.set_mask(mask.iter(), NodeValue::NEG_INFINITY),
                None => scaled_attention_scores,
            };

            let attention_weights = masked_attention_scores.softmax();
            attention_weights.matrix_product(&values)
        }

        fn scaled_self_attention_d(
            output_gradients: &Linear,
            queries: &Linear,
            keys: &Linear,
            values: &Linear,
        ) -> (Linear, Linear, Linear) {
            todo!();
        }

        fn new_kvq_linear(
            embedding_dimension: usize,
            head_dimension: usize,
            rng: &RngStrategy,
        ) -> Linear {
            let mut linear = Linear::new(embedding_dimension, head_dimension);
            linear.initialize_as_layer(&LayerInitStrategy::Kaiming, &rng);
            linear
        }

        pub fn set_mask(&mut self, mask: Option<Linear>) {
            self.mask = mask;
        }
    }

    #[cfg(test)]
    mod tests {
        use std::iter;

        use crate::ml::{transformer::layers::MultiHeadSelfAttentionLayer, LayerValues};

        use super::*;

        #[test]
        fn attention_can_compute_outputs_single_head() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_inputs(seq_len, embed_dim, &rng);
            let inputs = Linear::from_values(&inputs).unwrap();
            let attention_layer = SelfAttention::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn attention_can_compute_outputs_multi_head() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_inputs(seq_len, embed_dim, &rng);
            let inputs = Linear::from_values(&inputs).unwrap();
            let attention_layer = MultiHeadSelfAttentionLayer::new(seq_len, embed_dim, 3, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn attention_can_compute_for_kvq() {
            let seq_len = 3;
            let embed_dim = 4;

            let rng = RngStrategy::testable(1234);

            let queries = new_linear(seq_len, embed_dim, &rng);
            let keys = new_linear(seq_len, embed_dim, &rng);
            let values = new_linear(seq_len, embed_dim, &rng);

            let output = SelfAttention::scaled_self_attention(&queries, &keys, &values, None);

            assert_eq!(output.count(), seq_len);
            assert_eq!(output.stride(), embed_dim);
        }

        fn new_linear(token_count: usize, embedding_dimension: usize, rng: &RngStrategy) -> Linear {
            let mut linear = Linear::new(token_count, embedding_dimension);
            linear.initialize_as_layer(&LayerInitStrategy::Kaiming, &rng);
            linear
        }

        fn new_inputs(
            sequence_len: usize,
            embedding_dimension: usize,
            rng: &RngStrategy,
        ) -> Vec<LayerValues> {
            let init_strategy = LayerInitStrategy::Kaiming;

            let mut inputs: Vec<LayerValues> =
                iter::repeat(LayerValues::new(vec![0.0; embedding_dimension]))
                    .take(sequence_len)
                    .collect();

            init_strategy.apply(
                inputs.iter_mut().flat_map(|x| x.iter_mut()),
                iter::empty(),
                embedding_dimension,
                &rng,
            );

            inputs
        }
    }
}

pub mod dense {
    use anyhow::{anyhow, Result};

    use crate::ml::{layer::LayerInitStrategy, LayerValues, NetworkActivationMode, RngStrategy};

    use super::linear::Linear;

    #[derive(Debug, PartialEq)]
    pub struct Dense {
        weigths: Linear,
        bias: Option<Linear>,
        activation: Option<NetworkActivationMode>,
        inputs_count: usize,
        outputs_count: usize,
    }

    impl Dense {
        pub fn new(
            inputs_count: usize,
            outputs_count: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut weigths = Linear::new(inputs_count, outputs_count);
            weigths.initialize_as_layer(strategy, rng);

            let bias = if strategy.requires_bias() {
                let mut b = Linear::new(1, outputs_count);
                b.initialize_as_layer_bias(strategy, rng);
                Some(b)
            } else {
                None
            };

            Self {
                weigths,
                bias,
                activation: None,
                inputs_count,
                outputs_count,
            }
        }

        pub fn forward(&self, inputs: &LayerValues) -> Result<LayerValues> {
            if inputs.len() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }

            let mut weighted_inputs = self.weigths.forward(&inputs)?;

            if let Some(bias) = &self.bias {
                weighted_inputs
                    .iter_mut()
                    .zip(bias.values_iter())
                    .for_each(|(x, (b, ..))| *x += b);
            }

            let outputs = match &self.activation {
                Some(activation) => activation.apply(&weighted_inputs),
                None => weighted_inputs,
            };

            Ok(outputs)
        }

        pub fn backward(&self, output_gradients: LayerValues) -> Result<LayerValues> {
            let weighted_inputs_gradients = match &self.activation {
                Some(activation) => activation
                    .derivative(todo!(
                        "need values fron foward pass to calc activation deriv"
                    ))
                    .multiply_iter(&output_gradients)
                    .collect(),
                None => output_gradients,
            };
            let input_gradients = self.weigths.forward(&weighted_inputs_gradients)?;
            // if let Some(bias) = &self.bias { bias.apply_grads(&output_gradients); }
            Ok(input_gradients)
        }

        pub fn set_activation(&mut self, activation: NetworkActivationMode) {
            self.activation = Some(activation);
        }
    }
}

pub mod linear {
    use anyhow::Context;

    use super::*;

    #[derive(Debug, PartialEq)]
    pub struct Linear {
        inner: LayerValues,
        stride: usize,
        count: usize,
    }

    impl Linear {
        pub fn new(inputs_count: usize, outputs_count: usize) -> Self {
            let size = inputs_count * outputs_count;
            Self {
                inner: LayerValues::new(vec![0.0; size]),
                stride: outputs_count,
                count: inputs_count,
            }
        }

        pub fn from_iter<I: Iterator<Item = NodeValue>>(stride: usize, values: I) -> Result<Self> {
            let inner: LayerValues = values.collect();
            let count = inner.len() / stride;

            if inner.len() != stride * count {
                Err(anyhow!("mismatched values length/stride"))?;
            }

            Ok(Self {
                inner,
                stride,
                count,
            })
        }

        pub fn from_values(values: &[LayerValues]) -> Result<Self> {
            let stride = values.first().context("no values provided")?.len();
            let flattened_inputs = values.into_iter().flat_map(|x| x.iter());
            Linear::from_iter(stride, flattened_inputs.copied())
        }

        pub fn initialize_as_layer(&mut self, strategy: &LayerInitStrategy, rng: &RngStrategy) {
            strategy.apply(self.inner.iter_mut(), iter::empty(), self.count, &rng);
        }

        pub fn initialize_as_layer_bias(
            &mut self,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) {
            strategy.apply(iter::empty(), self.inner.iter_mut(), self.count, &rng);
        }

        pub fn forward(&self, inputs: &LayerValues) -> Result<LayerValues> {
            if inputs.len() != self.count {
                Err(anyhow!("mismatched input vector size"))?;
            }

            let mut output = vec![0.0; self.stride];

            self.inner
                .chunks(self.stride)
                .zip(inputs.iter())
                .flat_map(|(w, input)| w.into_iter().map(|&w| w * *input))
                .enumerate()
                .for_each(|(i, x)| output[i % self.stride] += x);

            Ok(LayerValues::new(output))
        }

        pub fn backward(&self, output_gradients: &LayerValues) -> Result<LayerValues> {
            if output_gradients.len() != self.stride {
                Err(anyhow!("mismatched ouput vector size"))?;
            }

            let input_gradients = self
                .inner
                .chunks(self.stride)
                .map(|w| {
                    output_gradients
                        .multiply_iter(&w.into_iter().copied().collect())
                        .sum::<NodeValue>()
                })
                .collect();

            Ok(input_gradients)
        }

        pub fn iter(&self) -> LinearIter {
            LinearIter {
                inner: Box::new(self.inner.iter().copied()),
                stride: self.stride,
                count: self.count,
            }
        }

        pub fn iter_transpose(&self) -> LinearIter {
            let stride = self.stride;
            let count = self.count;

            let x = (0..self.inner.len()).map(move |idx| {
                let inner_idx = |x, y| x + y * stride;
                // let identity_xy = |i| (i % stride, i / stride);
                let transpose_xy = |i| (i % count, i / count);

                let (x, y) = transpose_xy(idx);
                let transpose_idx = inner_idx(y, x);
                self.inner[transpose_idx]
            });
            LinearIter {
                inner: Box::new(x),
                stride: self.count,
                count: self.stride,
            }
        }

        pub fn concat<'a>(&'a self, rhs: &'a Linear) -> LinearIter<'a> {
            assert_eq!(self.count, rhs.count, "mismatched dimensions");
            let self_items = self.inner.chunks(self.stride);
            let rhs_items = rhs.inner.chunks(rhs.stride);
            LinearIter {
                inner: Box::new(
                    self_items
                        .zip(rhs_items)
                        .flat_map(|(lhs, rhs)| lhs.iter().chain(rhs).copied()),
                ),
                stride: self.stride + rhs.stride,
                count: self.count,
            }
        }

        pub fn matrix_product(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.stride, rhs.count, "mismatched dimensions");
            self.iter().matrix_transpose_product(rhs.iter_transpose())
        }

        pub fn stride(&self) -> usize {
            self.stride
        }

        pub fn count(&self) -> usize {
            self.count
        }

        pub fn to_values(self) -> Vec<LayerValues> {
            self.rows_iter()
                .map(|x| LayerValues::new(x.to_vec()))
                .collect()
        }

        pub fn rows_iter(&self) -> impl Iterator<Item = &[NodeValue]> {
            self.inner.chunks(self.stride)
        }

        pub fn values_iter(&self) -> impl Iterator<Item = (&NodeValue, usize, usize)> {
            self.inner
                .iter()
                .enumerate()
                .map(|(idx, x)| (x, idx % self.stride, idx / self.stride))
        }
    }

    pub struct LinearIter<'a> {
        inner: Box<dyn Iterator<Item = NodeValue> + 'a>,
        stride: usize,
        count: usize,
    }

    impl<'a> LinearIter<'a> {
        pub fn dot_product(self, other: Self) -> Self {
            assert_eq!(self.stride, other.stride, "mismatched dimensions");
            assert_eq!(self.count, other.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(other.inner).map(|x| x.0 * x.1)),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn matrix_transpose_product(self, rhs_transpose: Self) -> Linear {
            assert_eq!(self.stride, rhs_transpose.stride, "mismatched dimensions");
            let self_strides = self.inner.chunks(self.stride);
            let rhs_strides = rhs_transpose
                .inner
                .chunks(rhs_transpose.stride)
                .into_iter()
                .map(|chunk| chunk.collect_vec())
                .collect_vec();

            let inner = self_strides
                .into_iter()
                .flat_map(|a| {
                    let a = a.collect_vec();
                    rhs_strides.iter().map(move |b| {
                        a.iter()
                            .zip(b.iter())
                            .map(|(a, b)| a * b)
                            .sum::<NodeValue>()
                    })
                })
                .collect();

            Linear {
                inner,
                stride: rhs_transpose.count,
                count: self.count,
            }
        }
        pub fn add(self, other: Self) -> Self {
            assert_eq!(self.stride, other.stride, "mismatched dimensions");
            assert_eq!(self.count, other.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(other.inner).map(|x| x.0 + x.1)),
                stride: self.stride,
                count: self.count,
            }
        }
        // pub fn broadcast_add(self, other: Self) -> Linear {
        //     assert_eq!(self.count, other.count, "mismatched dimensions");
        //     let stride = self.stride.max(other.stride);
        //     Linear {
        //         inner: self
        //             .inner
        //             .chunks(self.stride)
        //             .into_iter()
        //             .zip(other.inner.chunks(other.stride).into_iter())
        //             .flat_map(|(x, y)| {
        //                 x.collect_vec()
        //                     .into_iter()
        //                     .cycle()
        //                     .take(stride)
        //                     .zip(y.collect_vec().into_iter().cycle().take(stride))
        //                     .map(|x| x.0 + x.1)
        //             })
        //             .collect(),
        //         stride: self.stride,
        //         count: self.count,
        //     }
        // }
        pub fn multiply_scalar(self, rhs: NodeValue) -> Self {
            Self {
                inner: Box::new(self.inner.map(move |x| x * rhs)),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn powf_scalar(self, n: NodeValue) -> Self {
            Self {
                inner: Box::new(self.inner.map(move |x| x.powf(n))),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn dropout(self, dropout_rate: f64, rng: RngStrategy) -> Self {
            assert!(
                dropout_rate <= 1.0 && dropout_rate >= 0.0,
                "invalid dropout rate"
            );
            if dropout_rate == 0.0 {
                return self;
            }

            Self {
                inner: Box::new(self.inner.map(move |x| {
                    if rng.rand() < dropout_rate {
                        0.0
                    } else {
                        x / (1.0 - dropout_rate)
                    }
                })),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn set_mask(self, mask: Self, masked_value: NodeValue) -> Self {
            assert_eq!(self.stride, mask.stride, "mismatched dimensions");
            assert_eq!(self.count, mask.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(mask.inner).map(move |(x, mask)| {
                    if mask == 0.0 {
                        masked_value
                    } else {
                        x
                    }
                })),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn grow(self, stride: usize) -> Self {
            assert_eq!(self.stride, 1, "can only grow when stride dimension = 1");
            assert_ne!(stride, 0, "invalid stride dimension");
            Self {
                inner: Box::new(self.inner.flat_map(move |x| iter::repeat(x).take(stride))),
                stride: stride,
                count: self.count,
            }
        }
        pub fn softmax(self) -> Linear {
            let strides = self
                .inner
                .chunks(self.stride)
                .into_iter()
                .flat_map(|chunk_iter| {
                    let chunk = chunk_iter.collect_vec();
                    let max = chunk
                        .iter()
                        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .cloned()
                        .unwrap_or_default();
                    let exp_iter = chunk.into_iter().map(move |x| (x - max).exp());
                    let sum: NodeValue = exp_iter.clone().sum();
                    exp_iter.map(move |x| x / sum)
                })
                .collect();
            Linear {
                inner: strides,
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn flatten_sum(self) -> Linear {
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .map(|x| x.sum::<NodeValue>())
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        pub fn flatten_mean(self) -> Linear {
            let stride = self.stride as NodeValue;
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .map(|x| x.sum::<NodeValue>() / stride)
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        pub fn flatten_stddev(self, mean: Self) -> Linear {
            let factor = (self.stride as NodeValue).powf(-0.5);
            Linear {
                inner: self
                    .inner
                    .chunks(self.stride)
                    .into_iter()
                    .zip(mean.inner)
                    .map(|(x, mean)| {
                        x.map(|x| (x - mean).powi(2)).sum::<NodeValue>().sqrt() * factor
                    })
                    .collect(),
                stride: 1,
                count: self.count,
            }
        }
        pub fn collect(self) -> Linear {
            Linear {
                inner: self.inner.collect(),
                stride: self.stride,
                count: self.count,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn can_linear_perform_mat_mul() {
            let a = Linear::from_iter(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [10.0, 11.0, 20.0, 21.0, 30.0, 31.0].into_iter()).unwrap();
            let y = a.matrix_product(&b);

            let expected = Linear::from_iter(2, [140.0, 146.0, 320.0, 335.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_transpose() {
            let a = Linear::from_iter(3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter()).unwrap();
            let y = a.iter_transpose().collect();

            let expected =
                Linear::from_iter(2, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_add_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 6.0, 7.0, 8.0].into_iter()).unwrap();
            let y = a.iter().add(b.iter()).collect();

            let expected = Linear::from_iter(2, [6.0, 8.0, 10.0, 12.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_dot_product() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 6.0, 7.0, 8.0].into_iter()).unwrap();
            let y = a.iter().dot_product(b.iter()).collect();

            let expected = Linear::from_iter(2, [5.0, 12.0, 21.0, 32.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_scale_linear() {
            let x = Linear::from_iter(2, [1.0, 1.0, 1.0, 1.0].into_iter()).unwrap();
            let y = x.iter().multiply_scalar(4.0).collect();

            let expected = Linear::from_iter(2, [4.0, 4.0, 4.0, 4.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_flatten_sum() {
            let x = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = x.iter().flatten_sum();

            let expected = Linear::from_iter(1, [3.0, 7.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_mean_stride_values() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = a.iter().flatten_mean();

            let expected = Linear::from_iter(1, [1.5, 3.5].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_softmax() {
            let x = Linear::from_iter(
                2,
                [1.0, 1.0, 0.0, 0.0, -100.0, -100.0, 100.0, 100.0].into_iter(),
            )
            .unwrap();
            let y = x.iter().softmax();

            let expected =
                Linear::from_iter(2, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_masked_softmax() {
            let x = Linear::from_iter(2, [1.0, 1.0, 0.0, 0.0, 100.0, 100.0].into_iter()).unwrap();
            let mask = Linear::from_iter(2, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0].into_iter()).unwrap();

            let y = x
                .iter()
                .set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                .softmax();

            let expected =
                Linear::from_iter(2, [1.0, 0.0, 0.5, 0.5, 0.0, 1.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }
    }
}
