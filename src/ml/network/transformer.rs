use std::iter;

use anyhow::anyhow;

use crate::ml::RngStrategy;

use self::{dense::Dense, linear::Linear};

use super::{layer::LayerInitStrategy, LayerValues, NodeValue};

pub mod encoder {
    use anyhow::Result;

    use super::{
        blocks::EncoderBlock,
        layers::{DropoutLayer, EmbeddingLayer, PositionalEmbeddingLayer},
        linear::Linear,
    };

    pub struct Encoder {
        blocks: Vec<EncoderBlock>,
        token_embedding: EmbeddingLayer,
        position_embedding: PositionalEmbeddingLayer,
        dropout: DropoutLayer,
    }

    impl Encoder {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            source_vocab_size: usize,
        ) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, source_vocab_size).build()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            source_vocab_size: usize,
        ) -> builder::EncoderBuilder {
            builder::EncoderBuilder::new(sequence_len, embedding_dimension, source_vocab_size)
        }

        pub fn forward_training<T: AsRef<[usize]>>(&self, input_sequence: T) -> Result<Linear> {
            let input_sequence = input_sequence.as_ref();
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = self.dropout.forward(&embeddings)?;

            let mut block_output = block_input;
            for block in &self.blocks {
                let block_input = &block_output;
                block_output = block.forward_training(&block_input)?;
            }

            Ok(block_output)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::RngStrategy;

        use super::*;

        #[test]
        fn encoder_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build();

            let inputs = [3, 6, 9];
            let output = encoder.forward_training(&inputs).unwrap();
            println!("outputs -> {output:#}");

            let output = output.to_values();
            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }
    }

    pub mod builder {
        use super::{super::layers::DropoutLayer, Encoder};

        use crate::ml::{
            layer::LayerInitStrategy,
            transformer::{
                blocks::EncoderBlock,
                layers::{EmbeddingLayer, PositionalEmbeddingLayer},
            },
            NodeValue, RngStrategy,
        };

        pub struct EncoderBuilder {
            pub(crate) sequence_len: usize,
            pub(crate) model_dimension: usize,
            pub(crate) head_count: usize,
            pub(crate) source_vocab_size: usize,
            pub(crate) block_count: usize,
            pub(crate) rng: RngStrategy,
            pub(crate) embedding_init_strategy: LayerInitStrategy,
            pub(crate) feed_forward_init_strategy: LayerInitStrategy,
            pub(crate) feed_forward_hidden_dimension: usize,
            pub(crate) dropout_rate: NodeValue,
        }

        impl EncoderBuilder {
            pub fn new(
                sequence_len: usize,
                model_dimension: usize,
                source_vocab_size: usize,
            ) -> Self {
                Self {
                    sequence_len,
                    model_dimension,
                    head_count: 3,
                    source_vocab_size,
                    block_count: 6,
                    rng: Default::default(),
                    embedding_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> Encoder {
                let block_builder = EncoderBlock::new_builder(
                    self.sequence_len,
                    self.model_dimension,
                    self.head_count,
                );
                let block_builder = block_builder
                    .with_rng(self.rng.clone())
                    .with_dropout_rate(self.dropout_rate)
                    .with_feed_forward_hidden_dimension(self.feed_forward_hidden_dimension)
                    .with_feed_forward_init_strategy(self.feed_forward_init_strategy);

                let mut blocks = vec![];
                for _ in 0..self.block_count {
                    let block = block_builder.clone().build();
                    blocks.push(block);
                }
                let token_embedding = EmbeddingLayer::new(
                    self.model_dimension,
                    self.source_vocab_size,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let position_embedding = PositionalEmbeddingLayer::new(
                    self.model_dimension,
                    self.sequence_len,
                    &self.embedding_init_strategy,
                    &self.rng,
                );
                let dropout = DropoutLayer::new(self.dropout_rate, &self.rng);

                Encoder {
                    blocks,
                    token_embedding,
                    position_embedding,
                    dropout,
                }
            }

            pub fn with_rng(mut self, rng: RngStrategy) -> Self {
                self.rng = rng;
                self
            }

            pub fn with_embedding_init_strategy(
                mut self,
                init_strategy: LayerInitStrategy,
            ) -> Self {
                self.embedding_init_strategy = init_strategy;
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

            pub fn with_block_count(mut self, block_count: usize) -> Self {
                self.block_count = block_count;
                self
            }

            pub fn with_head_count(mut self, head_count: usize) -> Self {
                self.head_count = head_count;
                self
            }
        }
    }
}

pub mod blocks {
    use anyhow::Result;

    use super::{
        layers::{DropoutLayer, FeedForwardLayer, LayerNormalization, MultiHeadSelfAttentionLayer},
        linear::Linear,
    };

    pub struct EncoderBlock {
        attention: MultiHeadSelfAttentionLayer,
        network: FeedForwardLayer,
        dropout: (DropoutLayer, DropoutLayer),
        layer_norm: (LayerNormalization, LayerNormalization),
    }

    impl EncoderBlock {
        pub fn new(sequence_len: usize, embedding_dimension: usize, head_count: usize) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, head_count).build()
        }

        pub fn new_builder(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
        ) -> builder::EncoderBlockBuilder {
            builder::EncoderBlockBuilder::new(sequence_len, embedding_dimension, head_count)
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
        fn encoder_block_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;

            let rng = RngStrategy::testable(1234);

            let encoder_block = EncoderBlock::new_builder(seq_len, embed_dim, head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng.clone())
                .build();

            let inputs = new_inputs(seq_len, embed_dim, &rng);
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
            EncoderBlock,
        };

        use crate::ml::{
            layer::LayerInitStrategy, transformer::layers::MultiHeadSelfAttentionLayer, NodeValue,
            RngStrategy,
        };

        #[derive(Debug, Clone)]
        pub struct EncoderBlockBuilder {
            pub(crate) sequence_len: usize,
            pub(crate) model_dimension: usize,
            pub(crate) head_count: usize,
            pub(crate) rng: RngStrategy,
            pub(crate) feed_forward_init_strategy: LayerInitStrategy,
            pub(crate) feed_forward_hidden_dimension: usize,
            pub(crate) dropout_rate: NodeValue,
        }

        impl EncoderBlockBuilder {
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

            pub fn build(self) -> EncoderBlock {
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

                EncoderBlock {
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
    use anyhow::{Context, Result};
    use itertools::Itertools;

    use crate::ml::NetworkActivationMode;

    use super::{attention::SelfAttentionHead, gradients::LossGradients, *};

    pub struct MultiHeadSelfAttentionLayer {
        heads: Vec<SelfAttentionHead>,
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
                SelfAttentionHead::new_head(sequence_len, embedding_dimension, head_count, rng)
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
                let dense_output = self.dense_layer.forward(&row.into())?;
                dense_layer_outputs.push(dense_output);
            }

            Linear::from_values(&dense_layer_outputs)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let mut dense_layer_input_gradients = vec![];

            for (row, inputs) in output_gradients.rows_iter().zip(inputs.rows_iter()) {
                let dense_output = self
                    .dense_layer
                    .backward(Some(&inputs.into()), row.into())?;

                dense_layer_input_gradients.push(dense_output);
            }

            let dense_layer_input_gradient = Linear::from_values(&dense_layer_input_gradients)?;
            let head_output_gradients = dense_layer_input_gradient.split(self.heads.len());

            let mut inputs_grads: Option<Linear> = None;

            for (head, output_gradients) in self.heads.iter_mut().zip(&head_output_gradients) {
                let (k, q, v) = head.backward(&inputs, &inputs, &inputs, &output_gradients)?;
                let head_input_gradients = k.iter().add(q.iter()).add(v.iter());

                inputs_grads = match &mut inputs_grads {
                    Some(inputs_grads) => {
                        Some(head_input_gradients.add(inputs_grads.iter()).collect())
                    }
                    _ => Some(head_input_gradients.collect()),
                };
            }

            Ok(inputs_grads.unwrap())
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            self.dense_layer.apply_gradients(learn_rate);

            for head in self.heads.iter_mut() {
                head.apply_gradients(learn_rate);
            }
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
                let layer_input = row.into();
                let hidden_layer = self.hidden_layer.forward(&layer_input)?;
                let layer_output = self.output_layer.forward(&hidden_layer)?;
                dense_layer_outputs.push(layer_output);
            }

            Linear::from_values(&dense_layer_outputs)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let mut output_layer_input_gradients = vec![];

            for (row, inputs) in output_gradients.rows_iter().zip(inputs.rows_iter()) {
                let output_gradients = row.into();
                let final_layer_gradients = self.output_layer.backward(None, output_gradients)?;

                let dense_output = self
                    .hidden_layer
                    .backward(Some(&inputs.into()), final_layer_gradients)?;

                output_layer_input_gradients.push(dense_output);
            }

            Linear::from_values(&output_layer_input_gradients)
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            self.hidden_layer.apply_gradients(learn_rate);
            self.output_layer.apply_gradients(learn_rate);
        }
    }

    pub struct EmbeddingLayer {
        embeddings: Vec<Linear>,
        vocab_size: usize,
        model_dimensions: usize,
        gradients: Option<LossGradients>,
    }

    impl EmbeddingLayer {
        pub fn new(
            model_dimensions: usize,
            vocab_size: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let embeddings = iter::repeat_with(|| {
                let mut linear = Linear::new(1, model_dimensions);
                linear.initialize_as_layer(strategy, rng);
                linear
            })
            .take(vocab_size)
            .collect();

            Self {
                embeddings,
                vocab_size,
                model_dimensions,
                gradients: None,
            }
        }

        pub fn forward<T: AsRef<[usize]>>(&self, token_sequence: T) -> Result<Linear> {
            let mut embedding_vectors = vec![];

            for &token in token_sequence.as_ref() {
                let embedding = self.embeddings.get(token).context("invalid token id")?;
                let embedding_vector = embedding.rows_iter().next().unwrap();
                embedding_vectors.push(embedding_vector.into());
            }

            Linear::from_values(&embedding_vectors)
        }

        pub fn backward<T: AsRef<[usize]>>(
            &mut self,
            token_sequence: T,
            output_gradients: &Linear,
        ) -> Result<()> {
            let mut embedding_weight_gradients = vec![];

            for (&token, grad) in token_sequence
                .as_ref()
                .iter()
                .zip(output_gradients.rows_iter())
            {
                let embedding = self.embeddings.get(token).context("invalid token id")?;
                let weight_gradients = embedding.backward(&grad.into())?;
                embedding_weight_gradients.push((token, weight_gradients));
            }

            self.add_gradients(embedding_weight_gradients);
            Ok(())
        }

        fn add_gradients(&mut self, embedding_weight_gradients: Vec<(usize, LayerValues)>) {
            let gradients =
                self.gradients
                    .get_or_insert_with(|| gradients::LossGradients::Embedding {
                        dweights: self
                            .embeddings
                            .iter()
                            .map(|_| Linear::new(1, self.model_dimensions))
                            .collect(),
                    });

            if let gradients::LossGradients::Embedding { dweights } = gradients {
                for (token_idx, gradients) in &embedding_weight_gradients
                    .into_iter()
                    .map(|(idx, grads)| (idx, Linear::from_values(&[grads]).unwrap()))
                    .group_by(|(idx, ..)| *idx)
                {
                    let dweights = &mut dweights[token_idx];
                    let gradients = gradients.collect_vec();

                    *dweights = gradients
                        .iter()
                        .fold(dweights.iter(), |iter, grads| iter.add(grads.1.iter()))
                        .collect();
                }
            }
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            // could mutate in-place these values back to zero rather than take
            let gradients = self.gradients.take();
            if let Some(gradients::LossGradients::Embedding { dweights }) = gradients {
                for (embedding, gradient) in self.embeddings.iter_mut().zip(&dweights) {
                    gradient_descent(embedding, &gradient, learn_rate);
                }
            }

            fn gradient_descent(target: &mut Linear, gradient: &Linear, learn_rate: NodeValue) {
                *target = target
                    .iter()
                    .sub(gradient.iter().multiply_scalar(learn_rate))
                    .collect();
            }
        }

        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    pub struct PositionalEmbeddingLayer {
        embeddings: EmbeddingLayer,
    }

    impl PositionalEmbeddingLayer {
        pub fn new(
            model_dimensions: usize,
            max_sequence_len: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            Self {
                embeddings: EmbeddingLayer::new(model_dimensions, max_sequence_len, strategy, &rng),
            }
        }

        pub fn forward(&self, start_index: usize, count: usize) -> Result<Linear> {
            let end_index = start_index + count;
            if end_index > self.embeddings.vocab_size() {
                return Err(anyhow!("positional index out of range"))?;
            }

            let sequence = start_index..end_index;
            self.embeddings.forward(&sequence.collect_vec())
        }

        pub fn backward<T: AsRef<[usize]>>(
            &mut self,
            start_index: usize,
            count: usize,
            output_gradients: &Linear,
        ) -> Result<()> {
            let end_index = start_index + count;
            if end_index > self.embeddings.vocab_size() {
                return Err(anyhow!("positional index out of range"))?;
            }
            if output_gradients.count() != count {
                return Err(anyhow!("mismatched gradient vector count"))?;
            }

            let sequence = start_index..end_index;
            self.embeddings
                .backward(&sequence.collect_vec(), output_gradients)
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            self.embeddings.apply_gradients(learn_rate);
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
                .sub(mean.iter().grow(input.stride()))
                .div(std_dev.iter().grow(input.stride()))
                .collect();

            Ok(norm)
        }
    }
}

pub mod attention {
    use anyhow::{anyhow, Result};
    use itertools::Itertools;

    use crate::ml::{layer::LayerInitStrategy, NodeValue, RngStrategy};

    use super::{
        gradients::{self, LossGradients},
        linear::Linear,
    };

    pub struct SelfAttentionHead {
        key_weights: Linear,
        query_weights: Linear,
        value_weights: Linear,
        mask: Option<Linear>,
        embedding_dimension: usize,
        sequence_len: usize,
        gradients: Option<LossGradients>,
    }

    impl SelfAttentionHead {
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
                key_weights: Self::new_kqv_linear(embedding_dimension, head_dimension, rng),
                query_weights: Self::new_kqv_linear(embedding_dimension, head_dimension, rng),
                value_weights: Self::new_kqv_linear(embedding_dimension, head_dimension, rng),
                mask: None,
                embedding_dimension,
                sequence_len,
                gradients: None,
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
            query_inputs: &Linear,
            value_inputs: &Linear,
        ) -> Result<Linear> {
            let keys = key_inputs.matrix_product(&self.key_weights);
            let queries = query_inputs.matrix_product(&self.query_weights);
            let values = value_inputs.matrix_product(&self.value_weights);
            let mask = self.mask.as_ref();

            let output = SelfAttentionHead::scaled_self_attention(&keys, &queries, &values, mask);

            Ok(output)
        }

        pub fn backward(
            &mut self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            if output_gradients.stride() != self.value_weights.stride() {
                Err(anyhow!("mismatched ouput vector size"))?;
            }

            let keys = key_inputs.matrix_product(&self.key_weights);
            let queries = query_inputs.matrix_product(&self.query_weights);
            let values = value_inputs.matrix_product(&self.value_weights);

            let (dkeys, dqueries, dvalues) =
                Self::scaled_self_attention_d(&keys, &queries, &values, &output_gradients)?;

            let dkey_weights = key_inputs.matrix_product_lhs_transposed(&dkeys);
            let dquery_weights = query_inputs.matrix_product_lhs_transposed(&dqueries);
            let dvalue_weights = value_inputs.matrix_product_lhs_transposed(&dvalues);

            self.add_gradients((dkey_weights, dquery_weights, dvalue_weights));

            let dkey_inputs = dkeys.matrix_product_rhs_transposed(&self.key_weights);
            let dquery_inputs = dqueries.matrix_product_rhs_transposed(&self.query_weights);
            let dvalue_inputs = dvalues.matrix_product_rhs_transposed(&self.value_weights);

            Ok((dkey_inputs, dvalue_inputs, dquery_inputs))
        }

        fn scaled_self_attention(
            keys: &Linear,
            queries: &Linear,
            values: &Linear,
            mask: Option<&Linear>,
        ) -> Linear {
            let attention_scores = queries.iter().matrix_transpose_product(keys.iter());

            let scaled_attention_scores = attention_scores
                .iter()
                .multiply_scalar(1.0 / keys.count() as NodeValue);

            let masked_attention_scores = match mask {
                Some(mask) => {
                    scaled_attention_scores.set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                }
                None => scaled_attention_scores,
            };

            let attention_weights = masked_attention_scores.softmax();
            attention_weights.matrix_product(&values)
        }

        fn scaled_self_attention_d(
            keys: &Linear,
            queries: &Linear,
            values: &Linear,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            // forward pass
            let attention_scores = queries.matrix_product_rhs_transposed(&keys);

            let scaled_attention_scores = attention_scores
                .iter()
                .multiply_scalar(1.0 / keys.count() as NodeValue);

            // TODO: figure out whats happening with the mask etc.
            // let masked_attention_scores = match mask {
            //     Some(mask) => {
            //         scaled_attention_scores.set_mask(mask.iter(), NodeValue::NEG_INFINITY)
            //     }
            //     None => scaled_attention_scores,
            // };
            let masked_attention_scores = scaled_attention_scores;

            let attention_weights = masked_attention_scores.softmax();
            // let output = attention_weights.matrix_product(&values);

            // backward pass
            let dvalues = attention_weights.matrix_product_lhs_transposed(&output_gradients);
            let dattention_weights = output_gradients.matrix_product_rhs_transposed(&values);
            let dmasked_attention_scores = softmax_d_iter(dattention_weights, attention_weights)?;
            // let dmasked_attention_scores = dattention_weights.softmax_d(&attention_weights);

            let dscaled_attention_scores = dmasked_attention_scores;
            let dattention_scores = dscaled_attention_scores
                .iter()
                .multiply_scalar(keys.count() as NodeValue)
                .collect();

            let dqueries = dattention_scores.matrix_product(&keys);
            let dkeys = dattention_scores.matrix_product_lhs_transposed(&queries);

            Ok((dkeys, dqueries, dvalues))
        }

        fn add_gradients(&mut self, kqv_weight_gradients: (Linear, Linear, Linear)) {
            let (dkey_weights, dquery_weights, dvalue_weights) = kqv_weight_gradients;

            let gradients =
                self.gradients
                    .get_or_insert_with(|| gradients::LossGradients::AttentionHead {
                        dkeys: Linear::with_dimensions(&dkey_weights),
                        dqueries: Linear::with_dimensions(&dquery_weights),
                        dvalues: Linear::with_dimensions(&dvalue_weights),
                    });

            if let gradients::LossGradients::AttentionHead {
                dkeys,
                dqueries,
                dvalues,
            } = gradients
            {
                *dkeys = dkeys.iter().add(dkey_weights.iter()).collect();
                *dqueries = dqueries.iter().add(dquery_weights.iter()).collect();
                *dvalues = dvalues.iter().add(dvalue_weights.iter()).collect();
            }
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            // could mutate in-place these values back to zero rather than take
            let gradients = self.gradients.take();
            if let Some(gradients::LossGradients::AttentionHead {
                dkeys,
                dqueries,
                dvalues,
            }) = gradients
            {
                gradient_descent(&mut self.key_weights, &dkeys, learn_rate);
                gradient_descent(&mut self.query_weights, &dqueries, learn_rate);
                gradient_descent(&mut self.value_weights, &dvalues, learn_rate);
            }

            fn gradient_descent(target: &mut Linear, gradient: &Linear, learn_rate: NodeValue) {
                *target = target
                    .iter()
                    .sub(gradient.iter().multiply_scalar(learn_rate))
                    .collect();
            }
        }

        fn new_kqv_linear(
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

    fn softmax_d_iter(dloss_dsoftmax: Linear, softmax: Linear) -> Result<Linear> {
        // let attention_weights = masked_attention_scores.softmax();
        // dL/dx_i = sum(dL/dy_j * softmax(x_j) * (delta_ij - softmax(x_i)))   for i = 1, ..., n and j = 1, ..., n
        let softmax_derivative = softmax
            .rows_iter()
            .zip(dloss_dsoftmax.rows_iter())
            .flat_map(|(probs, grads)| {
                let dot_product = probs
                    .iter()
                    .zip(grads)
                    .map(|(prob_j, grad_j)| prob_j * grad_j)
                    .sum::<f64>();

                probs
                    .iter()
                    .zip(grads)
                    .map(move |(prob_i, grad_i)| prob_i * (grad_i - dot_product))
            });

        Ok(Linear::from_iter(softmax.stride(), softmax_derivative)?)
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
            let attention_layer = SelfAttentionHead::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn attention_can_compute_outputs_multi_head_layer() {
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

            let output = SelfAttentionHead::scaled_self_attention(&queries, &keys, &values, None);

            assert_eq!(output.count(), seq_len);
            assert_eq!(output.stride(), embed_dim);
        }

        #[test]
        fn attention_can_compute_gradients_single_head() {
            let seq_len = 5;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_inputs(seq_len, embed_dim, &rng);
            let inputs = Linear::from_values(&inputs).unwrap();

            let mut attention_layer = SelfAttentionHead::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            let output_gradients = &new_linear(seq_len, embed_dim, &rng);
            let output_gradients = output_gradients.iter().multiply_scalar(0.01).collect();

            let grads = attention_layer
                .backward(&inputs, &inputs, &inputs, &output_gradients)
                .unwrap();

            let grads = grads.0.iter().add(grads.1.iter()).add(grads.2.iter());
            let grads = grads.collect().to_values();

            assert_eq!(grads.len(), output.len());
            assert_eq!(grads[0].len(), output[0].len());
        }

        #[test]
        fn attention_can_minimise_single_head() {
            let seq_len = 3;
            let embed_dim = 12;

            // setup attention head
            let rng = RngStrategy::testable(12345);
            let seed_inputs = new_inputs(seq_len, embed_dim, &rng);
            let mut inputs = Linear::from_values(&seed_inputs).unwrap();
            let mut attention_head = SelfAttentionHead::new(seq_len, embed_dim, &rng);

            // setup optimisation parameters
            let target = &new_linear(seq_len, embed_dim, &rng);
            let learn_rate = 0.001;
            let total_iterations = 50;
            let mut iteration = 0;
            let mut initial_mean_loss = None;

            let (grads, output, mean_grads, mean_loss) = loop {
                // compute forward pass
                let output = attention_head.forward(&inputs).unwrap();

                // calculate simple rms loss and gradients compared to target values
                let loss = output.iter().sub(target.iter()).powf_scalar(2.0);
                let dloss = output.iter().sub(target.iter()).multiply_scalar(2.0);

                // compute backward pass and return kqv gradients
                let grads = attention_head
                    .backward(&inputs, &inputs, &inputs, &dloss.collect())
                    .unwrap();

                // sum gradients for each kqv 'inputs' value provided
                let grads = grads.0.iter().add(grads.1.iter()).add(grads.2.iter());
                let grads = grads.collect();

                // apply grad descent
                attention_head.apply_gradients(learn_rate);
                let scaled_gradients = grads.iter().multiply_scalar(learn_rate);
                inputs = inputs.iter().sub(scaled_gradients).collect();

                // get gradients & loss for reporting
                let abs_grads = grads.iter().abs();
                let mean_grads = abs_grads.flatten_mean().iter_transpose().flatten_mean();
                let mean_grads = *mean_grads.values_iter().next().unwrap().0;
                let mean_loss = loss.flatten_mean().iter_transpose().flatten_mean();
                let mean_loss = *mean_loss.values_iter().next().unwrap().0;
                initial_mean_loss.get_or_insert(mean_loss);

                // report optimisation iteration
                if (iteration + 1) % (total_iterations / 10) == 0 || iteration == 0 {
                    println!(
                        "optimisation iter: {:<7} | loss: {:<10.5} | mean abs gradient: {:<5.2}",
                        iteration + 1,
                        mean_loss,
                        mean_grads
                    );
                }

                // progress iteration counter, or complete
                iteration += 1;
                if iteration > total_iterations {
                    break (grads, output, mean_grads, mean_loss);
                }
            };

            let grads = grads.to_values();
            let output = output.to_values();
            let initial_mean_loss = initial_mean_loss.unwrap();

            assert_eq!(grads.len(), output.len());
            assert_eq!(grads[0].len(), output[0].len());
            assert!(mean_grads.is_finite(), "final gradient has invalid value");
            assert!(mean_loss.is_finite(), "final loss has invalid value");

            assert!(
                initial_mean_loss > mean_loss,
                "loss failed to optimise (start={:.2}, end={:.2})",
                initial_mean_loss,
                mean_loss
            );
        }

        fn new_linear(seq_len: usize, embedding_dimension: usize, rng: &RngStrategy) -> Linear {
            let mut linear = Linear::new(seq_len, embedding_dimension);
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

pub mod gradients {
    use super::linear::Linear;

    #[derive(Debug, PartialEq)]
    pub enum LossGradients {
        Dense {
            weights: Linear,
            bias: Option<Linear>,
        },
        AttentionHead {
            dkeys: Linear,
            dvalues: Linear,
            dqueries: Linear,
        },
        Embedding {
            dweights: Vec<Linear>,
        },
    }
}

pub mod dense {
    use anyhow::{anyhow, Context, Result};

    use crate::ml::{
        layer::LayerInitStrategy, LayerValues, NetworkActivationMode, NodeValue, RngStrategy,
    };

    use super::{
        gradients,
        linear::{Linear, LinearIter},
    };

    #[derive(Debug, PartialEq)]
    pub struct Dense {
        weights: Linear,
        bias: Option<Linear>,
        activation: Option<NetworkActivationMode>,
        inputs_count: usize,
        outputs_count: usize,
        gradients: Option<gradients::LossGradients>,
    }

    impl Dense {
        pub fn new(
            inputs_count: usize,
            outputs_count: usize,
            strategy: &LayerInitStrategy,
            rng: &RngStrategy,
        ) -> Self {
            let mut weights = Linear::new(inputs_count, outputs_count);
            weights.initialize_as_layer(strategy, rng);

            let bias = if strategy.requires_bias() {
                let mut b = Linear::new(1, outputs_count);
                b.initialize_as_layer_bias(strategy, rng);
                Some(b)
            } else {
                None
            };

            Self {
                weights,
                bias,
                activation: None,
                inputs_count,
                outputs_count,
                gradients: None,
            }
        }

        pub fn forward(&self, inputs: &LayerValues) -> Result<LayerValues> {
            if inputs.len() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }

            let weighted_inputs = self.compute_weighted_inputs(inputs)?;

            let outputs = match &self.activation {
                Some(activation) => activation.apply(&weighted_inputs),
                None => weighted_inputs,
            };

            Ok(outputs)
        }

        fn compute_weighted_inputs(&self, inputs: &LayerValues) -> Result<LayerValues> {
            // use Linear/LinearIter matrix ops directly
            let mut weighted_inputs = self.weights.forward(&inputs)?;
            if let Some(bias) = &self.bias {
                weighted_inputs
                    .iter_mut()
                    .zip(bias.values_iter())
                    .for_each(|(x, (b, ..))| *x += b);
            }
            Ok(weighted_inputs)
        }

        pub fn backward(
            &mut self,
            inputs: Option<&LayerValues>,
            output_gradients: LayerValues,
        ) -> Result<LayerValues> {
            let weighted_inputs_gradients = match &self.activation {
                Some(activation) => {
                    let inputs = inputs
                        .context("require inputs to calculate gradients for layer activation")?;
                    let weighted_inputs = self.compute_weighted_inputs(inputs)?;

                    activation
                        .derivative(&weighted_inputs)
                        .multiply_iter(&output_gradients)
                        .collect()
                }
                None => output_gradients,
            };

            // TODO: fix impl. use matrix math with Linear/LinearIter directly
            let input_gradients = self.weights.backward(&weighted_inputs_gradients)?;

            let bias_gradients = Linear::from_values(&[weighted_inputs_gradients])?;
            let weights_gradients = bias_gradients.iter().stack(self.weights.count());
            self.add_gradients(weights_gradients, bias_gradients.iter());

            Ok(input_gradients)
        }

        pub fn set_activation(&mut self, activation: NetworkActivationMode) {
            self.activation = Some(activation);
        }

        fn add_gradients(&mut self, weight_gradients: LinearIter, bias_gradients: LinearIter) {
            let gradients = self
                .gradients
                .get_or_insert_with(|| gradients::LossGradients::Dense {
                    weights: Linear::with_dimensions(&self.weights),
                    bias: self
                        .bias
                        .as_ref()
                        .map(|bias| Linear::with_dimensions(&bias)),
                });

            if let gradients::LossGradients::Dense { weights, bias } = gradients {
                *weights = weights.iter().add(weight_gradients).collect();

                if let Some(bias) = bias {
                    *bias = bias.iter().add(bias_gradients).collect();
                }
            }
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            // could mutate in-place these values back to zero rather than take
            let gradients = self.gradients.take();
            if let Some(gradients::LossGradients::Dense { weights, bias }) = gradients {
                match (bias, &mut self.bias) {
                    (Some(bias_gradient), Some(bias)) => {
                        gradient_descent(bias, &bias_gradient, learn_rate);
                        gradient_descent(&mut self.weights, &weights, learn_rate);
                    }
                    _ => {
                        gradient_descent(&mut self.weights, &weights, learn_rate);
                    }
                }
            }

            fn gradient_descent(target: &mut Linear, gradient: &Linear, learn_rate: NodeValue) {
                *target = target
                    .iter()
                    .sub(gradient.iter().multiply_scalar(learn_rate))
                    .collect();
            }
        }
    }
}

pub mod linear {
    use std::iter;

    use anyhow::{anyhow, Context, Result};
    use itertools::Itertools;

    use crate::ml::{layer::LayerInitStrategy, LayerValues, NodeValue, RngStrategy};

    #[derive(Debug, PartialEq)]
    pub struct Linear {
        inner: LayerValues,
        stride: usize,
        count: usize,
    }

    impl Linear {
        pub fn new(count: usize, stride: usize) -> Self {
            let size = count * stride;
            Self {
                inner: LayerValues::new(vec![0.0; size]),
                stride,
                count,
            }
        }

        pub fn with_dimensions(other: &Self) -> Self {
            Self {
                inner: LayerValues::new(vec![0.0; other.inner.len()]),
                stride: other.stride,
                count: other.count,
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

        // TODO remove in favour of Dense::forward
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

        // TODO remove in favour of Dense::backward
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

        pub fn split(&self, n: usize) -> Vec<Self> {
            assert_eq!(self.stride % n, 0, "mismatched dimensions");
            let stride = self.stride / n;
            (0..n)
                .map(|i| {
                    let self_items = self.inner.chunks(self.stride);
                    Self {
                        inner: self_items
                            .flat_map(|row| row.into_iter().skip(i * stride).take(stride))
                            .copied()
                            .collect(),
                        stride,
                        count: self.count,
                    }
                })
                .collect()
        }

        pub fn softmax_d(&self, softmax_outputs: &Linear) -> Linear {
            self.iter()
                .sub(
                    softmax_outputs
                        .iter()
                        .dot_product(self.iter())
                        .flatten_sum()
                        .iter()
                        .grow(softmax_outputs.stride()),
                )
                .dot_product(softmax_outputs.iter())
                .collect()
        }

        pub fn matrix_product(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.stride, rhs.count, "mismatched dimensions");
            self.iter().matrix_transpose_product(rhs.iter_transpose())
        }

        pub fn matrix_product_rhs_transposed(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.stride, rhs.stride, "mismatched dimensions");
            self.iter().matrix_transpose_product(rhs.iter())
        }

        pub fn matrix_product_lhs_transposed(&self, rhs: &Linear) -> Linear {
            assert_eq!(self.count, rhs.count, "mismatched dimensions");
            self.iter_transpose()
                .matrix_transpose_product(rhs.iter_transpose())
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

        pub fn rows_iter_mut(&mut self) -> impl Iterator<Item = &mut [NodeValue]> {
            self.inner.chunks_mut(self.stride)
        }

        pub fn values_iter(&self) -> impl Iterator<Item = (&NodeValue, usize, usize)> {
            self.inner
                .iter()
                .enumerate()
                .map(|(idx, x)| (x, idx % self.stride, idx / self.stride))
        }
    }

    impl std::fmt::Display for Linear {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.rows_iter()
                .fold(&mut f.debug_list(), |list, row| list.entry(&row))
                .finish()
        }
    }

    pub struct LinearIter<'a> {
        inner: Box<dyn Iterator<Item = NodeValue> + 'a>,
        stride: usize,
        count: usize,
    }

    impl<'a> LinearIter<'a> {
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
        pub fn dot_product(self, other: Self) -> Self {
            assert_eq!(self.stride, other.stride, "mismatched dimensions");
            assert_eq!(self.count, other.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(other.inner).map(|(x, y)| x * y)),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn div(self, rhs: Self) -> Self {
            assert_eq!(self.stride, rhs.stride, "mismatched dimensions");
            assert_eq!(self.count, rhs.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(rhs.inner).map(|(x, y)| x / y)),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn add(self, other: Self) -> Self {
            assert_eq!(self.stride, other.stride, "mismatched dimensions");
            assert_eq!(self.count, other.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(other.inner).map(|(x, y)| x + y)),
                stride: self.stride,
                count: self.count,
            }
        }
        pub fn sub(self, rhs: Self) -> Self {
            assert_eq!(self.stride, rhs.stride, "mismatched dimensions");
            assert_eq!(self.count, rhs.count, "mismatched dimensions");
            Self {
                inner: Box::new(self.inner.zip(rhs.inner).map(|(x, y)| x - y)),
                stride: self.stride,
                count: self.count,
            }
        }
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
        pub fn abs(self) -> Self {
            Self {
                inner: Box::new(self.inner.map(move |x| x.abs())),
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
                stride,
                count: self.count,
            }
        }
        pub fn stack(self, count: usize) -> Self {
            assert_eq!(self.count, 1, "can only stack when count dimension = 1");
            assert_ne!(count, 0, "invalid stride dimension");
            Self {
                inner: Box::new(
                    self.inner
                        .collect_vec()
                        .into_iter()
                        .cycle()
                        .take(self.stride * count),
                ),
                stride: self.stride,
                count,
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
        fn can_linear_perform_sub_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 4.0, 3.0, 2.0].into_iter()).unwrap();
            let y = a.iter().sub(b.iter()).collect();

            let expected = Linear::from_iter(2, [-4.0, -2.0, 0.0, 2.0].into_iter()).unwrap();
            assert_eq!(expected, y);
        }

        #[test]
        fn can_linear_perform_div_pointwise() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let b = Linear::from_iter(2, [5.0, 4.0, 3.0, 2.0].into_iter()).unwrap();
            let y = a.iter().div(b.iter()).collect();

            let expected = Linear::from_iter(2, [0.2, 0.5, 1.0, 2.0].into_iter()).unwrap();
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
        fn can_linear_sum_rows_and_keep_shape() {
            let x = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let y = x.iter().flatten_sum().iter().grow(2).collect();

            let expected = Linear::from_iter(2, [3.0, 3.0, 7.0, 7.0].into_iter()).unwrap();
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
        fn can_linear_perform_mean_all_values_and_keep_shape() {
            let a = Linear::from_iter(2, [1.0, 2.0, 3.0, 4.0].into_iter()).unwrap();
            let mean_a = a.iter().flatten_mean().iter_transpose().flatten_mean();
            let y = mean_a.iter().grow(a.stride()).stack(a.count()).collect();

            let expected = Linear::from_iter(2, [2.5, 2.5, 2.5, 2.5].into_iter()).unwrap();
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
