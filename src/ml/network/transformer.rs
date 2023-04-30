use std::iter;

use anyhow::anyhow;

use crate::ml::RngStrategy;

use self::{dense::Dense, linear::Linear};

use super::{layer::LayerInitStrategy, NodeValue};

pub mod encoder {
    use anyhow::Result;

    use crate::ml::NodeValue;

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
        sequence_length: usize,
        padding_token: usize,
    }

    impl Encoder {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            source_vocab_size: usize,
        ) -> Self {
            Self::new_builder(sequence_len, embedding_dimension, source_vocab_size)
                .with_head_count(head_count)
                .build()
                .unwrap()
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
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = self.dropout.forward(&embeddings)?;

            let mut block_output = block_input;
            for block in &self.blocks {
                let block_input = &block_output;
                block_output = block.forward_training_with_mask(&block_input, mask)?;
            }

            Ok(block_output)
        }

        pub fn backward<T: AsRef<[usize]>>(
            &mut self,
            input_sequence: T,
            output_gradients: Linear,
        ) -> Result<()> {
            // forawrd pass
            let input_sequence = input_sequence.as_ref();
            let (input_sequence, mask) = self.apply_input_padding(input_sequence);
            let token_embeddings = self.token_embedding.forward(&input_sequence)?;
            let position_embeddings = self.position_embedding.forward(0, input_sequence.len())?;
            let mask = Some(&mask);

            let embeddings = token_embeddings
                .iter()
                .add(position_embeddings.iter())
                .collect();
            let block_input = self.dropout.forward(&embeddings)?;

            let mut block_inputs = vec![block_input];
            for block in &self.blocks {
                let block_input = block_inputs.last().unwrap();
                let block_output = block.forward_training_with_mask(&block_input, mask)?;
                block_inputs.push(block_output);
            }

            // backward pass
            let mut block_output_gradients = output_gradients;
            for (block, inputs) in self.blocks.iter_mut().zip(block_inputs.iter()).rev() {
                let block_input_gradients =
                    block.backward_with_mask(&inputs, mask, &block_output_gradients)?;
                block_output_gradients = block_input_gradients;
            }

            let d_embeddings = block_output_gradients;
            let d_token_embeddings = &d_embeddings;
            let d_position_embeddings = &d_embeddings;

            self.token_embedding
                .backward(&input_sequence, &d_token_embeddings)?;

            self.position_embedding
                .backward(0, input_sequence.len(), &d_position_embeddings)?;

            Ok(())
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            for block in self.blocks.iter_mut() {
                block.apply_gradients(learn_rate);
            }
            self.token_embedding.apply_gradients(learn_rate);
            self.position_embedding.apply_gradients(learn_rate);
        }

        fn apply_input_padding(&self, input_sequence: &[usize]) -> (Vec<usize>, Linear) {
            EmbeddingLayer::pad_inputs(input_sequence, self.sequence_length, self.padding_token)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::tests::helpers::{assert_optimisation_converges, new_linear},
            RngStrategy,
        };

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
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            let output = encoder.forward_training(&inputs).unwrap();
            println!("outputs -> {output:#}");

            let output = output.to_values();
            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn encoder_can_process_inputs_shorter_than_seq_len() {
            let seq_len = 4;
            let embed_dim = 12;
            let head_count = 3;
            let hidden_dim = 48;
            let vocab_size = 10;

            let rng = RngStrategy::testable(1234);

            let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                .with_head_count(head_count)
                .with_feed_forward_hidden_dimension(hidden_dim)
                .with_rng(rng)
                .build()
                .unwrap();

            let inputs = [3, 6, 9];
            assert!(inputs.len() < seq_len);

            let output = encoder.forward_training(&inputs).unwrap();
            println!("outputs -> {output:#}");

            let output = output.to_values();
            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn encoder_can_minimise() {
            let seq_len = 3;
            let embed_dim = 6;
            let head_count = 2;
            let hidden_dim = 8;
            let vocab_size = 10;
            let block_count = 6;

            let learn_rate = 0.01;
            let iters = 25;
            let dummy_grads = Linear::new(1, 1);

            assert_optimisation_converges(
                &move |rng| {
                    let encoder = Encoder::new_builder(seq_len, embed_dim, vocab_size)
                        .with_head_count(head_count)
                        .with_block_count(block_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .with_dropout_rate(0.0) // TODO: reenable dropout once relate TODOs are completed
                        .build()
                        .unwrap();
                    let inputs = [3, 6, 9];
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (encoder, inputs, target)
                },
                &move |encoder, inputs| encoder.forward_training(&inputs).unwrap(),
                &move |encoder, inputs, dloss| {
                    encoder.backward(&inputs, dloss).unwrap();
                    encoder.apply_gradients(learn_rate);
                    dummy_grads.clone()
                },
                iters,
            );
        }
    }

    pub mod builder {
        use anyhow::{anyhow, Result};

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
            pub(crate) padding_token: usize,
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
                    padding_token: 0,
                    rng: Default::default(),
                    embedding_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_init_strategy: LayerInitStrategy::KaimingZeroBias,
                    feed_forward_hidden_dimension: 64,
                    dropout_rate: 0.1,
                }
            }

            pub fn build(self) -> Result<Encoder> {
                if self.model_dimension % self.block_count != 0 {
                    Err(anyhow!("block_count is not a factor of model_dimension"))?;
                }
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

                Ok(Encoder {
                    blocks,
                    token_embedding,
                    position_embedding,
                    dropout,
                    sequence_length: self.sequence_len,
                    padding_token: self.padding_token,
                })
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

            pub fn with_padding_token(mut self, padding_token: usize) -> Self {
                self.padding_token = padding_token;
                self
            }
        }
    }
}

pub mod blocks {
    use anyhow::Result;

    use crate::ml::NodeValue;

    use super::{
        layers::{DropoutLayer, FeedForwardLayer, LayerNormalization, MultiHeadSelfAttentionLayer},
        linear::Linear,
    };

    #[derive(Debug)]
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
            self.forward_training_with_mask(inputs, None)
        }

        pub fn forward_training_with_mask(
            &self,
            inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let attention_output = self.attention.forward_with_mask(&inputs, mask)?;
            let attention_output = self.dropout.0.forward(&attention_output)?;

            let skip_attention_output = attention_output.iter().add(inputs.iter()).collect();
            let ff_network_inputs = self.layer_norm.0.forward(&skip_attention_output)?;

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let ff_network_output = self.dropout.1.forward(&ff_network_output)?;

            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();
            let encoder_output = self.layer_norm.1.forward(&skip_ff_output)?;

            Ok(encoder_output)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            self.backward_with_mask(inputs, None, output_gradients)
        }

        pub fn backward_with_mask(
            &mut self,
            inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Linear> {
            // forward pass
            let attention_output = self.attention.forward_with_mask(&inputs, mask)?;
            let skip_attention_output = attention_output.iter().add(inputs.iter()).collect();
            let ff_network_inputs = self.layer_norm.0.forward(&skip_attention_output)?;

            let ff_network_output = self.network.forward(&ff_network_inputs)?;
            let skip_ff_output = ff_network_output
                .iter()
                .add(ff_network_inputs.iter())
                .collect();

            // backward pass
            let d_skip_ff_output = self
                .layer_norm
                .1
                .backward(&skip_ff_output, &output_gradients)?;

            // TODO: add dropouts backward pass
            let d_ff_network_output = &d_skip_ff_output;

            let ff_gradients = self
                .network
                .backward(&ff_network_inputs, &d_ff_network_output)?;

            let d_ff_network_inputs = d_skip_ff_output.iter().add(ff_gradients.iter()).collect();
            let d_skip_attention_output = self
                .layer_norm
                .0
                .backward(&skip_attention_output, &d_ff_network_inputs)?;
            let d_attention_output = &d_skip_attention_output;

            let attention_gradients =
                self.attention
                    .backward_with_mask(&inputs, mask, &d_attention_output)?;

            let d_inputs = d_skip_attention_output
                .iter()
                .add(attention_gradients.iter())
                .collect();

            Ok(d_inputs)
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            self.attention.apply_gradients(learn_rate);
            self.network.apply_gradients(learn_rate);
            self.layer_norm.0.apply_gradients(learn_rate);
            self.layer_norm.1.apply_gradients(learn_rate);
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::tests::helpers::{assert_optimisation_converges, new_linear},
            RngStrategy,
        };

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

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = encoder_block.forward_training(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn encoder_block_can_optimise() {
            let seq_len = 3;
            let embed_dim = 8;
            let head_count = 2;
            let hidden_dim = 24;

            let learn_rate = 0.01;
            let iters = 25;

            assert_optimisation_converges(
                &move |rng| {
                    let encoder_block = EncoderBlock::new_builder(seq_len, embed_dim, head_count)
                        .with_feed_forward_hidden_dimension(hidden_dim)
                        .with_rng(rng.clone())
                        .build();
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (encoder_block, inputs, target)
                },
                &move |encoder_block, inputs| encoder_block.forward_training(&inputs).unwrap(),
                &move |encoder_block, inputs, dloss| {
                    let grads = encoder_block.backward(&inputs, &dloss).unwrap();
                    encoder_block.apply_gradients(learn_rate);
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
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
                let layer_norm1 = LayerNormalization::new(self.sequence_len);
                let layer_norm2 = LayerNormalization::new(self.sequence_len);

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

    use super::{attention::MultiHeadAttention, gradients::LossGradients, *};

    #[derive(Debug)]
    pub struct MultiHeadSelfAttentionLayer {
        attention: MultiHeadAttention,
        dense_layer: Dense,
    }

    impl MultiHeadSelfAttentionLayer {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            rng: &RngStrategy,
        ) -> Self {
            let attention =
                MultiHeadAttention::new(sequence_len, embedding_dimension, head_count, rng);

            let dense_layer = Dense::new(
                embedding_dimension,
                embedding_dimension,
                &LayerInitStrategy::KaimingZeroBias,
                &rng,
            );

            Self {
                attention,
                dense_layer,
            }
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            self.forward_with_mask(inputs, None)
        }

        pub fn forward_with_mask(&self, inputs: &Linear, mask: Option<&Linear>) -> Result<Linear> {
            let attention_output = self.attention.forward(inputs, inputs, inputs, mask)?;
            self.dense_layer.forward(&attention_output)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            self.backward_with_mask(inputs, None, output_gradients)
        }

        pub fn backward_with_mask(
            &mut self,
            inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Linear> {
            // Forward pass
            let attention_output = self.attention.forward(inputs, inputs, inputs, mask)?;

            // Backward pass
            let dense_input_grads = self
                .dense_layer
                .backward(&attention_output, &output_gradients)?;

            let kqv_input_grads =
                self.attention
                    .backward(inputs, inputs, inputs, mask, &dense_input_grads)?;

            self.sum_kqv_gradients(inputs, kqv_input_grads)
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            self.dense_layer.apply_gradients(learn_rate);
            self.attention.apply_gradients(learn_rate);
        }

        fn sum_kqv_gradients(
            &self,
            inputs: &Linear,
            kqv_input_grads: Vec<(Linear, Linear, Linear)>,
        ) -> Result<Linear> {
            let zero: Linear = Linear::with_dimensions(inputs);
            let mut sum_inputs_grads = zero.iter();
            for (k_grads, q_grads, v_grads) in &kqv_input_grads {
                let sum = Linear::sum([k_grads, q_grads, v_grads].into_iter()).unwrap();
                sum_inputs_grads = sum_inputs_grads.add(sum);
            }
            Ok(sum_inputs_grads.collect())
        }
    }

    #[derive(Debug)]
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
                // TODO: switch to forward rather than forward_row
                let hidden_layer = self.hidden_layer.forward_row(layer_input)?;
                let layer_output = self.output_layer.forward_row(hidden_layer)?;
                dense_layer_outputs.push(layer_output);
            }

            Linear::from_values(&dense_layer_outputs)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let mut output_layer_input_gradients = vec![];

            for (row, inputs) in output_gradients.rows_iter().zip(inputs.rows_iter()) {
                // TODO: switch to forward rather than forward_row
                let final_layer_inputs = self.hidden_layer.forward_row(inputs.into())?;
                let output_gradients = row.into();

                let final_layer_gradients = self
                    .output_layer
                    .backward_row(final_layer_inputs, output_gradients)?;

                let dense_input_gradients = self
                    .hidden_layer
                    .backward_row(inputs.into(), final_layer_gradients)?;

                output_layer_input_gradients.push(dense_input_gradients);
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
                let grad = Linear::from_iter(grad.len(), grad.iter().copied()).unwrap();
                embedding_weight_gradients.push((token, grad));
            }

            self.add_gradients(embedding_weight_gradients);
            Ok(())
        }

        fn add_gradients(&mut self, embedding_weight_gradients: Vec<(usize, Linear)>) {
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
                    .map(|(idx, grads)| (idx, grads))
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

        pub fn pad_inputs(
            input_sequence: &[usize],
            seq_len: usize,
            pad_token: usize,
        ) -> (Vec<usize>, Linear) {
            let input_token_len = input_sequence.len();

            let mut input_sequence = input_sequence.to_vec();
            let mask = if input_token_len == seq_len {
                Linear::with_value(seq_len, 1, 1.0)
            } else {
                input_sequence.resize(seq_len, pad_token);
                let mask_iter = (0..seq_len).map(|i| if i < input_token_len { 1.0 } else { 0.0 });
                Linear::from_iter(1, mask_iter).unwrap()
            };

            (input_sequence, mask)
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

        pub fn backward(
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

    #[derive(Debug)]
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

        pub fn backward(&mut self, input: &Linear, output_gradients: &Linear) -> Result<Linear> {
            todo!("complete backprop or scale gradients at least")
        }
    }

    #[derive(Debug)]
    pub struct LayerNormalization {
        beta: Linear,
        gamma: Linear,
        gradients: Option<LossGradients>,
    }

    impl LayerNormalization {
        pub fn new(seq_len: usize) -> Self {
            Self {
                beta: Linear::with_value(seq_len, 1, 0.0),
                gamma: Linear::with_value(seq_len, 1, 1.0),
                gradients: None,
            }
        }

        pub fn forward(&self, input: &Linear) -> Result<Linear> {
            let stride = input.stride();
            let mean = input.iter().flatten_mean();
            let std_dev = input.iter().flatten_stddev(mean.iter());

            let normalised_input = input
                .iter()
                .sub(mean.iter().grow(stride))
                .div(std_dev.iter().grow(stride), Some(1e-8))
                .collect();

            let norm_scaled_shifted = normalised_input
                .iter()
                .dot_product(self.gamma.iter().grow(stride))
                .add(self.beta.iter().grow(stride))
                .collect();

            Ok(norm_scaled_shifted)
        }

        pub fn backward(&mut self, input: &Linear, output_gradients: &Linear) -> Result<Linear> {
            // forward pass
            let mean = input.iter().flatten_mean();
            let std_dev = input.iter().flatten_stddev(mean.iter());

            let stride = input.stride();
            let mean_keep_dim = mean.iter().grow(stride).collect();
            let std_dev_keep_dim = std_dev.iter().grow(stride).collect();
            let epsilon = Some(1e-8);

            let normalised_input = input
                .iter()
                .sub(mean_keep_dim.iter())
                .div(std_dev_keep_dim.iter(), epsilon)
                .collect();

            // backward pass
            let dbeta = output_gradients.iter().flatten_sum();
            let dgamma = output_gradients
                .iter()
                .dot_product(normalised_input.iter())
                .flatten_sum();
            self.add_gradients(dbeta, dgamma);

            // dl/dnorm(X) = dl/dY * gamma
            let dl_dnorm = output_gradients
                .iter()
                .dot_product(self.gamma.iter().grow(stride))
                .collect();

            let stride_inv = 1.0 / stride as NodeValue;

            // dl/dmean(X) = sum(dl/dY * gamma / (std(X) + eps), axis=1) / -sqrt(N)
            let dmean = dl_dnorm
                .iter()
                .flatten_sum()
                .iter()
                .div(std_dev.iter(), epsilon)
                .multiply_scalar(-stride_inv.sqrt())
                .collect(); // not needed?

            // dl/dstd(X) = sum(dl/dY * gamma * (X - mean(X))
            //     / (N * (std(X) + eps)^2), axis=1)
            //     * (-0.5 * (std(X) + eps)^-3)
            let dstd_dev = dl_dnorm
                .iter()
                .multiply_scalar(-1.0)
                .dot_product(normalised_input.iter())
                .flatten_sum()
                .iter()
                .div(std_dev.iter().powf_scalar(4.0), epsilon)
                .multiply_scalar(-0.5 * stride_inv)
                .collect(); // not needed?

            // dl/dX = (dl/dY * gamma) / (std(X) + eps)
            //     - (dl/dmean(X) / N)
            //     - (dl/dstd(X) * (X - mean(X)) / (N * (std(X) + eps)))
            let dx = dl_dnorm
                .iter()
                .div(std_dev_keep_dim.iter(), epsilon)
                .sub(dmean.iter().multiply_scalar(stride_inv).grow(stride))
                .sub(
                    dstd_dev
                        .iter()
                        .div(std_dev.iter(), epsilon)
                        .multiply_scalar(stride_inv)
                        .grow(stride)
                        .dot_product(input.iter().sub(mean_keep_dim.iter())),
                )
                .collect();

            Ok(dx)
        }

        fn add_gradients(&mut self, beta_gradients: Linear, gamma_gradients: Linear) {
            let gradients = self.gradients.get_or_insert_with(|| {
                gradients::LossGradients::LayerNormalization {
                    dbeta: Linear::with_dimensions(&beta_gradients),
                    dgamma: Linear::with_dimensions(&gamma_gradients),
                }
            });

            if let gradients::LossGradients::LayerNormalization { dbeta, dgamma } = gradients {
                *dbeta = dbeta.iter().add(beta_gradients.iter()).collect();
                *dgamma = dgamma.iter().add(gamma_gradients.iter()).collect();
            }
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            // could mutate in-place these values back to zero rather than take
            let gradients = self.gradients.take();
            if let Some(gradients::LossGradients::LayerNormalization { dbeta, dgamma }) = gradients
            {
                gradient_descent(&mut self.beta, &dbeta, learn_rate);
                gradient_descent(&mut self.gamma, &dgamma, learn_rate);
            }

            fn gradient_descent(target: &mut Linear, gradient: &Linear, learn_rate: NodeValue) {
                *target = target
                    .iter()
                    .sub(gradient.iter().multiply_scalar(learn_rate))
                    .collect();
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::ml::{
            transformer::tests::helpers::{assert_optimisation_converges, new_linear},
            RngStrategy,
        };

        use super::*;

        #[test]
        fn feed_forward_layer_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;
            let hidden_dim = 48;

            let rng = RngStrategy::testable(1234);
            let strategy = LayerInitStrategy::KaimingZeroBias;

            let feed_forward_layer = FeedForwardLayer::new(embed_dim, hidden_dim, &strategy, &rng);

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = feed_forward_layer.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn layer_normalization_layer_can_process_inputs() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let layer_norm = LayerNormalization::new(seq_len);

            let inputs = new_linear(seq_len, embed_dim, &rng);
            let output = layer_norm.forward(&inputs).unwrap().to_values();

            assert_eq!(output.len(), seq_len);
            assert_eq!(output[0].len(), embed_dim);
        }

        #[test]
        fn feed_forward_layer_can_optimise() {
            let seq_len = 3;
            let embed_dim = 12;
            let hidden_dim = 48;
            let strategy = LayerInitStrategy::KaimingZeroBias;

            let learn_rate = 0.01;
            let iters = 15;

            assert_optimisation_converges(
                &move |rng| {
                    let ff_layer = FeedForwardLayer::new(embed_dim, hidden_dim, &strategy, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (ff_layer, inputs, target)
                },
                &move |ff_layer, inputs| ff_layer.forward(&inputs).unwrap(),
                &move |ff_layer, inputs, dloss| {
                    let grads = ff_layer.backward(&inputs, &dloss).unwrap();
                    ff_layer.apply_gradients(learn_rate);
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
        }

        #[test]
        fn layer_normalization_layer_can_optimise_inputs() {
            let seq_len = 3;
            let embed_dim = 12;

            let learn_rate = 0.001;
            let iters = 500;

            assert_optimisation_converges(
                &move |rng| {
                    let layer_norm = LayerNormalization::new(seq_len);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (layer_norm, inputs, target)
                },
                &move |layer_norm, inputs| layer_norm.forward(&inputs).unwrap(),
                &move |layer_norm, inputs, dloss| {
                    let grads = layer_norm.backward(&inputs, &dloss).unwrap();
                    layer_norm.apply_gradients(learn_rate);
                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);

                    grads
                },
                iters,
            );
        }
    }
}

pub mod attention {
    use std::iter;

    use anyhow::{anyhow, Result};

    use crate::ml::{layer::LayerInitStrategy, NetworkActivationMode, NodeValue, RngStrategy};

    use super::{
        gradients::{self, LossGradients},
        linear::Linear,
    };

    #[derive(Debug)]
    pub struct MultiHeadAttention {
        heads: Vec<AttentionHead>,
    }

    impl MultiHeadAttention {
        pub fn new(
            sequence_len: usize,
            embedding_dimension: usize,
            head_count: usize,
            rng: &RngStrategy,
        ) -> Self {
            let heads = iter::repeat_with(|| {
                AttentionHead::new_head(sequence_len, embedding_dimension, head_count, rng)
            })
            .take(head_count)
            .collect();

            Self { heads }
        }

        pub fn forward(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let mut head_outputs = vec![];
            for head in &self.heads {
                let head_output =
                    head.forward_advanced(key_inputs, query_inputs, value_inputs, mask)?;
                head_outputs.push(head_output);
            }

            let attention_output = head_outputs
                .into_iter()
                .fold(None, |concat: Option<Linear>, head| match concat {
                    Some(iter) => Some(iter.concat(&head).collect()),
                    None => Some(head),
                })
                .unwrap();

            Ok(attention_output)
        }

        pub fn backward(
            &mut self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<Vec<(Linear, Linear, Linear)>> {
            let head_output_gradients = output_gradients.split(self.heads.len());

            let mut kqv_input_grads = vec![];

            for (head, output_gradients) in self.heads.iter_mut().zip(&head_output_gradients) {
                let head_input_gradients = head.backward_advanced(
                    key_inputs,
                    query_inputs,
                    value_inputs,
                    mask,
                    &output_gradients,
                )?;
                kqv_input_grads.push(head_input_gradients);
            }

            Ok(kqv_input_grads)
        }

        pub fn apply_gradients(&mut self, learn_rate: NodeValue) {
            for head in self.heads.iter_mut() {
                head.apply_gradients(learn_rate);
            }
        }
    }

    #[derive(Debug)]
    pub struct AttentionHead {
        key_weights: Linear,
        query_weights: Linear,
        value_weights: Linear,
        mask: Option<Linear>,
        embedding_dimension: usize,
        sequence_len: usize,
        gradients: Option<LossGradients>,
    }

    impl AttentionHead {
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

            let mask = self.mask.as_ref();
            self.forward_advanced(&inputs, &inputs, &inputs, mask)
        }

        pub fn forward_advanced(
            &self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
        ) -> Result<Linear> {
            let keys = key_inputs.matrix_product(&self.key_weights);
            let queries = query_inputs.matrix_product(&self.query_weights);
            let values = value_inputs.matrix_product(&self.value_weights);

            let mask = mask.or(self.mask.as_ref());
            let output = AttentionHead::scaled_attention(&keys, &queries, &values, mask);

            Ok(output)
        }

        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let kqv_grads =
                self.backward_advanced(&inputs, &inputs, &inputs, None, output_gradients)?;

            let (k_grads, q_grads, v_grads) = kqv_grads;
            let summed_grads = k_grads.iter().add(q_grads.iter()).add(v_grads.iter());
            let grads = summed_grads.collect();
            Ok(grads)
        }

        pub fn backward_advanced(
            &mut self,
            key_inputs: &Linear,
            query_inputs: &Linear,
            value_inputs: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            if output_gradients.stride() != self.value_weights.stride() {
                Err(anyhow!("mismatched ouput vector size"))?;
            }

            let keys = key_inputs.matrix_product(&self.key_weights);
            let queries = query_inputs.matrix_product(&self.query_weights);
            let values = value_inputs.matrix_product(&self.value_weights);

            let mask = mask.or(self.mask.as_ref());
            let (dkeys, dqueries, dvalues) =
                Self::scaled_attention_d(&keys, &queries, &values, mask, &output_gradients)?;

            let dkey_weights = key_inputs.matrix_product_lhs_transposed(&dkeys);
            let dquery_weights = query_inputs.matrix_product_lhs_transposed(&dqueries);
            let dvalue_weights = value_inputs.matrix_product_lhs_transposed(&dvalues);

            self.add_gradients((dkey_weights, dquery_weights, dvalue_weights));

            let dkey_inputs = dkeys.matrix_product_rhs_transposed(&self.key_weights);
            let dquery_inputs = dqueries.matrix_product_rhs_transposed(&self.query_weights);
            let dvalue_inputs = dvalues.matrix_product_rhs_transposed(&self.value_weights);

            Ok((dkey_inputs, dvalue_inputs, dquery_inputs))
        }

        fn scaled_attention(
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
                    let mask = mask.iter().grow(attention_scores.stride());
                    scaled_attention_scores.set_mask(mask, NodeValue::NEG_INFINITY)
                }
                None => scaled_attention_scores,
            };

            let attention_weights = masked_attention_scores.softmax();
            attention_weights.matrix_product(&values)
        }

        fn scaled_attention_d(
            keys: &Linear,
            queries: &Linear,
            values: &Linear,
            mask: Option<&Linear>,
            output_gradients: &Linear,
        ) -> Result<(Linear, Linear, Linear)> {
            // forward pass
            let attention_scores = queries.iter().matrix_transpose_product(keys.iter());

            let scaled_attention_scores = attention_scores
                .iter()
                .multiply_scalar(1.0 / keys.count() as NodeValue);

            let mask = mask.map(|mask| mask.iter().grow(attention_scores.stride()).collect());
            let masked_attention_scores = match &mask {
                Some(mask) => {
                    scaled_attention_scores.set_mask(mask.iter(), NodeValue::NEG_INFINITY)
                }
                None => scaled_attention_scores,
            };

            let attention_weights = masked_attention_scores.softmax();

            // backward pass
            let dvalues = attention_weights.matrix_product_lhs_transposed(&output_gradients);
            let dattention_weights = output_gradients.matrix_product_rhs_transposed(&values);
            let dmasked_attention_scores = softmax_d_iter(dattention_weights, attention_weights)?;
            // let dmasked_attention_scores = dattention_weights.softmax_d(&attention_weights);

            let dscaled_attention_scores = match &mask {
                Some(mask) => dmasked_attention_scores.iter().set_mask(mask.iter(), 0.0),
                None => dmasked_attention_scores.iter(),
            };

            let dattention_scores = dscaled_attention_scores
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

    // TODO: benchmark against LinearIter::softmax_d, delete loser!
    fn softmax_d_iter(dloss_dsoftmax: Linear, softmax: Linear) -> Result<Linear> {
        // let attention_weights = masked_attention_scores.softmax();
        Linear::from_iter(
            softmax.stride(),
            dloss_dsoftmax
                .rows_iter()
                .zip(softmax.rows_iter())
                .flat_map(|(grads, probs)| NetworkActivationMode::softmax_d(grads, probs)),
        )
    }

    #[cfg(test)]
    mod tests {
        use std::iter;

        use crate::ml::{
            transformer::{
                layers::MultiHeadSelfAttentionLayer,
                tests::helpers::{assert_optimisation_converges, new_linear},
            },
            LayerValues,
        };

        use super::*;

        #[test]
        fn attention_can_compute_outputs_single_head() {
            let seq_len = 3;
            let embed_dim = 12;

            let rng = RngStrategy::testable(1234);

            let inputs = new_inputs(seq_len, embed_dim, &rng);
            let inputs = Linear::from_values(&inputs).unwrap();
            let attention_layer = AttentionHead::new(seq_len, embed_dim, &rng);
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

            let output = AttentionHead::scaled_attention(&queries, &keys, &values, None);

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

            let mut attention_layer = AttentionHead::new(seq_len, embed_dim, &rng);
            let output = attention_layer.forward(&inputs).unwrap().to_values();

            let output_gradients = &new_linear(seq_len, embed_dim, &rng);
            let output_gradients = output_gradients.iter().multiply_scalar(0.01).collect();

            let grads = attention_layer
                .backward(&inputs, &output_gradients)
                .unwrap();

            let grads = grads.to_values();

            assert_eq!(grads.len(), output.len());
            assert_eq!(grads[0].len(), output[0].len());
        }

        #[test]
        fn attention_can_minimise_single_head() {
            let seq_len = 3;
            let embed_dim = 12;
            let learn_rate = 0.001;
            let total_iterations = 25;

            assert_optimisation_converges(
                &move |rng| {
                    let attention_head = AttentionHead::new(seq_len, embed_dim, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_head, inputs, target)
                },
                &move |attention_head, inputs| attention_head.forward(&inputs).unwrap(),
                &move |attention_head, inputs, dloss| {
                    let grads = attention_head.backward(&inputs, &dloss).unwrap();

                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);
                    attention_head.apply_gradients(learn_rate);
                    grads
                },
                total_iterations,
            );
        }

        #[test]
        fn attention_can_minimise_multi_head_layer() {
            let seq_len = 3;
            let embed_dim = 12;
            let head_count = 3;
            let learn_rate = 0.001;
            let total_iterations = 25;

            assert_optimisation_converges(
                &move |rng| {
                    let attention_layer =
                        MultiHeadSelfAttentionLayer::new(seq_len, embed_dim, head_count, &rng);
                    let inputs = new_linear(seq_len, embed_dim, &rng);
                    let target = new_linear(seq_len, embed_dim, &rng);
                    (attention_layer, inputs, target)
                },
                &move |attention_layer, inputs| attention_layer.forward(&inputs).unwrap(),
                &move |attention_layer, inputs, dloss| {
                    let grads = attention_layer.backward(&inputs, &dloss).unwrap();

                    *inputs = inputs.iter().apply_gradients(grads.iter(), learn_rate);
                    attention_layer.apply_gradients(learn_rate);
                    grads
                },
                total_iterations,
            );
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
        LayerNormalization {
            dbeta: Linear,
            dgamma: Linear,
        },
    }
}

pub mod dense {
    use anyhow::{anyhow, Result};

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

        pub fn forward_row(&self, inputs: LayerValues) -> Result<LayerValues> {
            if inputs.len() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }

            let inputs = Linear::from_values(&[inputs])?;
            let outputs = self.forward(&inputs)?;
            let row = outputs.rows_iter().next().unwrap().into();

            Ok(row)
        }

        pub fn forward(&self, inputs: &Linear) -> Result<Linear> {
            if inputs.stride() != self.inputs_count {
                Err(anyhow!("mismatched input vector size"))?;
            }
            let mut weighted_inputs = self.compute_weighted_inputs(&inputs)?;

            let outputs = match &self.activation {
                Some(activation) => {
                    weighted_inputs.rows_iter_mut().for_each(|row| {
                        let activation_row = activation.apply(&row.into());
                        row.clone_from_slice(&activation_row)
                    });
                    weighted_inputs
                }
                None => weighted_inputs,
            };

            Ok(outputs)
        }

        // optimise? too many transposes for func that is called many times forward & backward?
        fn compute_weighted_inputs(&self, inputs: &Linear) -> Result<Linear> {
            // -> output.T = weights.T * inputs.T
            let output = self
                .weights
                .iter_transpose()
                .matrix_transpose_product(inputs.iter());
            // let mut weighted_inputs = self.weights.iter().apply_one(inputs);
            if let Some(bias) = &self.bias {
                let bias = bias.iter().stack(inputs.count());
                Ok(output.iter_transpose().add(bias).collect())
            } else {
                Ok(output.iter_transpose().collect())
            }
        }

        // TODO: optimise by making backward_row the trivial impl and backward operate on Linear matrixes
        pub fn backward(&mut self, inputs: &Linear, output_gradients: &Linear) -> Result<Linear> {
            let mut input_gradients = vec![];

            for (row, inputs) in output_gradients.rows_iter().zip(inputs.rows_iter()) {
                let input_grads = self.backward_row(inputs.into(), row.into())?;
                input_gradients.push(input_grads);
            }

            Linear::from_values(&input_gradients)
        }

        pub fn backward_row(
            &mut self,
            inputs: LayerValues,
            output_gradients: LayerValues,
        ) -> Result<LayerValues> {
            let inputs = Linear::from_values(&[inputs])?;

            let weighted_inputs_gradients = match &self.activation {
                Some(activation) => {
                    let mut weighted_inputs = self.compute_weighted_inputs(&inputs)?;
                    weighted_inputs.rows_iter_mut().for_each(|x| {
                        let outputs = x.into();
                        let activated_inputs = activation.apply(&outputs);
                        activation
                            .derivative(&output_gradients, &activated_inputs)
                            .multiply_iter(&output_gradients)
                            .zip(x)
                            .for_each(|(grad, x)| *x = grad)
                    });
                    weighted_inputs
                }
                None => Linear::from_values(&[output_gradients])?,
            };

            let bias_gradients = &weighted_inputs_gradients;

            // -> output.T = weights.T * inputs.T
            // -> weighted_inputs_gradients.T = weights_gradients.T * inputs.T
            // -> weights_gradients * weighted_inputs_gradients.T = inputs.T
            // -> weights_gradients = inputs.T * weighted_inputs_gradients
            let weights_gradients =
                inputs.matrix_product_lhs_transposed(&weighted_inputs_gradients);
            self.add_gradients(weights_gradients.iter(), bias_gradients.iter());

            // -> output.T = weights.T * inputs.T
            // -> weighted_inputs_gradients.T = self.weights.T * input_gradients_m.T
            // -> input_gradients_m = weighted_inputs_gradients * self.weights.T
            let input_gradients_m =
                weighted_inputs_gradients.matrix_product_rhs_transposed(&self.weights);
            let input_gradients = input_gradients_m.rows_iter().next().unwrap().into();
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

    #[derive(Debug, Clone, PartialEq)]
    pub struct Linear {
        inner: LayerValues,
        stride: usize,
        count: usize,
    }

    impl Linear {
        pub fn new(count: usize, stride: usize) -> Self {
            Self::with_value(count, stride, 0.0)
        }

        pub fn with_dimensions(other: &Self) -> Self {
            Self::with_value(other.count, other.stride, 0.0)
        }

        pub fn with_value(count: usize, stride: usize, value: NodeValue) -> Self {
            let size = count * stride;
            Self {
                inner: LayerValues::new(vec![value; size]),
                stride,
                count,
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

        pub fn is_finite(&self) -> bool {
            self.inner.iter().all(|x| x.is_finite())
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

        pub fn sum<'a, I: Iterator<Item = &'a Self> + 'a>(mut iter: I) -> Option<LinearIter<'a>> {
            let first = iter.next()?;
            let mut sum_iter = first.iter();
            for x in iter {
                sum_iter = sum_iter.add(x.iter());
            }
            Some(sum_iter)
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
        pub fn div(self, rhs: Self, epsilon: Option<NodeValue>) -> Self {
            assert_eq!(self.stride, rhs.stride, "mismatched dimensions");
            assert_eq!(self.count, rhs.count, "mismatched dimensions");
            Self {
                inner: match epsilon {
                    Some(e) => Box::new(self.inner.zip(rhs.inner).map(move |(x, y)| x / (y + e))),
                    None => Box::new(self.inner.zip(rhs.inner).map(move |(x, y)| x / y)),
                },
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
        pub fn apply_one(self, lhs_inputs: &[NodeValue]) -> LayerValues {
            assert_eq!(self.count, lhs_inputs.len(), "mismatched dimensions");

            self.inner
                .chunks(self.stride)
                .into_iter()
                .zip(lhs_inputs)
                .flat_map(|(row, input)| row.map(move |x| x * input))
                .collect()
        }
        pub fn apply_gradients(self, grads: LinearIter, learn_rate: f64) -> Linear {
            self.sub(grads.multiply_scalar(learn_rate)).collect()
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
            assert_eq!(self.count, mean.count, "mismatched dimensions");
            assert_eq!(mean.stride, 1, "invalid mean stride dimension");
            assert!(self.stride > 1, "invalid stride dimension");
            let factor = ((self.stride - 1) as NodeValue).powf(-0.5);
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
            let y = a.iter().div(b.iter(), None).collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    pub mod helpers {
        use super::*;

        pub fn assert_optimisation_converges<T, I>(
            create: &dyn Fn(RngStrategy) -> (T, I, Linear),
            forward: &dyn Fn(&T, &I) -> Linear,
            backward: &dyn Fn(&mut T, &mut I, Linear) -> Linear,
            iters: usize,
        ) {
            let rng = RngStrategy::testable(12345);

            let (mut testable_instance, mut inputs, target) = create(rng.clone());
            let mut iteration = 0;
            let mut initial_mean_loss = None;

            let (output, mean_loss) = loop {
                // compute forward pass
                let output = forward(&testable_instance, &inputs);

                // calculate simple rms loss and gradients compared to target values
                let loss = output.iter().sub(target.iter()).powf_scalar(2.0);
                let dloss = output.iter().sub(target.iter()).multiply_scalar(2.0);

                // compute backward pass, apply gradients, and return inputs gradients
                let grads = backward(&mut testable_instance, &mut inputs, dloss.collect());
                if !grads.is_finite() {
                    println!("{:?}", &output);
                    panic!("failed optimisation: grad value is non-finite");
                }

                // get gradients & loss for reporting
                let abs_grads = grads.iter().abs();
                let mean_grads = abs_grads.flatten_mean().iter_transpose().flatten_mean();
                let mean_grads = *mean_grads.values_iter().next().unwrap().0;
                let mean_loss = loss.flatten_mean().iter_transpose().flatten_mean();
                let mean_loss = *mean_loss.values_iter().next().unwrap().0;
                initial_mean_loss.get_or_insert(mean_loss);

                // report optimisation iteration
                if (iteration + 1) % (iters / 10).max(1) == 0 || iteration == 0 {
                    println!(
                        "optimisation iter: {:<7} | loss: {:<10.5} | mean abs gradient: {:<5.2}",
                        iteration + 1,
                        mean_loss,
                        mean_grads
                    );
                }

                // progress iteration counter, or complete
                iteration += 1;
                if iteration > iters {
                    break (output, mean_loss);
                }
            };
            dbg!((target.to_string(), output.to_string()));

            let initial_mean_loss = initial_mean_loss.unwrap();

            assert!(mean_loss.is_finite(), "final loss has invalid value");

            assert!(
                initial_mean_loss > mean_loss,
                "loss failed to optimise (start={:.2}, end={:.2})",
                initial_mean_loss,
                mean_loss
            );
        }

        pub fn new_linear(seq_len: usize, embedding_dimension: usize, rng: &RngStrategy) -> Linear {
            let mut linear = Linear::new(seq_len, embedding_dimension);
            linear.initialize_as_layer(&LayerInitStrategy::Kaiming, &rng);
            linear
        }
    }
}
