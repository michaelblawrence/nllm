# nllm
A (**N**ot so) **L**arge **L**anguage **M**odel, powered by a next-token predictive decoder-only transformer engine called 'plane'. Built for educational purposes.

Features:
- CLI-based model training engine and chat-based inference tool
- Support for Transformer, Decoder-only and general MLP models
- On-demand training checkpoint exports + configurable auto-save checkpoints
- Unsupervised training supports input corpus from any local plaintext file  
- Sub-word tokenization support using byte-pair encoding (BPE)
- Configurable numeric solver implementations (SGD, Adam, RMSProp)
- Portable, and hopefully deployable, due to tiny size of models
