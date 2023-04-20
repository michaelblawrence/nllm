build:
  cargo build --features="cli thread threadpool" --release --bin embed

install:
  cargo install --offline --features="cli thread" --bin embed --path .

testrun-tinyshakespeare: install
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --training-rounds 250 --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    -o out -O dev-tinyshakespeare-superfast -i ./tinyshakespeare.txt

docker-build:
  docker build -t embed .

docker-test: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare