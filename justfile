mongodb_user := env_var_or_default('MONGO_INITDB_ROOT_USERNAME', 'root')
mongodb_pass := env_var_or_default('MONGO_INITDB_ROOT_PASSWORD', 'example')
mongodb_host := env_var_or_default('MONGO_INITDB_HOST', 'localhost')
model_json := "out/model.json"

build:
  cargo build --features="cli thread threadpool" --release --bin embed

install:
  cargo install --offline --features="cli thread" --bin embed --path .

testrun-tinyshakespeare:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --training-rounds 250 --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    -o out -O dev-tinyshakespeare-superfast -i ./tinyshakespeare.txt

upload-tinyshakespeare:
  MONGODB_HOST=mongodb://{{mongodb_user}}:{{mongodb_pass}}@{{mongodb_host}}:27017 cargo run --features="db" --release --bin upload -- \
    ./dev-tinyshakespeare-superfast

upload:
  MONGODB_HOST=mongodb://{{mongodb_user}}:{{mongodb_pass}}@{{mongodb_host}}:27017 cargo run --features="db" --release --bin upload -- \
    {{model_json}}

docker-build:
  docker build -t embed .

docker-up-mongo:
  docker stop mongo mongo-express
  docker rm mongo mongo-express
  docker run -d \
    --name mongo \
    --restart always \
    -p 27017:27017 \
    -e MONGO_INITDB_ROOT_USERNAME={{mongodb_user}} \
    -e MONGO_INITDB_ROOT_PASSWORD={{mongodb_pass}} \
    mongo
  docker run -d \
    --name mongo-express \
    --restart always \
    --network host \
    -e ME_CONFIG_MONGODB_ADMINUSERNAME={{mongodb_user}} \
    -e ME_CONFIG_MONGODB_ADMINPASSWORD={{mongodb_pass}} \
    -e ME_CONFIG_MONGODB_URL=mongodb://{{mongodb_user}}:{{mongodb_pass}}@localhost:27017 \
    mongo-express

docker-test: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare