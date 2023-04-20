mongodb_user := env_var_or_default('MONGO_INITDB_ROOT_USERNAME', 'root')
mongodb_pass := env_var_or_default('MONGO_INITDB_ROOT_PASSWORD', 'example')
mongodb_host := env_var_or_default('MONGO_INITDB_HOST', 'localhost')
model_output_label := "dev-model"

build:
  cargo build --features="cli thread threadpool" --release --bin embed

install:
  cargo install --features="cli thread" --bin embed --path .

install-upload:
  cargo build --features="db" --release --bin upload
  cargo install --features="db" --bin upload --path .

testrun-tinyshakespeare:
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --training-rounds 15 --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    -o out -O {{model_output_label}} -i ./tinyshakespeare.txt

run input_file="./tinyshakespeare.txt" rounds="100000":
  embed --quit-on-complete \
    --single-batch-iterations --char --train-rate 0.0016 --batch-size 32 \
    --phrase-word-length-bounds .. --phrase-test-set-max-tokens 500 \
    --hidden-layer-nodes 650  --embedding-size 16 --input-stride-width 64 --repl "PR" \
    --training-rounds {{rounds}} -i {{input_file}} \
    -o out -O {{model_output_label}}

upload-tinyshakespeare:
  MONGODB_HOST=mongodb://{{mongodb_user}}:{{mongodb_pass}}@{{mongodb_host}}:27017 \
    upload out/{{model_output_label}}

upload:
  MONGODB_HOST=mongodb://{{mongodb_user}}:{{mongodb_pass}}@{{mongodb_host}}:27017 \
    upload out/{{model_output_label}}

docker-build:
  docker build -t embed .

docker-down-mongo:
  docker rm -f mongo mongo-express || true

docker-up-mongo: docker-down-mongo
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
  docker run --name embedtestrun --rm embed just testrun-tinyshakespeare
  docker logs -f embedtestrun

docker-test-tty: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare

docker-test-upload: docker-build
  docker run --name embedtestupload --rm embed just testrun-tinyshakespeare upload-tinyshakespeare
  docker logs -f embedtestupload

docker-test-upload-tty: docker-build
  docker run --rm -it embed just testrun-tinyshakespeare upload-tinyshakespeare

push-node node_ip version="v1":
  ssh root@{{node_ip}} mkdir -p /home/docker/code/{{version}}
  git archive --format=zip HEAD > ./res/archive.zip
  scp -p ./res/archive.zip root@{{node_ip}}:/home/docker/code
  ssh root@{{node_ip}} "cd /home/docker/code/{{version}}; unzip ../archive.zip"
  scp -p ./res/tiny*.txt root@{{node_ip}}:/home/docker/code/{{version}}/res/
  ssh root@{{node_ip}} "cd /home/docker/code/{{version}}; just docker-test"
