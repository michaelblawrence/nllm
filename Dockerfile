FROM rust:1.68.2-alpine3.17 as builder

ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
RUN set -eux; \
        apk add --no-cache musl-dev; \
		rustup --version; \
        cargo install just;

WORKDIR /app
RUN set -eux; \
		cargo init --vcs none --bin; \
		ls -l; \
		mkdir src/bin; \
		mkdir src/bin/embed; \
		cp src/main.rs src/bin/derivate.rs; \
		cp src/main.rs src/bin/upload.rs; \
		cp src/main.rs src/bin/embed/main.rs;

COPY Cargo.toml Cargo.lock ./
        
RUN set -eux; \
        cargo install --features="multi_threaded" --bin embed --path .;

COPY src src
COPY res/tiny*.txt ./res/
RUN touch src/bin/embed/main.rs src/bin/derivate.rs src/main.rs;

RUN set -eux; \
        cargo install --features="multi_threaded" --bin embed --path .;

COPY justfile ./

RUN set -eux; \
        mkdir out
