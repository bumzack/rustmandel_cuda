#!/bin/sh

cd cuda_kernel_mandel
cargo build --release &&
cd .. &&
cargo run --release


