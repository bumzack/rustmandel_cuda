#!/bin/sh

cd cuda_kernel_mandel
cargo build &&
cd .. &&
cargo run

