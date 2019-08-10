extern crate image;
#[macro_use]
extern crate rustacuda;
use crate::mandel_cuda::mandel_cuda;
use crate::mandel_multi_core::mandel_threads;
use crate::mandel_single_core::mandel_single;
use std::error::Error;

mod mandel_cuda;
mod mandel_multi_core;
mod mandel_single_core;
mod mandel_utils;

fn main() -> Result<(), Box<dyn Error>> {
    let w = 800;
    let h = 600;

    println!("---------- SINGLE CORE  --------------------");
    mandel_single(w, h);

    println!("\n\n---------- MULTI CORE  --------------------");
    mandel_threads(w, h);

    println!("\n\n---------- CUDA   --------------------");
    mandel_cuda(w, h)?;
    Ok(())
}
