#[macro_use]
extern crate rustacuda;


use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use rustacuda::memory::{DeviceBox};



fn render_mandelbrot_cuda() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_kernel_mandel/6def2f1805f66bf6/nvptx64-nvidia-cuda/release/cuda_kernel_mandel.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut width = DeviceBox::new(&640f32)?;
    let mut height = DeviceBox::new(&480f32)?;

    // Allocate space on the device and copy numbers to it.
    let mut pixels = DeviceBuffer::from_slice(&[0f32; 640 * 480 * 3])?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.

    // pub unsafe extern "ptx-kernel" fn calc_mandel(pixels: *const Pixel, w: usize, h: usize) {
    unsafe {
        // Launch the `add` function with one block containing four threads on the stream.
        launch!(module.calc_mandel<<<(15, 15, 1), (1, 1, 1), 0, stream>>>(
            pixels.as_device_ptr(),
           width.as_device_ptr(),
            height.as_device_ptr()
        ))?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the result back to the host
    let mut result_host = [0f32; 640 * 480 * 3];

    let mut iter = pixels.chunks(32);

    // pixels.copy_to(&mut result_host)?;

    /// println!("bla sum is {:?}", result_host[100 * 3]);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    render_mandelbrot_cuda()
}