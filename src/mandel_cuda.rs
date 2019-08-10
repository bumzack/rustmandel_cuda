use image::{ImageBuffer, RgbImage};

use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::time::Instant;

pub fn mandel_cuda(w: u32, h: u32) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;

    let _context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("/tmp/ptx-builder-0.5/cuda_kernel_mandel/dfa49d970a356ec/nvptx64-nvidia-cuda/release/cuda_kernel_mandel.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut width = DeviceBox::new(&(w as f32))?;
    let mut height = DeviceBox::new(&(h as f32))?;


    // TODO: pass Vec<Pixel>  to GPU  - but the DeviceBufferFromSlice does not like that
    let pixels_vec = vec![0f32; w as usize * h as usize * 3];
    let mut pixels = DeviceBuffer::from_slice(&pixels_vec)?;

    let b = (16, 1, 1);
    let block = (b.0 as u32, b.1 as u32, b.2 as u32);

    let g = (
        (w as i32 + block.0 as i32 - 1) / block.0 as i32,
        (h as i32 + block.1 as i32 - 1) / block.1 as i32,
        1 as i32,
    );
    let grid = (g.0 as u32, g.1 as u32, 1 as u32);
    println!("block = {:?}, grid = {:?}", block, grid);

    unsafe {
        launch!(module.calc_mandel<<<grid, block, 0, stream>>>(
            pixels.as_device_ptr(),
            width.as_device_ptr(),
            height.as_device_ptr(),
            block.0,
            block.1
        ))?;
    }
    stream.synchronize()?;

    let mut result_host = vec![0f32; w as usize * h as usize * 3];

    pixels.copy_to(&mut result_host)?;

    let mut image: RgbImage = ImageBuffer::new(w as u32, h as u32);
    let mut x = 0;
    let mut y = 0;
    let mut idx = 0;
    for _i in 0..result_host.len() / 3 {
        let pixel = image::Rgb([
            (result_host[idx]) as u8,
            (result_host[idx + 1]) as u8,
            (result_host[idx + 2]) as u8,
        ]);

        image.put_pixel(x as u32, y as u32, pixel);

        x = x + 1;
        idx = idx + 3;
        if x % w == 0 {
            y = y + 1;
            x = 0;
        }
    }
    let stopped = Instant::now();
    println!("\n\ncuda   {:?} \n\n", stopped.duration_since(start));
    image.save("fractal_cuda.png").unwrap();

    Ok(())
}
