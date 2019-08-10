use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use crate::mandel_utils::get_color_from_iterations;
use image::{ImageBuffer, RgbImage};

pub fn mandel_threads(w: u32, h: u32) {
    let width = w as f32;
    let height = h as f32;

    let image: RgbImage = ImageBuffer::new(w, h);

    let num_threads = 8;

    let data = Arc::new(Mutex::new(image));

    let mut children = vec![];

    let act_y: usize = 0;
    let act_y_mutex = Arc::new(Mutex::new(act_y));

    let start = Instant::now();

    for _i in 0..num_threads {
        let cloned_data = data.clone();
        let cloned_act_y = act_y_mutex.clone();

        children.push(thread::spawn(move || {
            let max_iterations = 10000;
            let max_radius = 4.0;

            let mut z_real;
            let mut z_img;
            let mut z_new_real;
            let mut z_new_img;

            let left_bottom_real = -2.0;
            let left_bottom_img = -1.0;
            let right_top_real = 1.0;
            let right_top_img = 1.0;

            let mut c_real;
            let mut c_img;

            let inc_real = (right_top_real - left_bottom_real) / width;
            let inc_img = (right_top_img - left_bottom_img) / height;

            let mut y = 0;

            while *cloned_act_y.lock().unwrap() < h as usize {
                if y < h {
                    let mut acty = cloned_act_y.lock().unwrap();
                    y = (*acty) as u32;
                    *acty = *acty + 1;
                }
                for x in 0..w {
                    c_real = left_bottom_real + x as f32 * inc_real;
                    c_img = left_bottom_img + y as f32 * inc_img;

                    z_real = 0.0;
                    z_img = 0.0;

                    let mut i = 0;

                    while z_real * z_real + z_img * z_img < max_radius && i < max_iterations {
                        z_new_real = z_real * z_real - z_img * z_img + c_real;
                        z_new_img = 2.0 * z_real * z_img + c_img;

                        z_real = z_new_real;
                        z_img = z_new_img;

                        i += 1;
                    }
                    let (red, green, blue) = get_color_from_iterations(i, max_iterations);
                    let pixel = image::Rgb([red as u8, green as u8, blue as u8]);

                    let mut img = cloned_data.lock().unwrap();
                    img.put_pixel(x as u32, y as u32, pixel);
                }
            }
        }));
    }

    for child in children {
        let _ = child.join();
    }

    let stopped = Instant::now();
    println!(
        "\n\nmulti core ({} threads) {:?} \n\n",
        num_threads,
        stopped.duration_since(start)
    );
    let img = data.lock().unwrap();
    img.save("fractal_multi.png").unwrap();
}
