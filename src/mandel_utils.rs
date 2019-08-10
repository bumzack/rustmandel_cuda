pub fn get_color_from_iterations(i: i32, max_iterations: i32) -> (i32, i32, i32) {
    if i == max_iterations {
        (0, 0, 0)
    } else {
        let mut red = (i % 32) * 3;
        if red > 255 {
            red = 255;
        }

        let mut green = (i % 16) * 2;
        if green > 255 {
            green = 255;
        }

        let mut blue = (i % 128) * 14;
        if blue > 255 {
            blue = 255;
        }
        (red, green, blue)
    }
}
