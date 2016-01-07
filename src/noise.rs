use std::f32::consts::PI;

use rand::{self, XorShiftRng, Rng};

#[derive(Clone)]
pub struct Noise(XorShiftRng);

impl Noise {
    pub fn new() -> Noise {
        Noise(rand::weak_rng())
    }

    pub fn next(&mut self) -> f32 {
        self.0.next_f32()
    }

    pub fn next_phase(&mut self) -> f32 {
        self.0.gen_range(-PI, PI)
    }
}
