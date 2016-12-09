#![feature(inclusive_range_syntax)]

extern crate arrayvec;
extern crate collect_slice;
extern crate num;
extern crate quad_osc;
extern crate rand;
extern crate thread_scoped;

pub mod decoder;
pub mod consts;

mod allocs;
mod chunk;
mod coefs;
mod descramble;
mod enhance;
mod errors;
mod gain;
mod params;
mod prev;
mod scan;
mod spectral;
mod unvoiced;
mod voiced;
mod window;
