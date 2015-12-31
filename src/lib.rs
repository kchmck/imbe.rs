#![allow(warnings)]

extern crate arrayvec;
extern crate collect_slice;
extern crate num;

mod allocs;
mod chunk;
mod coefs;
mod consts;
mod decoder;
mod descramble;
mod enhance;
mod error;
mod errors;
mod gain;
mod noise;
mod params;
mod prev;
mod scan;
mod spectral;
mod unvoiced;
mod voiced;
mod window;
