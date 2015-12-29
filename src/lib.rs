#![allow(warnings)]

extern crate arrayvec;
extern crate collect_slice;
extern crate num;

mod allocs;
mod chunk;
mod coefs;
mod consts;
mod descramble;
mod gain;
mod noise;
mod params;
mod prev;
mod scan;
mod spectral;
mod unvoiced;
mod window;
