//! Decode the Improved Multi-Band Excitation (IMBE) digital voice codec.

#![feature(inclusive_range_syntax)]

extern crate arrayvec;
extern crate collect_slice;
extern crate crossbeam;
extern crate map_in_place;
extern crate num;
extern crate iq_osc;
extern crate rand;

pub mod allocs;
pub mod coefs;
pub mod consts;
pub mod decode;
pub mod descramble;
pub mod enhance;
pub mod frame;
pub mod gain;
pub mod params;
pub mod prev;
pub mod scan;
pub mod spectral;
pub mod unvoiced;
pub mod voiced;
pub mod window;

pub use decode::ImbeDecoder;
pub use frame::ReceivedFrame;
