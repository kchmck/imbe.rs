# imbe.rs

[Documentation](http://kchmck.github.io/doc/imbe/)

This library implements a real-time decoder for the Improved Multi-Band
Excitation (IMBE) digital voice codec. The implementation performs no
allocations and uses optimized [unvoiced spectrum
synthesis](http://kchmck.github.io/doc/imbe/unvoiced/index.html).

IMBE is a voice codec published in '95 that encodes 20ms frames of speech into
11 bytes. It's used for low-bitrate (4400bps) voice transmissions in the
[Project 25](https://github.com/kchmck/p25.rs) radio protocol.
