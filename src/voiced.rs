//! Voiced spectrum synthesis.

use std::cmp::max;
use std::f32::consts::PI;

use collect_slice::CollectSlice;
use map_in_place::MapInPlace;
use rand::Rng;

use consts::{SAMPLES_PER_FRAME, MAX_HARMONICS};
use descramble::VoiceDecisions;
use enhance::EnhancedSpectrals;
use params::BaseParams;
use prev::PrevFrame;
use window;

/// Computes the base phase offsets Ψ<sub>l</sub>.
pub struct PhaseBase([f32; MAX_HARMONICS]);

impl PhaseBase {
    /// Create a new `PhaseBase` from the given current and previous frame parameters.
    pub fn new(params: &BaseParams, prev: &PrevFrame) -> Self {
        let mut base = [0.0; MAX_HARMONICS];

        // Compute common scaling factor in Eq 139.
        let scale = (prev.params.fundamental + params.fundamental) *
            SAMPLES_PER_FRAME as f32 / 2.0;

        // Compute Eq 139.
        (1...MAX_HARMONICS).map(|l| {
            prev.phase_base.get(l) + scale * l as f32
        }).collect_slice_checked(&mut base[..]);

        PhaseBase(base)
    }

    /// Retrieve the phase term Ψ<sub>l</sub>, 1 ≤ l ≤ 56.
    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

impl Default for PhaseBase {
    /// Create a new `PhaseBase` in the default state.
    fn default() -> Self {
        // By default all phase terms are 0 [p64].
        PhaseBase([0.0; MAX_HARMONICS])
    }
}

/// Computes the random phase terms Φ<sub>l</sub>.
pub struct Phase([f32; MAX_HARMONICS]);

impl Phase {
    /// Create a new `Phase` building on the given base phase terms.
    pub fn new<R: Rng>(base: &PhaseBase, params: &BaseParams, prev: &PrevFrame,
                       voice: &VoiceDecisions, mut noise: R)
        -> Self
    {
        let mut phase = [0.0; MAX_HARMONICS];

        // Derive phase terms from base phase offsets according to Eq 140.
        (&mut phase[..]).copy_from_slice(&base.0[..]);

        // Compute bounds for modification used in Eq 140.
        let start = params.harmonics as usize / 4;
        let stop = max(params.harmonics, prev.params.harmonics) as usize;

        // Compute common scaling factor in Eq 140.
        let scale = voice.unvoiced_count() as f32 / params.harmonics as f32;

        // Modify Ψ_l from start + 1 ≤ l ≤ stop. Since i = l - 1, start ≤ i ≤ stop - 1.
        (&mut phase[start..stop]).map_in_place(|&x| {
            // Compute Eq 140.
            x + scale * noise.gen_range(-PI, PI)
        });

        Phase(phase)
    }

    /// Retrieve the phase term Φ<sub>l</sub>, 1 ≤ l ≤ 56.
    pub fn get(&self, l: usize) -> f32 { self.0[l - 1] }
}

impl Default for Phase {
    /// Create a new `Phase` in the default state.
    fn default() -> Self {
        // By default all phase terms are 0 [p64].
        Phase([0.0; MAX_HARMONICS])
    }
}

/// Synthesizes voiced spectrum signal s<sub>v</sub>(n).
pub struct Voiced<'a, 'b, 'c, 'd> {
    prev: &'a PrevFrame,
    phase: &'b Phase,
    amps: &'c EnhancedSpectrals,
    voice: &'d VoiceDecisions,
    /// Synthesis window w<sub>s</sub> for combining voiced/unvoiced frames.
    window: window::Window,
    /// Fundamental frequency of current frame.
    fundamental: f32,
    /// Number of harmonics that make up each signal sample.
    end: usize,
}

impl<'a, 'b, 'c, 'd> Voiced<'a, 'b, 'c, 'd> {
    pub fn new(params: &BaseParams, prev: &'a PrevFrame, phase: &'b Phase,
               amps: &'c EnhancedSpectrals, voice: &'d VoiceDecisions)
        -> Self
    {
        Voiced {
            prev: prev,
            phase: phase,
            amps: amps,
            voice: voice,
            window: window::synthesis(),
            fundamental: params.fundamental,
            // Compute the sum bound in Eq 127.
            end: max(params.harmonics, prev.params.harmonics) as usize,
        }
    }

    /// Compute s<sub>v,l</sub>(n), the signal level at sample n for the l'th spectral
    /// amplitude.
    fn get_pair(&self, l: usize, n: isize) -> f32 {
        match (self.voice.is_voiced(l), self.prev.voice.is_voiced(l)) {
            // Use Eq 130.
            (false, false) => 0.0,
            // Use Eq 131.
            (false, true) => self.sig_prev(l, n),
            // Use Eq 132.
            (true, false) => self.sig_cur(l, n),
            // Use Eq 133. The Eq 134 form for voiced/voiced frames isn't used due to its
            // complexity and lack of rationale.
            (true, true) => self.sig_prev(l, n) + self.sig_cur(l, n)
        }
    }

    /// Compute s<sub>v,l</sub>(n) for a voiced current frame and unvoiced previous frame.
    fn sig_cur(&self, l: usize, n: isize) -> f32 {
        // Compute Eq 132.
        self.window.get(n - SAMPLES_PER_FRAME as isize) * self.amps.get(l) * (
            self.fundamental * (n - SAMPLES_PER_FRAME as isize) as f32 * l as f32 +
                self.phase.get(l)
        ).cos()
    }

    /// Compute s<sub>v,l</sub>(n) for an unvoiced current frame and voiced previous frame.
    fn sig_prev(&self, l: usize, n: isize) -> f32 {
        // Compute Eq 131.
        self.window.get(n) * self.prev.enhanced.get(l) * (
            self.prev.params.fundamental * n as f32 * l as f32 +
                self.prev.phase.get(l)
        ).cos()
    }

    /// Compute the voiced signal sample s<sub>v</sub>(n) for the given sample n, 0 ≤ n <
    /// 160.
    pub fn get(&self, n: usize) -> f32 {
        debug_assert!(n < SAMPLES_PER_FRAME);

        // Compute Eq 127
        2.0 * (1...self.end)
            .map(|l| self.get_pair(l, n as isize))
            .fold(0.0, |s, x| s + x)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use params::BaseParams;
    use prev::PrevFrame;
    use descramble::{Bootstrap, descramble};
    use rand::XorShiftRng;

    #[test]
    fn test_phase_base() {
        // Verify results match standalone python script.

        let chunks = [
            0b001000010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b10101110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());
        let mut prev = PrevFrame::default();

        for l in 1...56 {
            prev.phase_base.0[l-1] = l as f32;
        }

        assert!((p.fundamental - 0.17575344).abs() < 0.000001);
        assert!((prev.params.fundamental - 0.0937765407).abs() < 0.000001);
        assert_eq!(p.harmonics, 16);

        let pb = PhaseBase::new(&p, &prev);

        assert!((pb.get(1) - 22.56239845600000037961763155180961).abs() < 1e-3);
        assert!((pb.get(2) - 45.12479691200000075923526310361922).abs() < 1e-3);
        assert!((pb.get(3) - 67.68719536800000469156657345592976).abs() < 1e-3);
        assert!((pb.get(4) - 90.24959382400000151847052620723844).abs() < 1e-3);
        assert!((pb.get(5) - 112.81199227999999834537447895854712).abs() < 1e-3);
        assert!((pb.get(6) - 135.37439073600000938313314691185951).abs() < 1e-3);
        assert!((pb.get(7) - 157.93678919199999199918238446116447).abs() < 1e-3);
        assert!((pb.get(8) - 180.49918764800000303694105241447687).abs() < 1e-3);
        assert!((pb.get(9) - 203.06158610399998565299028996378183).abs() < 1e-3);
        assert!((pb.get(10) - 225.62398455999999669074895791709423).abs() < 1e-3);
        assert!((pb.get(11) - 248.18638301600003615021705627441406).abs() < 1e-3);
        assert!((pb.get(12) - 270.74878147200001876626629382371902).abs() < 1e-3);
        assert!((pb.get(13) - 293.31117992800000138231553137302399).abs() < 1e-3);
        assert!((pb.get(14) - 315.87357838399998399836476892232895).abs() < 1e-3);
        assert!((pb.get(15) - 338.43597684000002345783286727964878).abs() < 1e-3);
        assert!((pb.get(16) - 360.99837529600000607388210482895374).abs() < 1e-3);
        assert!((pb.get(17) - 383.56077375199998868993134237825871).abs() < 1e-3);
        assert!((pb.get(18) - 406.12317220799997130598057992756367).abs() < 1e-3);
        assert!((pb.get(19) - 428.68557066400001076544867828488350).abs() < 1e-3);
        assert!((pb.get(20) - 451.24796911999999338149791583418846).abs() < 1e-3);
        assert!((pb.get(21) - 473.81036757599997599754715338349342).abs() < 1e-3);
        assert!((pb.get(22) - 496.37276603200007230043411254882812).abs() < 1e-3);
        assert!((pb.get(23) - 518.93516448800005491648335009813309).abs() < 1e-3);
        assert!((pb.get(24) - 541.49756294400003753253258764743805).abs() < 1e-3);
        assert!((pb.get(25) - 564.05996140000002014858182519674301).abs() < 1e-3);
        assert!((pb.get(26) - 586.62235985600000276463106274604797).abs() < 1e-3);
        assert!((pb.get(27) - 609.18475831199998538068030029535294).abs() < 1e-3);
        assert!((pb.get(28) - 631.74715676799996799672953784465790).abs() < 1e-3);
        assert!((pb.get(29) - 654.30955522399995061277877539396286).abs() < 1e-3);
        assert!((pb.get(30) - 676.87195368000004691566573455929756).abs() < 1e-3);
        assert!((pb.get(31) - 699.43435213600002953171497210860252).abs() < 1e-3);
        assert!((pb.get(32) - 721.99675059200001214776420965790749).abs() < 1e-3);
        assert!((pb.get(33) - 744.55914904799999476381344720721245).abs() < 1e-3);
        assert!((pb.get(34) - 767.12154750399997737986268475651741).abs() < 1e-3);
        assert!((pb.get(35) - 789.68394595999995999591192230582237).abs() < 1e-3);
        assert!((pb.get(36) - 812.24634441599994261196115985512733).abs() < 1e-3);
        assert!((pb.get(37) - 834.80874287200003891484811902046204).abs() < 1e-3);
        assert!((pb.get(38) - 857.37114132800002153089735656976700).abs() < 1e-3);
        assert!((pb.get(39) - 879.93353978400000414694659411907196).abs() < 1e-3);
        assert!((pb.get(40) - 902.49593823999998676299583166837692).abs() < 1e-3);
        assert!((pb.get(41) - 925.05833669599996937904506921768188).abs() < 1e-3);
        assert!((pb.get(42) - 947.62073515199995199509430676698685).abs() < 1e-3);
        assert!((pb.get(43) - 970.18313360799993461114354431629181).abs() < 1e-3);
        assert!((pb.get(44) - 992.74553206400014460086822509765625).abs() < 1e-3);
        assert!((pb.get(45) - 1015.30793052000012721691746264696121).abs() < 1e-3);
        assert!((pb.get(46) - 1037.87032897600010983296670019626617).abs() < 1e-3);
        assert!((pb.get(47) - 1060.43272743200009244901593774557114).abs() < 1e-3);
        assert!((pb.get(48) - 1082.99512588800007506506517529487610).abs() < 1e-3);
        assert!((pb.get(49) - 1105.55752434400005768111441284418106).abs() < 1e-3);
        assert!((pb.get(50) - 1128.11992280000004029716365039348602).abs() < 1e-3);
        assert!((pb.get(51) - 1150.68232125600002291321288794279099).abs() < 1e-3);
        assert!((pb.get(52) - 1173.24471971200000552926212549209595).abs() < 1e-3);
        assert!((pb.get(53) - 1195.80711816799998814531136304140091).abs() < 1e-3);
        assert!((pb.get(54) - 1218.36951662399997076136060059070587).abs() < 1e-3);
        assert!((pb.get(55) - 1240.93191507999995337740983814001083).abs() < 1e-3);
        assert!((pb.get(56) - 1263.49431353599993599345907568931580).abs() < 1e-3);
    }

    #[test]
    fn test_phase() {
        // Verify results match standalone python script.

        let chunks = [
            0b001000010010,
            0b110011001100,
            0b111000111000,
            0b111111111111,
            0b10101110101,
            0b00101111010,
            0b01110111011,
            0b00001000,
        ];

        let b = Bootstrap::new(&chunks);
        let p = BaseParams::new(b.unwrap_period());
        let (_, voice, _) = descramble(&chunks, &p);
        let mut prev = PrevFrame::default();

        for l in 1...56 {
            prev.phase_base.0[l-1] = l as f32;
        }

        assert_eq!(p.harmonics, 16);
        assert_eq!(prev.params.harmonics, 30);
        assert_eq!(voice.unvoiced_count(), 6);

        let pb = PhaseBase::new(&p, &prev);
        let p = Phase::new(&pb, &p, &prev, &voice, XorShiftRng::new_unseeded());

        assert!((p.get(1) - 22.56239845600000037961763155180961).abs() < 1e-3);
        assert!((p.get(2) - 45.12479691200000075923526310361922).abs() < 1e-3);
        assert!((p.get(3) - 67.68719536800000469156657345592976).abs() < 1e-3);
        assert!((p.get(4) - 90.24959382400000151847052620723844).abs() < 1e-3);
        assert!((p.get(5) - 113.72102393936469866275729145854712).abs() < 1e-3);
        assert!((p.get(6) - 136.03487853683657249348470941185951).abs() < 1e-3);
        assert!((p.get(7) - 157.33372207705076561978785321116447).abs() < 1e-3);
        assert!((p.get(8) - 180.89240547985005491682386491447687).abs() < 1e-3);
        assert!((p.get(9) - 203.73622094806469817740435246378183).abs() < 1e-3);
        assert!((p.get(10) - 226.08445127271323826789739541709423).abs() < 1e-3);
        assert!((p.get(11) - 247.40521646250272169709205627441406).abs() < 1e-3);
        assert!((p.get(12) - 269.58186015442265670571941882371902).abs() < 1e-3);
        assert!((p.get(13) - 292.96847452920201249071396887302399).abs() < 1e-3);
        assert!((p.get(14) - 315.54974697478405687434133142232895).abs() < 1e-3);
        assert!((p.get(15) - 337.87577965054975948078208602964878).abs() < 1e-3);
        assert!((p.get(16) - 361.83932748377657162563991732895374).abs() < 1e-3);
        assert!((p.get(17) - 383.78916983988489164403290487825871).abs() < 1e-3);
        assert!((p.get(18) - 407.27547486059739867414464242756367).abs() < 1e-3);
        assert!((p.get(19) - 429.51964262802550820197211578488350).abs() < 1e-3);
        assert!((p.get(20) - 451.09838437959967905044322833418846).abs() < 1e-3);
        assert!((p.get(21) - 474.87744446766066630516434088349342).abs() < 1e-3);
        assert!((p.get(22) - 496.48125262476969510316848754882812).abs() < 1e-3);
        assert!((p.get(23) - 519.98516804639734800730366259813309).abs() < 1e-3);
        assert!((p.get(24) - 540.45943817407396636554040014743805).abs() < 1e-3);
        assert!((p.get(25) - 563.98577140924112427455838769674301).abs() < 1e-3);
        assert!((p.get(26) - 586.43880994521623506443575024604797).abs() < 1e-3);
        assert!((p.get(27) - 609.11271337319726626446936279535294).abs() < 1e-3);
        assert!((p.get(28) - 630.72016101971848911489360034465790).abs() < 1e-3);
        assert!((p.get(29) - 655.26625760537240239500533789396286).abs() < 1e-3);
        assert!((p.get(30) - 677.98135886975262565101729705929756).abs() < 1e-3);
        assert!((p.get(31) - 699.43435213600002953171497210860252).abs() < 1e-3);
        assert!((p.get(32) - 721.99675059200001214776420965790749).abs() < 1e-3);
        assert!((p.get(33) - 744.55914904799999476381344720721245).abs() < 1e-3);
        assert!((p.get(34) - 767.12154750399997737986268475651741).abs() < 1e-3);
        assert!((p.get(35) - 789.68394595999995999591192230582237).abs() < 1e-3);
        assert!((p.get(36) - 812.24634441599994261196115985512733).abs() < 1e-3);
        assert!((p.get(37) - 834.80874287200003891484811902046204).abs() < 1e-3);
        assert!((p.get(38) - 857.37114132800002153089735656976700).abs() < 1e-3);
        assert!((p.get(39) - 879.93353978400000414694659411907196).abs() < 1e-3);
        assert!((p.get(40) - 902.49593823999998676299583166837692).abs() < 1e-3);
        assert!((p.get(41) - 925.05833669599996937904506921768188).abs() < 1e-3);
        assert!((p.get(42) - 947.62073515199995199509430676698685).abs() < 1e-3);
        assert!((p.get(43) - 970.18313360799993461114354431629181).abs() < 1e-3);
        assert!((p.get(44) - 992.74553206400014460086822509765625).abs() < 1e-3);
        assert!((p.get(45) - 1015.30793052000012721691746264696121).abs() < 1e-3);
        assert!((p.get(46) - 1037.87032897600010983296670019626617).abs() < 1e-3);
        assert!((p.get(47) - 1060.43272743200009244901593774557114).abs() < 1e-3);
        assert!((p.get(48) - 1082.99512588800007506506517529487610).abs() < 1e-3);
        assert!((p.get(49) - 1105.55752434400005768111441284418106).abs() < 1e-3);
        assert!((p.get(50) - 1128.11992280000004029716365039348602).abs() < 1e-3);
        assert!((p.get(51) - 1150.68232125600002291321288794279099).abs() < 1e-3);
        assert!((p.get(52) - 1173.24471971200000552926212549209595).abs() < 1e-3);
        assert!((p.get(53) - 1195.80711816799998814531136304140091).abs() < 1e-3);
        assert!((p.get(54) - 1218.36951662399997076136060059070587).abs() < 1e-3);
        assert!((p.get(55) - 1240.93191507999995337740983814001083).abs() < 1e-3);
        assert!((p.get(56) - 1263.49431353599993599345907568931580).abs() < 1e-3);
    }
}
