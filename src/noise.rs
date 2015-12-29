pub struct Noise {
    prev: u32,
}

impl Noise {
    pub fn new() -> Noise {
        Noise {
            prev: 3147,
        }
    }

    fn next_val(&self) -> u32 {
        (171 * self.prev + 11213) % 53125
    }

    pub fn next(&mut self) -> u32 {
        let next = self.next_val();
        self.prev = next;

        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise() {
        let mut n = Noise::new();

        assert_eq!(n.next(), 18100);
        assert_eq!(n.next(), 25063);
        assert_eq!(n.next(), 46986);
        assert_eq!(n.next(), 23944);
        assert_eq!(n.next(), 15012);
        assert_eq!(n.next(), 28265);
        assert_eq!(n.next(), 10153);
        assert_eq!(n.next(), 47376);
        assert_eq!(n.next(), 37509);
        assert_eq!(n.next(), 50252);
    }
}
