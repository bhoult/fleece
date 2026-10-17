//! The on-screen text: FRAKTAL256's 5x7 bitmap font drawn into a small RGBA
//! overlay over a dimmed panel (the GPU blends it over the frame), and rows
//! of buttons the mouse can press (each does what its key does).

/// Glyphs for 0-9, A-Z, then the symbols of `SYMBOLS`; bit 4 is the leftmost
/// column.
static FONT: [[u8; 7]; 49] = [
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    [0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100],
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0b11111, 0, 0, 0],
    [0, 0, 0, 0, 0, 0b01100, 0b01100],
    [0, 0b01100, 0b01100, 0, 0b01100, 0b01100, 0],
    [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0, 0b00100],
    [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
    [0, 0, 0, 0, 0b01100, 0b00100, 0b01000],
    [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
];

const SYMBOLS: &[u8] = b" -.:/!+()[],%";

fn glyph(c: u8) -> Option<usize> {
    let c = c.to_ascii_uppercase();
    Some(match c {
        b'0'..=b'9' => (c - b'0') as usize,
        b'A'..=b'Z' => (c - b'A') as usize + 10,
        _ => SYMBOLS.iter().position(|&s| s == c)? + 36,
    })
}

pub const LINE: u32 = 10;
const ADVANCE: u32 = 6;

/// A button row's height: the box (two pixels around the glyphs) and a gap.
const BUTTON_LINE: u32 = 14;

/// An RGBA overlay with lines of text on a translucent dark panel.
pub struct Hud {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
    /// where the buttons are (overlay pixels)
    pub buttons: Vec<Button>,
}

/// A piece of a panel row: text, or a button doing what the key `action` does.
pub struct Item {
    pub text: String,
    /// a button's key, shown dimmer before its text
    pub hint: &'static str,
    pub action: Option<char>,
    pub style: Style,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Style {
    Plain,
    Dim,
    Red,
    /// a button whose setting is on (lit) or off
    On,
    Off,
}

impl Item {
    pub fn text(text: impl Into<String>, style: Style) -> Item {
        Item { text: text.into(), hint: "", action: None, style }
    }
    pub fn button(hint: &'static str, text: impl Into<String>, action: char, on: bool) -> Item {
        Item { text: text.into(), hint, action: Some(action), style: if on { Style::On } else { Style::Off } }
    }
}

/// A button's box on the overlay, and its key.
#[derive(Clone, Copy, Debug)]
pub struct Button {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub action: char,
}

impl Hud {
    /// A panel just big enough for `lines`.
    pub fn new(lines: &[String]) -> Hud {
        let cols = lines.iter().map(|l| l.len()).max().unwrap_or(1) as u32;
        let w = cols * ADVANCE + 8;
        let h = lines.len() as u32 * LINE + 6;
        let mut hud = Hud { w, h, px: vec![0; (w * h * 4) as usize], buttons: Vec::new() };
        for p in hud.px.chunks_exact_mut(4) {
            p.copy_from_slice(&[0, 0, 0, 150]);
        }
        for (i, l) in lines.iter().enumerate() {
            // ~ dims a line, * makes it red
            let (text, rgb) = match l.chars().next() {
                Some('~') => (&l[1..], [150, 160, 175]),
                Some('*') => (&l[1..], [255, 80, 70]),
                _ => (&l[..], [235, 238, 245]),
            };
            hud.text(4, 4 + i as u32 * LINE, text, rgb);
        }
        hud
    }

    /// A panel of rows (each wrapped at `max_w` pixels, a text item standing
    /// alone wrapped by words), the button under `hover` lit.
    pub fn panel(rows: &[Vec<Item>], max_w: u32, hover: Option<(u32, u32)>) -> Hud {
        // where every item goes: (row's top, x, item), rows of buttons taller
        let width = |it: &Item| {
            let chars = it.hint.len() + (!it.hint.is_empty()) as usize + it.text.len();
            chars as u32 * ADVANCE - 1 + if it.action.is_some() { 6 } else { 0 }
        };
        let mut placed: Vec<(u32, u32, &Item)> = Vec::new();
        let (mut y, mut wmax) = (4u32, 0u32);
        for row in rows {
            let tall = row.iter().any(|it| it.action.is_some());
            let line = if tall { BUTTON_LINE } else { LINE };
            let mut x = 4u32;
            for it in row {
                let w = width(it);
                if x > 4 && x + w > max_w {
                    y += line;
                    x = 4;
                }
                placed.push((y, x, it));
                x += w + ADVANCE + if tall { 2 } else { 0 };
                wmax = wmax.max(x);
            }
            y += line;
        }
        let (w, h) = (wmax.max(16) + 2, y + 2);
        let mut hud = Hud { w, h, px: vec![0; (w * h * 4) as usize], buttons: Vec::new() };
        for p in hud.px.chunks_exact_mut(4) {
            p.copy_from_slice(&[0, 0, 0, 150]);
        }
        for (y, x, it) in placed {
            let rgb = |s: Style| match s {
                Style::Plain => [235, 238, 245],
                Style::Dim => [150, 160, 175],
                Style::Red => [255, 80, 70],
                Style::On => [255, 255, 255],
                Style::Off => [170, 178, 192],
            };
            let Some(action) = it.action else {
                hud.text(x, y, &it.text, rgb(it.style));
                continue;
            };
            let b = Button { x0: x, y0: y, x1: x + width(it), y1: y + 11, action };
            let lit = hover.is_some_and(|(hx, hy)| hx >= b.x0 && hx < b.x1 && hy >= b.y0 && hy < b.y1);
            let fill: [u8; 4] = match (it.style == Style::On, lit) {
                (true, false) => [40, 92, 190, 220],
                (true, true) => [70, 130, 235, 235],
                (false, false) => [58, 62, 72, 205],
                (false, true) => [100, 106, 120, 225],
            };
            for py in b.y0..b.y1 {
                for px in b.x0..b.x1 {
                    let o = ((py * w + px) * 4) as usize;
                    hud.px[o..o + 4].copy_from_slice(&fill);
                }
            }
            let mut tx = x + 3;
            if !it.hint.is_empty() {
                let hint = if it.style == Style::On { [195, 215, 250] } else { [125, 135, 150] };
                hud.text(tx, y + 2, it.hint, hint);
                tx += (it.hint.len() as u32 + 1) * ADVANCE;
            }
            hud.text(tx, y + 2, &it.text, rgb(it.style));
            hud.buttons.push(b);
        }
        hud
    }

    /// The key of the button at overlay pixel (`x`, `y`).
    pub fn hit(buttons: &[Button], x: u32, y: u32) -> Option<char> {
        buttons.iter().find(|b| x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1).map(|b| b.action)
    }

    fn text(&mut self, x: u32, y: u32, s: &str, rgb: [u8; 3]) {
        for (i, c) in s.bytes().enumerate() {
            let Some(g) = glyph(c) else { continue };
            for (row, bits) in FONT[g].iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) != 0 {
                        let (px, py) = (x + i as u32 * ADVANCE + col, y + row as u32);
                        if px < self.w && py < self.h {
                            let o = ((py * self.w + px) * 4) as usize;
                            self.px[o..o + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buttons_hit_where_drawn() {
        let rows = vec![
            vec![Item::text("SHEEP 1", Style::Plain)],
            vec![Item::button("O", "GLOW SOFT", 'o', true), Item::button("D", "DRIFT", 'd', false), Item::text("SPEED 1X", Style::Dim)],
            vec![Item::button("ESC", "QUIT", 'q', false)],
        ];
        let hud = Hud::panel(&rows, 1000, None);
        assert_eq!(hud.buttons.len(), 3);
        for b in &hud.buttons {
            assert_eq!(Hud::hit(&hud.buttons, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2), Some(b.action));
            assert!(b.x1 <= hud.w && b.y1 <= hud.h);
        }
        // the text line, the gaps and outside hit nothing
        assert_eq!(Hud::hit(&hud.buttons, 6, 5), None);
        assert_eq!(Hud::hit(&hud.buttons, hud.buttons[0].x1 + 2, hud.buttons[0].y0 + 4), None);
        assert_eq!(Hud::hit(&hud.buttons, hud.w + 5, 5), None);
        // a narrow panel wraps the row: the second button goes below the first
        let narrow = Hud::panel(&rows, 90, None);
        assert!(narrow.buttons[1].y0 > narrow.buttons[0].y0);
        assert!(narrow.w <= 100);
    }
}
