//! Reading flam3 XML: the <flame> elements of a .flam3 file (Electric Sheep,
//! flam3-genome, Apophysis, Fractorium, or this program's own flock file).
//! A small tolerant parser for the attributes flam3 writes; unknown
//! variations are dropped with a note.

use crate::genome::{Affine, Genome, IDENTITY, Motion, VarUse, Xform};
use crate::var;

/// The attributes of the element starting at `s` (just after its name).
fn attrs(s: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(eq) = rest.find('=') {
        let name = rest[..eq].trim();
        let after = rest[eq + 1..].trim_start();
        let Some(q) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else { break };
        let body = &after[1..];
        let Some(end) = body.find(q) else { break };
        out.push((name.rsplit(char::is_whitespace).next().unwrap_or(name), &body[..end]));
        rest = &body[end + 1..];
    }
    out
}

fn nums(v: &str) -> Vec<f32> {
    v.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect()
}

fn affine(v: &str) -> Option<Affine> {
    let n = nums(v);
    (n.len() >= 6).then(|| [[n[0], n[1]], [n[2], n[3]], [n[4], n[5]]])
}

/// Elements (name, attribute text, inner text up to the matching close or
/// None for self-closing) at the top level of `xml`, in order.
fn elements(xml: &str) -> Vec<(&str, &str, Option<&str>)> {
    let mut out = Vec::new();
    let mut i = 0;
    let b = xml.as_bytes();
    while let Some(lt) = xml[i..].find('<') {
        let start = i + lt;
        if xml[start..].starts_with("<!--") {
            i = xml[start..].find("-->").map_or(xml.len(), |e| start + e + 3);
            continue;
        }
        if b.get(start + 1).is_some_and(|c| *c == b'/' || *c == b'?' || *c == b'!') {
            i = xml[start..].find('>').map_or(xml.len(), |e| start + e + 1);
            continue;
        }
        let Some(gt) = xml[start..].find('>') else { break };
        let tag = &xml[start + 1..start + gt];
        let self_closing = tag.ends_with('/');
        let tag = tag.trim_end_matches('/');
        let name_end = tag.find(char::is_whitespace).unwrap_or(tag.len());
        let name = &tag[..name_end];
        let attr = &tag[name_end..];
        i = start + gt + 1;
        if self_closing {
            out.push((name, attr, None));
        } else {
            let close = format!("</{name}>");
            let end = xml[i..].find(&close).map_or(xml.len(), |e| i + e);
            out.push((name, attr, Some(&xml[i..end])));
            i = (end + close.len()).min(xml.len());
        }
    }
    out
}

fn xform(attr: &str, fin: bool, unknown: &mut Vec<String>) -> Xform {
    let mut x = Xform::new();
    x.vars.clear();
    x.weight = if fin { 0.0 } else { 0.5 };
    x.animate = !fin;
    let mut params: Vec<(&str, f32)> = Vec::new();
    let mut symmetry = None;
    for (k, v) in attrs(attr) {
        let f = v.trim().parse::<f32>().ok();
        match k {
            "weight" => x.weight = f.unwrap_or(0.5),
            "color" => x.color = nums(v).first().copied().unwrap_or(0.0),
            "color_speed" => x.speed = f.unwrap_or(0.5),
            "symmetry" => symmetry = f,
            "animate" => x.animate = f.unwrap_or(1.0) != 0.0,
            "opacity" => x.opacity = f.unwrap_or(1.0),
            "coefs" => x.c = affine(v).unwrap_or(IDENTITY),
            "post" => x.post = affine(v).unwrap_or(IDENTITY),
            "chaos" => x.xaos = nums(v),
            "var_color" | "name" | "plotmode" => {}
            _ => {
                if let Some(vi) = var::VARS.iter().find(|vi| vi.name == k) {
                    if let Some(w) = f {
                        x.vars.push(VarUse { id: vi.id, w, p: var::defaults(vi.id) });
                    }
                } else if let (true, Some(f)) = (k.contains('_'), f) {
                    params.push((k, f));
                } else if f.is_some() && !unknown.iter().any(|u| u == k) {
                    unknown.push(k.to_string());
                }
            }
        }
    }
    // older genomes (parser.c): symmetry s means colour speed (1 - s) / 2,
    // and a symmetric xform (s > 0) does not turn with the loop
    if let Some(s) = symmetry {
        x.speed = (1.0 - s) / 2.0;
        x.animate = s <= 0.0;
    }
    for (k, v) in params {
        let mut found = false;
        for u in x.vars.iter_mut() {
            if let Some(i) = var::info(u.id).params.iter().position(|p| *p == k) {
                u.p[i] = v;
                found = true;
            }
        }
        if !found && !unknown.iter().any(|u| u == k) && !var::VARS.iter().any(|vi| vi.params.contains(&k)) {
            unknown.push(k.to_string());
        }
    }
    if x.vars.is_empty() {
        x.vars.push(VarUse { id: 0, w: 1.0, p: [0.0; crate::genome::NP] });
    }
    x
}

/// The <motion> children of an xform: deltas over the loop.
fn motions(inner: &str) -> Vec<Motion> {
    let mut out = Vec::new();
    for (name, attr, _) in elements(inner) {
        if name != "motion" {
            continue;
        }
        let mut m = Motion { freq: 1.0, c: [[0.0; 2]; 3], post: [[0.0; 2]; 3], ..Default::default() };
        let mut params: Vec<(&str, f32)> = Vec::new();
        for (k, v) in attrs(attr) {
            let f = v.trim().parse::<f32>().ok();
            match k {
                "motion_frequency" => m.freq = f.unwrap_or(1.0),
                "motion_function" => {
                    m.func = match v.trim() {
                        "triangle" => 1,
                        "hill" => 2,
                        _ => 0,
                    }
                }
                "weight" => m.weight = f.unwrap_or(0.0),
                "color" => m.color = nums(v).first().copied().unwrap_or(0.0),
                "color_speed" => m.speed = f.unwrap_or(0.0),
                "opacity" => m.opacity = f.unwrap_or(0.0),
                "coefs" => m.c = affine(v).unwrap_or([[0.0; 2]; 3]),
                "post" => m.post = affine(v).unwrap_or([[0.0; 2]; 3]),
                _ => {
                    if let Some(vi) = var::VARS.iter().find(|vi| vi.name == k) {
                        m.vars.push(VarUse { id: vi.id, w: f.unwrap_or(0.0), p: [0.0; crate::genome::NP] });
                    } else if let Some(f) = f {
                        params.push((k, f));
                    }
                }
            }
        }
        // parameter deltas go to their variation (added with weight 0 if absent)
        for (k, f) in params {
            if let Some(vi) = var::VARS.iter().find(|vi| vi.params.contains(&k)) {
                let i = vi.params.iter().position(|p| *p == k).unwrap();
                match m.vars.iter_mut().find(|u| u.id == vi.id) {
                    Some(u) => u.p[i] += f,
                    None => {
                        let mut p = [0.0; crate::genome::NP];
                        p[i] = f;
                        m.vars.push(VarUse { id: vi.id, w: 0.0, p });
                    }
                }
            }
        }
        out.push(m);
    }
    out
}

/// The flames in `xml` with their flock ratings (fleece_rating, 0 if none),
/// and the attribute names it did not understand.
pub fn parse(xml: &str) -> (Vec<(Genome, i32)>, Vec<String>) {
    let mut out = Vec::new();
    let mut unknown = Vec::new();
    let mut stack = vec![xml];
    while let Some(text) = stack.pop() {
        for (name, attr, inner) in elements(text) {
            if name != "flame" {
                if let Some(inner) = inner {
                    // <flames>, <pick> and the like hold flames
                    stack.push(inner);
                }
                continue;
            }
            let mut g = Genome::empty();
            g.palette = vec![[0.0; 3]; 256];
            let mut size = (1280.0, 720.0);
            let mut scale = None;
            let mut zoom = 0.0;
            let mut have_palette = false;
            let mut rating = 0;
            let mut symmetry_kind = None;
            for (k, v) in attrs(attr) {
                let n = nums(v);
                let f = n.first().copied();
                match k {
                    "name" => g.name = v.to_string(),
                    "nick" => g.nick = v.chars().filter(|c| !c.is_control()).take(32).collect(),
                    "fleece_nc" => g.nc = v.trim() == "1",
                    "size" if n.len() >= 2 => size = (n[0], n[1]),
                    "center" if n.len() >= 2 => g.center = [n[0], n[1]],
                    "scale" => scale = f,
                    "zoom" => zoom = f.unwrap_or(0.0),
                    "rotate" => g.rotate = f.unwrap_or(0.0),
                    "brightness" => g.brightness = f.unwrap_or(4.0),
                    "gamma" => g.gamma = f.unwrap_or(4.0),
                    "gamma_threshold" => g.gamma_threshold = f.unwrap_or(0.01),
                    "vibrancy" => g.vibrancy = f.unwrap_or(1.0),
                    "highlight_power" => g.highlight = f.unwrap_or(-1.0),
                    "background" if n.len() >= 3 => g.background = [n[0], n[1], n[2]],
                    "symmetry" => g.symmetry = f.unwrap_or(0.0) as i32,
                    "hue" => g.hue = f.unwrap_or(0.0),
                    "fleece_rating" => rating = f.unwrap_or(0.0) as i32,
                    "estimator_radius" => g.de[0] = f.unwrap_or(9.0),
                    "estimator_minimum" => g.de[1] = f.unwrap_or(0.0),
                    "estimator_curve" => g.de[2] = f.unwrap_or(0.4),
                    "palette_mode" => g.palette_linear = v.trim() == "linear",
                    "supersample" => g.supersample = f.unwrap_or(1.0).clamp(1.0, 4.0) as u32,
                    "filter" => g.filter = f.unwrap_or(0.5).clamp(0.0, 4.0),
                    "cam_pitch" => g.cam.pitch = f.unwrap_or(0.0),
                    "cam_yaw" => g.cam.yaw = f.unwrap_or(0.0),
                    "cam_perspective" => g.cam.persp = f.unwrap_or(0.0),
                    "cam_zpos" => g.cam.zpos = f.unwrap_or(0.0),
                    "cam_dof" => g.cam.dof = f.unwrap_or(0.0),
                    "palette" => {
                        g.palette_index = f.unwrap_or(0.0) as i32;
                        g.palette = crate::genome::palette(g.palette_index, 0.0);
                        have_palette = true;
                    }
                    _ => {}
                }
            }
            g.scale = scale.unwrap_or(size.0 / 4.0) * 2f32.powf(zoom) / size.0.max(1.0);
            let body = inner.unwrap_or("");
            for (en, ea, ei) in elements(body) {
                match en {
                    "xform" => {
                        let mut x = xform(ea, false, &mut unknown);
                        x.motion = ei.map(motions).unwrap_or_default();
                        g.xforms.push(x);
                    }
                    "finalxform" => {
                        let mut x = xform(ea, true, &mut unknown);
                        x.motion = ei.map(motions).unwrap_or_default();
                        g.fin = Some(x);
                    }
                    // flam3 turns <symmetry kind="N"/> into symmetry xforms when it loads
                    "symmetry" => {
                        let kind = attrs(ea).iter().find(|(k, _)| *k == "kind").and_then(|(_, v)| v.trim().parse::<i32>().ok());
                        if let Some(k) = kind {
                            symmetry_kind = Some(k);
                        }
                    }
                    "color" => {
                        let a = attrs(ea);
                        let i = a.iter().find(|(k, _)| *k == "index").and_then(|(_, v)| v.trim().parse::<usize>().ok());
                        let c = a.iter().find(|(k, _)| *k == "rgb").map(|(_, v)| nums(v));
                        if let (Some(i), Some(c)) = (i, c) {
                            if i < 256 && c.len() >= 3 {
                                g.palette[i] = [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0];
                                have_palette = true;
                            }
                        }
                    }
                    "palette" | "colors" => {
                        // hex RRGGBB (or 00RRGGBB / RRGGBBAA by count) in the text or a data attribute
                        let data = attrs(ea).into_iter().find(|(k, _)| *k == "data").map(|(_, v)| v).or(ei).unwrap_or("");
                        let hex: String = data.chars().filter(|c| c.is_ascii_hexdigit()).collect();
                        let per = if hex.len() >= 256 * 8 { 8 } else { 6 };
                        for i in 0..256.min(hex.len() / per) {
                            let e = &hex[i * per + (per - 6)..i * per + per];
                            let b = |j: usize| u8::from_str_radix(&e[j..j + 2], 16).unwrap_or(0) as f32 / 255.0;
                            g.palette[i] = [b(0), b(2), b(4)];
                            have_palette = true;
                        }
                    }
                    _ => {}
                }
            }
            if g.xforms.is_empty() {
                continue;
            }
            // an evolved Electric Sheep (no designer) is CC BY-NC
            if g.name.contains("electricsheep.") && g.nick.is_empty() {
                g.nc = true;
            }
            if let Some(k) = symmetry_kind {
                g.add_symmetry(&mut crate::rng::Rng::new(1), k);
            }
            // a few server genomes have center="nan nan" or scale="nan"
            if !(g.center[0].is_finite() && g.center[1].is_finite() && g.scale.is_finite() && g.scale > 0.0) {
                continue;
            }
            if !have_palette {
                g.palette = crate::genome::palette(0, 0.0);
            }
            if g.hue != 0.0 {
                g.palette = g
                    .palette
                    .iter()
                    .map(|&c| {
                        let mut h = crate::genome::rgb2hsv(c);
                        h[0] = (h[0] + g.hue * 6.0).rem_euclid(6.0);
                        crate::genome::hsv2rgb(h)
                    })
                    .collect();
            }
            g.xforms.truncate(crate::sample::MAXX);
            if g.name.is_empty() {
                g.name = format!("flame {}", out.len() + 1);
            } else {
                g.name = g.name.chars().filter(|c| !c.is_control()).take(40).collect();
            }
            out.push((g, rating));
        }
    }
    (out, unknown)
}

/// The flock file: every sheep as a <flame> with its rating.
pub fn write_flock(path: &str, sheep: &[(Genome, i32)]) -> std::io::Result<()> {
    let mut s = String::from("<flames name=\"fleece flock\">\n");
    for (g, r) in sheep {
        let x = g.to_flam3(1280, 720).replacen("<flame ", &format!("<flame fleece_rating=\"{r}\" "), 1);
        s.push_str(&x);
    }
    s.push_str("</flames>\n");
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = format!("{path}.new");
    std::fs::write(&tmp, s)?;
    std::fs::rename(tmp, path)
}
