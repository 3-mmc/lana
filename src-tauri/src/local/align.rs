// SPDX-License-Identifier: GPL-3.0-or-later
// Two uses of a CTC model's frame probabilities beyond reading the audio.
//
// Re-timing: once a stretch's labels have been corrected, the model finds
// where each corrected label lies (forced alignment). The labels are the
// annotator's; the model only places the boundaries, so this is not the
// dictionary-driven alignment the README sets Lana apart from: no lexicon
// predicts what was said.
//
// Search by example: a stretch's phones, or phones typed in, are looked for
// across the whole recording (keyword spotting over the frames), so a
// documentary linguist can find the other tokens of a word without having
// transcribed them yet.

use super::ctc::{Piece, BLANK, FRAME_S};
use super::frames::Mat;

/// A phone pass's frames, held for search.
pub struct Kept {
    pub hash: String,
    pub model: String,
    pub pieces: Vec<Piece>,
}

/// The key a recording is held under: the first 16 bytes of its SHA-256, as
/// the speaker pass uses.
pub fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// A label as the model's symbols, longest match first; characters the
/// model has no symbol for are skipped (a stress mark, a tie bar). Tried in
/// lower case first, since the word models write lower case ("Teil" would
/// otherwise be scored as a capital T the model never wrote), then as
/// written; IPA has no case, except small capitals, which are other letters.
pub fn tokenize(label: &str, symbols: &[Option<String>]) -> Vec<usize> {
    let mut by_text: Vec<(&str, usize)> = symbols
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_deref().filter(|s| *s != " ").map(|s| (s, i)))
        .collect();
    by_text.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    let lower = label.to_lowercase();
    let mut out = Vec::new();
    for text in [lower.as_str(), label] {
        let mut rest = text;
        out.clear();
        while !rest.is_empty() {
            match by_text.iter().find(|(s, _)| rest.starts_with(s)) {
                Some((s, i)) => {
                    out.push(*i);
                    rest = &rest[s.len()..];
                }
                None => {
                    let skip = rest.chars().next().map_or(1, char::len_utf8);
                    rest = &rest[skip..];
                }
            }
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

/// The symbol of CTC state `s`: blanks between and around the symbols.
fn label(ids: &[usize], s: usize) -> usize {
    if s % 2 == 0 { BLANK } else { ids[s / 2] }
}

/// Whether state `s` may be reached from `s - 2`, skipping the blank
/// between: only between two different symbols.
fn may_skip(ids: &[usize], s: usize) -> bool {
    s >= 2 && s % 2 == 1 && ids[s / 2] != ids[s / 2 - 1]
}

/// Forced alignment of `ids` to every frame of `m`: for each symbol, the
/// first and last frame it holds, and its probability at its surest frame
/// (with the blank set aside, as in decoding). None when the stretch is too
/// short to hold them.
pub fn force(m: &Mat, ids: &[usize]) -> Option<Vec<(usize, usize, f32)>> {
    let (t_n, s_n) = (m.t, 2 * ids.len() + 1);
    if ids.is_empty() || t_n == 0 {
        return None;
    }
    let neg = f32::NEG_INFINITY;
    let mut d = vec![neg; s_n];
    let mut back = vec![0u8; t_n * s_n];
    d[0] = m.row(0)[BLANK];
    if s_n > 1 {
        d[1] = m.row(0)[ids[0]];
    }
    for t in 1..t_n {
        let row = m.row(t);
        let mut next = vec![neg; s_n];
        for s in 0..s_n {
            let mut best = (d[s], 0u8);
            if s >= 1 && d[s - 1] > best.0 { best = (d[s - 1], 1); }
            if may_skip(ids, s) && d[s - 2] > best.0 { best = (d[s - 2], 2); }
            if best.0 > neg {
                next[s] = best.0 + row[label(ids, s)];
                back[t * s_n + s] = best.1;
            }
        }
        d = next;
    }
    let mut s = if s_n >= 2 && d[s_n - 2] > d[s_n - 1] { s_n - 2 } else { s_n - 1 };
    if d[s] == neg {
        return None;
    }
    let mut span: Vec<Option<(usize, usize, usize)>> = vec![None; ids.len()];
    for t in (0..t_n).rev() {
        if s % 2 == 1 {
            let k = s / 2;
            let lp = m.row(t)[ids[k]];
            span[k] = Some(match span[k] {
                None => (t, t, t),
                Some((_, last, best)) => (t, last, if lp > m.row(best)[ids[k]] { t } else { best }),
            });
        }
        if t > 0 {
            s -= back[t * s_n + s] as usize;
        }
    }
    span.iter()
        .zip(ids)
        .map(|(sp, &id)| {
            let (first, last, best) = (*sp)?;
            let row = m.row(best);
            let rest = (1.0 - row[BLANK].exp()).max(1e-6);
            Some((first, last, (row[id].exp() / rest).min(1.0)))
        })
        .collect()
}

/// One place the sought phones were found.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub start: f64,
    pub end: f64,
    /// 1 when the model's own best reading there is exactly the phones
    /// sought, falling as it would rather have heard something else.
    pub score: f32,
}

/// Where in the held frames `ids` were most likely said, best first, at most
/// `max` places, none overlapping another.
///
/// Each frame's probabilities are taken relative to that frame's likeliest
/// symbol, so a path costs nothing where the model's best reading agrees with
/// the phones sought and pays for each frame where it would rather hear
/// something else. A path may begin at any frame with the first phone and end
/// at any frame with the last; its cost per phone gives the score. Places that
/// straddle two pieces are not found.
pub fn spot(pieces: &[Piece], ids: &[usize], max: usize, min_score: f32) -> Vec<Hit> {
    let l = ids.len();
    if l == 0 {
        return Vec::new();
    }
    let s_n = 2 * l + 1;
    let neg = f32::NEG_INFINITY;
    let mut found: Vec<Hit> = Vec::new();
    for piece in pieces {
        let m = &piece.mat;
        // Score and start frame of the best path into each state.
        let mut d: Vec<(f32, usize)> = vec![(neg, 0); s_n];
        for t in 0..m.t {
            let row = m.row(t);
            let top = row.iter().cloned().fold(neg, f32::max);
            let mut next = vec![(neg, 0); s_n];
            for s in 1..s_n {
                let mut best = d[s];
                if d[s - 1].0 > best.0 { best = d[s - 1]; }
                if may_skip(ids, s) && d[s - 2].0 > best.0 { best = d[s - 2]; }
                // A new path starts here with the first phone.
                if s == 1 && 0.0 > best.0 {
                    best = (0.0, t);
                }
                if best.0 > neg {
                    next[s] = (best.0 + row[label(ids, s)] - top, best.1);
                }
            }
            d = next;
            let (score, first) = d[s_n - 2];
            if score > neg {
                let score = (score / l as f32).exp();
                // A match far longer than its phones could be said in is a
                // path that waited out a pause or a stretch of other speech.
                let long = (t + 1 - first) as f64 * FRAME_S > 0.3 + 0.15 * l as f64;
                if score >= min_score && !long {
                    found.push(Hit {
                        start: piece.offset + first as f64 * FRAME_S,
                        end: piece.offset + (t + 1) as f64 * FRAME_S,
                        score,
                    });
                }
            }
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out: Vec<Hit> = Vec::new();
    for h in found {
        if out.len() >= max {
            break;
        }
        if out.iter().all(|o| h.end <= o.start || h.start >= o.end) {
            out.push(h);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames where each listed symbol is near certain, from symbols 0..v.
    fn mat(v: usize, seq: &[usize]) -> Mat {
        let mut lp = Vec::new();
        for &k in seq {
            for j in 0..v {
                lp.push(if j == k { (0.97f32).ln() } else { (0.03 / (v - 1) as f32).ln() });
            }
        }
        Mat { t: seq.len(), v, lp }
    }

    #[test]
    fn forced_alignment_finds_each_symbol() {
        // blank a a blank b blank blank b blank
        let m = mat(4, &[0, 1, 1, 0, 2, 0, 0, 2, 0]);
        let a = force(&m, &[1, 2, 2]).unwrap();
        assert_eq!((a[0].0, a[0].1), (1, 2));
        assert_eq!(a[1].0, 4);
        assert_eq!(a[2].0, 7);
        assert!(a[0].2 > 0.9);
    }

    #[test]
    fn a_repeated_symbol_needs_a_blank_between() {
        assert!(force(&mat(3, &[1, 1]), &[1, 1]).is_none());
        assert!(force(&mat(3, &[1, 0, 1]), &[1, 1]).is_some());
    }

    #[test]
    fn spotting_finds_both_places_and_ranks_the_clean_one_first() {
        // a b at frames 1-2, a c at 5-6, a b again at 9-10.
        let m = mat(4, &[0, 1, 2, 0, 0, 1, 3, 0, 0, 1, 2, 0]);
        let hits = spot(&[Piece { offset: 0.0, mat: m }], &[1, 2], 5, 0.01);
        assert!(hits.len() >= 2);
        assert!(hits[0].score > 0.9 && hits[1].score > 0.9);
        let starts: Vec<f64> = hits.iter().take(2).map(|h| h.start).collect();
        assert!(starts.contains(&0.02) && starts.contains(&0.18), "{starts:?}");
        if let Some(h) = hits.get(2) {
            assert!(h.score < 0.5, "the a-c place scores low: {h:?}");
        }
    }

    #[test]
    fn labels_are_split_into_the_models_symbols() {
        let syms: Vec<Option<String>> = ["<blk>", "t", "ʼ", "ts", "a"].iter().map(|s| Some(s.to_string())).collect();
        assert_eq!(tokenize("tsʼa", &syms), vec![3, 2, 4]);
        assert_eq!(tokenize("t͡a", &syms), vec![1, 4]);
        assert_eq!(tokenize("TA", &syms), vec![1, 4]);
        let caps: Vec<Option<String>> = ["<blk>", "T", "t", "a"].iter().map(|s| Some(s.to_string())).collect();
        assert_eq!(tokenize("Ta", &caps), vec![2, 3]);
    }
}
