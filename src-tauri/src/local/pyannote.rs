// SPDX-License-Identifier: GPL-3.0-or-later
// Speaker diarization as sherpa-onnx did it for Lana, ported so the models run
// on Lana's own ONNX Runtime (runtime.rs): pyannote segmentation 3.0 over
// 10 s windows a second apart says who of up to three local voices speaks in
// each 17 ms frame; a CAM++ embedding of each local voice's single-speaker
// frames places it among all the others; complete-linkage clustering of those
// embeddings says which local voices are one speaker; and the frames are
// counted back into turns. Every step follows sherpa-onnx 1.13.8's
// offline-speaker-diarization-pyannote-impl.h, down to its rounding, so that
// the turns are the ones Lana gave before (tests/golden/sherpa.json).

use std::{path::Path, sync::Mutex};

use ort::value::Tensor;

use super::{frames, runtime::{self, Role}};

/// From the segmentation model's metadata (csukuangfj's export of
/// pyannote/segmentation-3.0): 10 s windows, frames 270 samples apart and 991
/// wide, three local speakers, powerset classes of at most two at once.
const WINDOW: usize = 160_000;
/// sherpa-onnx's default window shift, a tenth of the window.
const SHIFT: usize = 16_000;
const FRAME_SHIFT: usize = 270;
const FRAME_SIZE: usize = 991;
const LOCAL: usize = 3;
const RATE: f32 = 16_000.0;
/// A local voice heard in fewer frames than this in a window gets no
/// embedding of its own (sherpa-onnx's constant).
const MIN_FRAMES: i32 = 10;
/// Shorter turns are dropped; shorter gaps between one speaker's turns are
/// closed. Lana's settings since the start.
const MIN_DURATION_ON: f32 = 0.3;
const MIN_DURATION_OFF: f32 = 0.5;

/// Which local speakers each of the 7 powerset classes means: none, each one
/// alone, each pair.
const POWERSET: [[u8; LOCAL]; 7] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]];

/// A turn: start, end in seconds, speaker.
pub type Segment = (f32, f32, i32);

/// The speaker embedding model (CAM++, from wespeaker): 80-band fbank of the
/// samples at 16-bit scale, no mean taken off, one 512-dimensional vector.
///
/// Both speaker models stay on the CPU: they are small and run on short
/// inputs, and on a GTX 1080 the whole pass took 1.36 s on CUDA against
/// 0.84 s on the CPU.
pub struct Embedder {
    model: Mutex<runtime::Model>,
}

impl Embedder {
    pub fn load(model: &Path, threads: i32) -> Result<Self, String> {
        Ok(Embedder { model: Mutex::new(runtime::Model::load(model, Role::Light, threads)?) })
    }

    /// The embedding of some audio, or None if it is too short for a frame.
    pub fn embed(&self, samples: &[f32]) -> Result<Option<Vec<f32>>, String> {
        let err = |e: ort::Error| format!("a voice embedding failed: {e}");
        let scaled: Vec<f32> = samples.iter().map(|x| x * 32768.0).collect();
        let (n, feats) = frames::fbank(&scaled);
        if n == 0 {
            return Ok(None);
        }
        self.model.lock().unwrap().with(|session| {
            let x = Tensor::from_array(([1usize, n, frames::MEL_BINS], feats.clone())).map_err(err)?;
            let out = session.run(ort::inputs!["feats" => x]).map_err(err)?;
            Ok(Some(out["embs"].try_extract_tensor::<f32>().map_err(err)?.1.to_vec()))
        })
    }
}

/// Who speaks when in 16 kHz audio. `speakers` > 0 asks for exactly that
/// many; otherwise clusters closer than `threshold` (cosine distance) are one.
pub fn diarize(segmentation: &Path, embedder: &Embedder, threads: i32, audio: &[f32], speakers: i32, threshold: f32) -> Result<Vec<Segment>, String> {
    let n = audio.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let model = runtime::Model::load(segmentation, Role::Light, threads)?;
    let labels = segment(model, audio)?;
    let has_last = (n as i64 - WINDOW as i64) % SHIFT as i64 > 0;
    if labels.len() == 1 {
        // One window: its own three voices, no clustering.
        let keep = if has_last { (n / FRAME_SHIFT).min(labels[0].len()) } else { labels[0].len() };
        let rows: Vec<Vec<u8>> = labels[0][..keep].iter().map(|r| r.to_vec()).collect();
        return Ok(result(&rows));
    }
    let per_frame = speakers_per_frame(&labels);
    if per_frame.iter().all(|&k| k == 0) {
        return Ok(Vec::new());
    }
    let (pairs, spans) = local_voices(&labels);
    let mut kept = Vec::new();
    let mut vecs = Vec::new();
    for (pair, span) in pairs.into_iter().zip(spans) {
        let mut x = Vec::new();
        for (a, b) in span {
            let b = b.min(n);
            if b > a {
                x.extend_from_slice(&audio[a..b]);
            }
        }
        if let Some(e) = embedder.embed(&x)? {
            if !e.iter().any(|v| v.is_nan()) {
                kept.push(pair);
                vecs.push(e);
            }
        }
    }
    let clusters = cluster(&vecs, speakers, threshold);
    if std::env::var_os("LANA_DIARIZE_DEBUG").is_some() {
        for (i, (p, v)) in kept.iter().zip(&vecs).enumerate() {
            let d: Vec<String> = vecs.iter().map(|u| {
                let c: f32 = u.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() / (u.iter().map(|x| x * x).sum::<f32>().sqrt() * v.iter().map(|x| x * x).sum::<f32>().sqrt());
                format!("{:.4}", 1.0 - c)
            }).collect();
            eprintln!("voice {i} {p:?} -> {} | {}", clusters[i], d.join(" "));
        }
    }
    let Some(&max) = clusters.iter().max() else { return Ok(Vec::new()) };
    let k = max + 1;
    // Each window's frames under the global speaker each local voice became.
    let mut global = vec![[None; LOCAL]; labels.len()];
    for (&(c, s), &g) in kept.iter().zip(&clusters) {
        global[c][s] = Some(g);
    }
    let mut count = vec![vec![0i32; k]; total_frames(labels.len())];
    for (c, chunk) in labels.iter().enumerate() {
        let start = start_frame(c);
        for (f, row) in chunk.iter().enumerate() {
            let Some(slot) = count.get_mut(start + f) else { break };
            for s in 0..LOCAL {
                if let (1, Some(g)) = (row[s], global[c][s]) {
                    slot[g] += 1;
                }
            }
        }
    }
    if has_last {
        count.truncate((n / FRAME_SHIFT).min(count.len() - 1) + 1);
    }
    // In each frame, as many speakers as were heard there on average, the
    // ones heard in the most windows.
    let rows: Vec<Vec<u8>> = count.iter().zip(&per_frame).map(|(c, &want)| {
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|a, b| c[*b].cmp(&c[*a]));
        let mut row = vec![0u8; k];
        for &g in order.iter().take(want.max(0) as usize) {
            row[g] = 1;
        }
        row
    }).collect();
    Ok(result(&rows))
}

/// The segmentation of every 10 s window a second apart (the last one padded
/// with silence), as local-speaker activity per frame.
fn segment(mut model: runtime::Model, audio: &[f32]) -> Result<Vec<Vec<[u8; LOCAL]>>, String> {
    let n = audio.len();
    let mut windows: Vec<Vec<f32>> = Vec::new();
    let pad = |x: &[f32]| { let mut v = x.to_vec(); v.resize(WINDOW, 0.0); v };
    if n <= WINDOW {
        windows.push(pad(audio));
    } else {
        let full = (n - WINDOW) / SHIFT + 1;
        for i in 0..full {
            windows.push(audio[i * SHIFT..i * SHIFT + WINDOW].to_vec());
        }
        if (n - WINDOW) % SHIFT > 0 {
            windows.push(pad(&audio[full * SHIFT..]));
        }
    }
    let err = |e: ort::Error| format!("the speaker segmentation failed: {e}");
    let mut out = Vec::with_capacity(windows.len());
    for w in windows {
        let labels = model.with(|session| {
            let x = Tensor::from_array(([1usize, 1, WINDOW], w.clone())).map_err(err)?;
            let o = session.run(ort::inputs!["x" => x]).map_err(err)?;
            let (shape, y) = o["y"].try_extract_tensor::<f32>().map_err(err)?;
            let classes = shape[2] as usize;
            Ok(y.chunks(classes).map(|row| {
                let best = row.iter().enumerate().fold(0, |b, (i, v)| if *v > row[b] { i } else { b });
                POWERSET[best]
            }).collect::<Vec<_>>())
        })?;
        out.push(labels);
    }
    Ok(out)
}

/// Where window `c`'s first frame falls among all frames, rounded as
/// sherpa-onnx does (in single precision).
fn start_frame(c: usize) -> usize {
    ((c as f32 * SHIFT as f32 / FRAME_SHIFT as f32) as f64 + 0.5) as usize
}

fn total_frames(chunks: usize) -> usize {
    (WINDOW + (chunks - 1) * SHIFT) / FRAME_SHIFT + 1
}

/// How many speakers talk in each frame: the mean over the windows that
/// cover it, rounded.
fn speakers_per_frame(labels: &[Vec<[u8; LOCAL]>]) -> Vec<i32> {
    let t = total_frames(labels.len());
    let (mut count, mut weight) = (vec![0f32; t], vec![0f32; t]);
    for (c, chunk) in labels.iter().enumerate() {
        let start = start_frame(c);
        for (f, row) in chunk.iter().enumerate() {
            if let (Some(n), Some(w)) = (count.get_mut(start + f), weight.get_mut(start + f)) {
                *n += row.iter().map(|&x| x as f32).sum::<f32>();
                *w += 1.0;
            }
        }
    }
    count.iter().zip(&weight).map(|(c, w)| (c / (w + 1e-12) + 0.5) as i32).collect()
}

/// Every (window, local speaker) heard alone in at least MIN_FRAMES frames,
/// with the sample spans where it is (overlapped frames left out).
#[allow(clippy::type_complexity)]
fn local_voices(labels: &[Vec<[u8; LOCAL]>]) -> (Vec<(usize, usize)>, Vec<Vec<(usize, usize)>>) {
    let (mut pairs, mut spans) = (Vec::new(), Vec::new());
    for (c, chunk) in labels.iter().enumerate() {
        let alone: Vec<[u8; LOCAL]> = chunk.iter()
            .map(|r| if r.iter().map(|&x| x as i32).sum::<i32>() < 2 { *r } else { [0; LOCAL] })
            .collect();
        let frames = alone.len();
        let offset = c * SHIFT;
        let at = |f: usize| (f as f32 / frames as f32 * WINDOW as f32 + offset as f32) as usize;
        for s in 0..LOCAL {
            if alone.iter().map(|r| r[s] as i32).sum::<i32>() < MIN_FRAMES {
                continue;
            }
            let mut mine = Vec::new();
            let mut from = None;
            for (f, r) in alone.iter().enumerate() {
                match (r[s] != 0, from) {
                    (true, None) => from = Some(f),
                    (false, Some(a)) => {
                        mine.push((at(a), at(f)));
                        from = None;
                    }
                    _ => {}
                }
            }
            if let Some(a) = from {
                mine.push((at(a), at(frames - 1)));
            }
            pairs.push((c, s));
            spans.push(mine);
        }
    }
    (pairs, spans)
}

/// Complete-linkage clustering on cosine distance, cut at `k` clusters if
/// `k` > 0 or else where merging would join voices further apart than
/// `threshold`. Labels number the clusters by first member, as fastcluster's
/// cutree does.
fn cluster(vecs: &[Vec<f32>], k: i32, threshold: f32) -> Vec<usize> {
    let n = vecs.len();
    if n <= 1 {
        return vec![0; n];
    }
    let unit: Vec<Vec<f32>> = vecs.iter().map(|v| {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }).collect();
    let mut d = Condensed::new(n);
    for i in 0..n {
        for j in i + 1..n {
            let cos: f32 = unit[i].iter().zip(&unit[j]).map(|(a, b)| a * b).sum();
            d.set(i, j, (1.0 - cos as f64).max(0.0));
        }
    }
    let merges = complete_linkage(n, d);
    let steps = if k > 0 {
        let k = k as usize;
        if k > n || k < 2 { n - 1 } else { n - k }
    } else {
        merges.iter().take_while(|m| m.2 < threshold as f64).count()
    };
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &(a, b, _) in merges.iter().take(steps) {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[ra.max(rb)] = ra.min(rb);
    }
    let mut ids: Vec<usize> = Vec::new();
    (0..n).map(|i| {
        let r = find(&mut parent, i);
        ids.iter().position(|&x| x == r).unwrap_or_else(|| { ids.push(r); ids.len() - 1 })
    }).collect()
}

/// The upper triangle of a distance matrix, row by row.
struct Condensed {
    n: usize,
    d: Vec<f64>,
}

impl Condensed {
    fn new(n: usize) -> Self {
        Condensed { n, d: vec![0.0; n * (n - 1) / 2] }
    }
    fn ix(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        self.n * i - i * (i + 1) / 2 + j - i - 1
    }
    fn get(&self, i: usize, j: usize) -> f64 {
        self.d[self.ix(i, j)]
    }
    fn set(&mut self, i: usize, j: usize, v: f64) {
        let k = self.ix(i, j);
        self.d[k] = v;
    }
}

/// The merges of complete-linkage clustering, lowest first, each as one
/// member of either side and the height: the nearest-neighbour chain
/// algorithm, as fastcluster uses for this linkage, in O(n²).
fn complete_linkage(n: usize, mut d: Condensed) -> Vec<(usize, usize, f64)> {
    let mut active = vec![true; n];
    let mut merges = Vec::with_capacity(n - 1);
    let mut chain: Vec<usize> = Vec::new();
    while merges.len() < n - 1 {
        if chain.is_empty() {
            chain.push((0..n).find(|&i| active[i]).unwrap());
        }
        loop {
            let a = *chain.last().unwrap();
            let prev = chain.len().checked_sub(2).map(|i| chain[i]);
            // The nearest other cluster, the previous link on a tie.
            let mut best = prev;
            let mut dist = prev.map_or(f64::INFINITY, |p| d.get(a, p));
            for x in 0..n {
                if active[x] && x != a && d.get(a, x) < dist {
                    best = Some(x);
                    dist = d.get(a, x);
                }
            }
            let b = best.unwrap();
            if Some(b) == prev {
                chain.truncate(chain.len() - 2);
                let (keep, gone) = (a.min(b), a.max(b));
                for x in 0..n {
                    if active[x] && x != keep && x != gone {
                        let m = d.get(keep, x).max(d.get(gone, x));
                        d.set(keep, x, m);
                    }
                }
                active[gone] = false;
                merges.push((keep, gone, dist));
                break;
            }
            chain.push(b);
        }
    }
    merges.sort_by(|x, y| x.2.total_cmp(&y.2));
    merges
}

/// Frame activity per speaker -> turns: runs of active frames, a speaker's
/// turns joined across gaps up to MIN_DURATION_OFF, those no longer than
/// MIN_DURATION_ON dropped; sorted by start.
fn result(rows: &[Vec<u8>]) -> Vec<Segment> {
    let frames = rows.len();
    if frames == 0 {
        return Vec::new();
    }
    let speakers = rows[0].len();
    let scale = FRAME_SHIFT as f32 / RATE;
    let offset = 0.5 * FRAME_SIZE as f32 / RATE;
    let time = |f: usize| f as f32 * scale + offset;
    let mut out = Vec::new();
    for s in 0..speakers {
        let mut mine: Vec<(f32, f32)> = Vec::new();
        let mut from = (rows[0][s] > 0).then_some(0);
        for (f, row) in rows.iter().enumerate().skip(1) {
            match (from, row[s]) {
                (Some(a), 0) => {
                    mine.push((time(a), time(f)));
                    from = None;
                }
                (None, 1) => from = Some(f),
                _ => {}
            }
        }
        if let Some(a) = from {
            mine.push((time(a), time(frames - 1)));
        }
        // Join across short gaps, one pair at a time, as sherpa-onnx does.
        let mut i = 0;
        while i + 1 < mine.len() {
            let (a, b) = (mine[i], mine[i + 1]);
            if a.1 < b.0 && a.1 + MIN_DURATION_OFF >= b.0 {
                mine[i] = (a.0, b.1);
                mine.remove(i + 1);
                i = 0;
            } else {
                i += 1;
            }
        }
        out.extend(mine.into_iter().filter(|(a, b)| b - a > MIN_DURATION_ON).map(|(a, b)| (a, b, s as i32)));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_linkage_joins_the_close_pairs_first() {
        let v = vec![vec![1.0, 0.0], vec![0.99, 0.1], vec![0.0, 1.0], vec![0.1, 0.99]];
        assert_eq!(cluster(&v, 2, 0.0), [0, 0, 1, 1]);
        assert_eq!(cluster(&v, -1, 0.5), [0, 0, 1, 1]);
        assert_eq!(cluster(&v, -1, 0.001), [0, 1, 2, 3]);
        assert_eq!(cluster(&v, 9, 0.0), [0, 0, 0, 0], "more clusters than voices: fastcluster gives one");
    }

    #[test]
    fn turns_join_across_short_gaps_and_short_turns_go() {
        let on = |a: usize, b: usize, n: usize| (0..n).map(|f| vec![(a..b).contains(&f) as u8]).collect::<Vec<_>>();
        let mut rows = on(10, 100, 400);
        for f in 110..300 {
            rows[f][0] = 1;
        }
        let t = result(&rows);
        assert_eq!(t.len(), 1, "a 10-frame (0.17 s) gap is closed: {t:?}");
        let t = result(&on(10, 20, 400));
        assert!(t.is_empty(), "0.17 s is too short a turn");
    }
}

