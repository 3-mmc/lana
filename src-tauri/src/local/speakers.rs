// SPDX-License-Identifier: GPL-3.0-or-later
// Who is talking: pyannote segmentation 3.0 and CAM++ speaker embeddings,
// through sherpa-onnx. A port of the diarizer service Lana grew up with
// (diarize_server.py), keeping its two hard-won rules: a voice heard for under
// two seconds is not a speaker of its own, and re-clustering takes a count,
// never a threshold.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};

pub const RATE: u32 = 16_000;
/// When the count is inferred, a voice heard for less than this in the whole
/// recording is folded into its neighbour. On a 15 s single-speaker FLEURS
/// clip the clusterer made the first 1.3 s and one 0.4 s fragment a second
/// speaker: an embedding over a second or two is mostly noise.
const MIN_SPEAKER_S: f32 = 2.0;
/// Re-clustering embeds fixed windows across the speech found by the pass.
const WIN_S: f32 = 2.0;
const HOP_S: f32 = 1.0;
/// Only windows at least this long are clustered; shorter ones join the
/// nearest cluster afterwards. Clustering them too made the first split
/// "short windows against long ones" instead of one voice against another.
const MIN_CLUSTER_S: f32 = 1.2;

pub struct Models {
    pub segmentation: PathBuf,
    pub embedding: PathBuf,
    pub threads: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Turn {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

/// What a re-cluster needs from the pass: the audio, the speech map, and the
/// window embeddings once they have been computed.
pub struct Held {
    pub hash: String,
    pub samples: Vec<f32>,
    pub turns: Vec<Turn>,
    pub windows: Option<(Vec<(usize, f32, f32)>, Vec<Vec<f32>>)>,
}

fn embedding_config(m: &Models) -> SpeakerEmbeddingExtractorConfig {
    SpeakerEmbeddingExtractorConfig {
        model: Some(m.embedding.to_string_lossy().into_owned()),
        num_threads: m.threads,
        debug: false,
        provider: Some("cpu".into()),
    }
}

/// One full pass. `speakers` > 0 asks for exactly that many; otherwise the
/// count comes from `threshold` (smaller splits more) and the brief-voice rule.
pub fn diarize(m: &Models, samples: &[f32], speakers: i32, threshold: f32) -> Result<Vec<Turn>, String> {
    let cfg = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(m.segmentation.to_string_lossy().into_owned()),
                ..Default::default()
            },
            num_threads: m.threads,
            ..Default::default()
        },
        embedding: embedding_config(m),
        clustering: FastClusteringConfig {
            num_clusters: if speakers > 0 { speakers } else { -1 },
            threshold,
            ..Default::default()
        },
        min_duration_on: 0.3,
        min_duration_off: 0.5,
    };
    let sd = OfflineSpeakerDiarization::create(&cfg).ok_or("sherpa-onnx could not load the speaker models")?;
    let result = sd.process(samples).ok_or("speaker diarization failed")?;
    let mut turns: Vec<Turn> = result
        .sort_by_start_time()
        .into_iter()
        .map(|s| Turn { start: r3(s.start), end: r3(s.end), speaker: s.speaker })
        .collect();
    if speakers <= 0 {
        turns = absorb_brief_voices(turns);
    }
    Ok(renumber(turns))
}

/// Give the turns of a barely heard voice to the speaker beside them, the
/// briefest first, until every voice left has MIN_SPEAKER_S of speech or only
/// one is left. Same-voice turns that then touch or overlap become one.
fn absorb_brief_voices(mut turns: Vec<Turn>) -> Vec<Turn> {
    loop {
        let mut talk: Vec<(i32, f32)> = Vec::new();
        for t in &turns {
            match talk.iter_mut().find(|(s, _)| *s == t.speaker) {
                Some((_, d)) => *d += t.end - t.start,
                None => talk.push((t.speaker, t.end - t.start)),
            }
        }
        if talk.len() < 2 {
            break;
        }
        let (brief, said) = talk.iter().copied().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        if said >= MIN_SPEAKER_S {
            break;
        }
        let others: Vec<Turn> = turns.iter().copied().filter(|t| t.speaker != brief).collect();
        for t in turns.iter_mut().filter(|t| t.speaker == brief) {
            let mid = (t.start + t.end) / 2.0;
            let dist = |o: &Turn| if o.start <= mid && mid <= o.end { 0.0 } else { (mid - o.start).abs().min((mid - o.end).abs()) };
            t.speaker = others.iter().min_by(|a, b| dist(a).total_cmp(&dist(b))).unwrap().speaker;
        }
    }
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut joined: Vec<Turn> = Vec::new();
    for t in turns {
        match joined.last_mut() {
            Some(j) if j.speaker == t.speaker && t.start <= j.end + 1e-3 => j.end = j.end.max(t.end),
            _ => joined.push(t),
        }
    }
    joined
}

/// Speakers numbered by first appearance, as S1, S2... will read. The
/// clusterer's own labels are arbitrary.
fn renumber(turns: Vec<Turn>) -> Vec<Turn> {
    let mut order: Vec<i32> = Vec::new();
    turns
        .into_iter()
        .map(|mut t| {
            let n = order.iter().position(|&s| s == t.speaker).unwrap_or_else(|| {
                order.push(t.speaker);
                order.len() - 1
            });
            t.speaker = n as i32;
            t
        })
        .collect()
}

pub fn count(turns: &[Turn]) -> usize {
    let mut s: Vec<i32> = turns.iter().map(|t| t.speaker).collect();
    s.sort();
    s.dedup();
    s.len()
}

pub fn cues(turns: &[Turn]) -> Vec<Value> {
    // Through f64 and rounded, or 0.74 goes out as 0.7400000095367432.
    let ms = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
    turns.iter().map(|t| json!({ "start": ms(t.start), "end": ms(t.end), "speaker": t.speaker })).collect()
}

/// Fixed windows across the speech the pass found, never across silence.
/// Each window is a full WIN_S where the turn allows; the last is pushed flush
/// against the turn's end rather than left short.
fn windows_over(turns: &[Turn]) -> Vec<(usize, f32, f32)> {
    let mut out = Vec::new();
    for (ti, t) in turns.iter().enumerate() {
        if t.end - t.start <= WIN_S {
            out.push((ti, t.start, t.end));
            continue;
        }
        let mut starts = Vec::new();
        let mut x = t.start;
        while x + WIN_S <= t.end + 1e-6 {
            starts.push(x);
            x += HOP_S;
        }
        if starts.last().is_none_or(|&s| s + WIN_S < t.end - 1e-6) {
            starts.push(t.end - WIN_S);
        }
        out.extend(starts.into_iter().map(|s| (ti, s, s + WIN_S)));
    }
    out
}

fn normalise(v: &[f32]) -> Vec<f64> {
    let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() + 1e-9;
    v.iter().map(|x| *x as f64 / n).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Average-linkage agglomerative clustering on cosine distance, down to k.
fn agglomerate(v: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = v.len();
    let k = k.clamp(1, n.max(1));
    let mut d = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                d[i][j] = 1.0 - dot(&v[i], &v[j]);
            }
        }
    }
    let mut size = vec![1.0f64; n];
    let mut alive = vec![true; n];
    let mut label: Vec<usize> = (0..n).collect();
    for _ in 0..n.saturating_sub(k) {
        let (mut bi, mut bj, mut best) = (0, 0, f64::INFINITY);
        for i in 0..n {
            if !alive[i] { continue; }
            for j in (i + 1)..n {
                if alive[j] && d[i][j] < best {
                    (bi, bj, best) = (i, j, d[i][j]);
                }
            }
        }
        // Lance-Williams for average linkage.
        for x in 0..n {
            if x != bi && x != bj {
                let m = (size[bi] * d[bi][x] + size[bj] * d[bj][x]) / (size[bi] + size[bj]);
                d[bi][x] = m;
                d[x][bi] = m;
            }
        }
        size[bi] += size[bj];
        alive[bj] = false;
        for l in label.iter_mut() {
            if *l == bj { *l = bi; }
        }
    }
    let mut ids: Vec<usize> = label.clone();
    ids.sort();
    ids.dedup();
    label.iter().map(|l| ids.iter().position(|x| x == l).unwrap()).collect()
}

/// Re-partition the held turns into exactly `speakers` voices. Embeddings are
/// computed once and kept, so only the first re-cluster costs anything. A
/// turn whose windows agree keeps the pass's boundaries; a turn that splits
/// gets a boundary at window resolution, which the answer reports.
pub fn recluster(m: &Models, held: &mut Held, speakers: usize) -> Result<(Vec<Turn>, bool), String> {
    if held.turns.is_empty() {
        return Ok((Vec::new(), false));
    }
    if held.windows.is_none() {
        let ex = SpeakerEmbeddingExtractor::create(&embedding_config(m)).ok_or("sherpa-onnx could not load the voice model")?;
        let wins = windows_over(&held.turns);
        let mut vecs = Vec::with_capacity(wins.len());
        for &(_, a, b) in &wins {
            let (ia, ib) = ((a * RATE as f32) as usize, ((b * RATE as f32) as usize).min(held.samples.len()));
            let st = ex.create_stream().ok_or("could not start a voice embedding")?;
            st.accept_waveform(RATE as i32, &held.samples[ia.min(ib)..ib]);
            st.input_finished();
            vecs.push(ex.compute(&st).ok_or("a voice embedding failed")?);
        }
        held.windows = Some((wins, vecs));
    }
    let (wins, emb) = held.windows.as_ref().unwrap();
    let v: Vec<Vec<f64>> = emb.iter().map(|e| normalise(e)).collect();
    let mut long_ix: Vec<usize> = (0..wins.len()).filter(|&i| wins[i].2 - wins[i].1 >= MIN_CLUSTER_S).collect();
    if long_ix.is_empty() {
        long_ix = (0..wins.len()).collect();
    }
    let core = agglomerate(&long_ix.iter().map(|&i| v[i].clone()).collect::<Vec<_>>(), speakers);
    let mut labels: Vec<Option<usize>> = vec![None; wins.len()];
    for (&i, &l) in long_ix.iter().zip(&core) {
        labels[i] = Some(l);
    }
    let k = core.iter().max().map_or(0, |m| m + 1);
    let cents: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            let members: Vec<&Vec<f64>> = long_ix.iter().zip(&core).filter(|(_, &l)| l == c).map(|(&i, _)| &v[i]).collect();
            let mut sum = vec![0.0; v[0].len()];
            for m in &members {
                for (s, x) in sum.iter_mut().zip(m.iter()) { *s += x; }
            }
            normalise(&sum.iter().map(|x| *x as f32).collect::<Vec<_>>())
        })
        .collect();
    for i in 0..wins.len() {
        if labels[i].is_none() {
            labels[i] = Some((0..k).max_by(|&a, &b| dot(&v[i], &cents[a]).total_cmp(&dot(&v[i], &cents[b]))).unwrap_or(0));
        }
    }
    let mut out = Vec::new();
    let mut split = false;
    for (ti, t) in held.turns.iter().enumerate() {
        let mine: Vec<(f32, usize)> = wins.iter().zip(&labels).filter(|(w, _)| w.0 == ti).map(|(w, l)| (w.1, l.unwrap())).collect();
        if mine.is_empty() {
            continue;
        }
        if mine.iter().all(|(_, l)| *l == mine[0].1) {
            out.push(Turn { speaker: mine[0].1 as i32, ..*t });
            continue;
        }
        split = true;
        let (mut cur, mut start) = (mine[0].1, t.start);
        for &(wa, l) in &mine[1..] {
            if l != cur {
                let cut = (start + 1e-3).max((t.end - 1e-3).min(wa));
                out.push(Turn { start: r3(start), end: r3(cut), speaker: cur as i32 });
                start = cut;
                cur = l;
            }
        }
        out.push(Turn { start: r3(start), end: t.end, speaker: cur as i32 });
    }
    Ok((renumber(out), split))
}

pub fn models_at(segmentation: &Path, embedding: &Path, threads: i32) -> Models {
    Models { segmentation: segmentation.to_path_buf(), embedding: embedding.to_path_buf(), threads }
}

fn r3(x: f32) -> f32 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(start: f32, end: f32, speaker: i32) -> Turn {
        Turn { start, end, speaker }
    }

    #[test]
    fn a_voice_heard_for_under_two_seconds_joins_its_neighbour() {
        // The FLEURS Georgian clip: 1.3 s and 0.37 s of a "second speaker".
        let out = renumber(absorb_brief_voices(vec![t(0.74, 2.039, 0), t(2.039, 14.375, 1), t(13.97, 14.341, 0)]));
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start, out[0].end, out[0].speaker), (0.74, 14.375, 0));
    }

    #[test]
    fn two_real_speakers_stay_two() {
        let out = renumber(absorb_brief_voices(vec![t(0.0, 5.0, 0), t(5.0, 9.0, 1), t(9.0, 9.8, 2), t(9.8, 15.0, 0)]));
        assert_eq!(count(&out), 2);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn windows_never_cross_a_turn_and_the_last_is_flush() {
        let w = windows_over(&[t(0.0, 1.5, 0), t(2.0, 5.5, 0)]);
        assert_eq!(w[0], (0, 0.0, 1.5));
        assert_eq!(w.last().copied(), Some((1, 3.5, 5.5)));
    }

    #[test]
    fn agglomerate_finds_the_two_groups() {
        let v: Vec<Vec<f64>> = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
            .iter().map(|x| normalise(&[x[0] as f32, x[1] as f32])).collect();
        let l = agglomerate(&v, 2);
        assert_eq!(l[0], l[1]);
        assert_eq!(l[2], l[3]);
        assert_ne!(l[0], l[2]);
    }
}
