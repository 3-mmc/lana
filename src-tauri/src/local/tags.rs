// SPDX-License-Identifier: GPL-3.0-or-later
// Sounds that are not words, for an events tier: an AudioSet tagger (k2-fsa's
// zipformer small, Apache 2.0, 527 classes) run over a sliding window, each
// window given its likeliest sound, and runs of the same sound merged into
// one interval.
//
// Why this tagger and not CED mini (10 MB, better on AudioSet's own 10 s
// clips): on 2 s windows CED called the German demo clip "Clicking" and
// "Speech synthesizer" at 0.3-0.5 throughout, and wavered between "Cat" and
// "Animal"; the zipformer gave "Speech" at 0.99 and nothing else above 0.11.
//
// The tagger names a whole window, so the boundaries are only as fine as the
// hop (HOP_S): an events tier says roughly where a laugh or a cough is, for
// finding it, and its edges are to be set by ear before they are measured.

use std::{collections::HashSet, path::Path, sync::Mutex};

use ort::value::Tensor;
use serde_json::{json, Value};

use super::{frames, runtime::{self, Role}};

/// Each window is this long, and windows start this far apart, so each sound
/// is heard twice. A window of one second is too short for the tagger to
/// tell a cough from a consonant; four smears a short laugh over its
/// neighbours.
pub const WIN_S: f64 = 2.0;
pub const HOP_S: f64 = 1.0;
/// A window's sound is named only when the tagger is at least this sure of
/// it; AudioSet scores are independent per class, not shares of one.
const THRESHOLD: f32 = 0.3;
/// The tagger hears level: the FLEURS Georgian clip, peaking near -50 dBFS,
/// was "Silence" at 0.9 throughout, and "Speech" at 0.99 once its peak was
/// raised to 0.9. So the audio it is given is raised to that peak, by at most
/// this factor (30 dB), so a recording of a quiet room stays quiet.
const MAX_GAIN: f32 = 31.6;
/// How many of the top classes to look through for one that is not speech.
const TOP_K: usize = 12;

/// Speech, in AudioSet's words. These are left out, since the word tier
/// already says where speech is, and a tier where nearly every second reads
/// "Speech" would hide the few seconds of laughter it exists to show. Their
/// presence still matters: below them the tagger is asked for the next
/// class, so a laugh during speech is still named.
const SPEECH: &[&str] = &[
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Babbling",
];
/// Classes that describe the room rather than a sound in it; they would sit
/// under every word of a recording made indoors.
const ROOM: &[&str] = &[
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation", "Echo",
];

pub struct Tagger {
    model: Mutex<runtime::Model>,
    labels: Vec<String>,
}

/// One labelled stretch: label, start, end, and the tagger's highest score in it.
pub type Event = (String, f64, f64, f32);
/// One hop: its start, and the sounds heard in it, likeliest first.
pub type Hop = (f64, Vec<(String, f32)>);

impl Tagger {
    pub fn load(model: &Path, labels: &Path, threads: i32) -> Result<Self, String> {
        let text = std::fs::read_to_string(labels).map_err(|e| format!("{}: {e}", labels.display()))?;
        let labels = read_labels(&text)?;
        let model = runtime::Model::load(model, Role::Heavy, threads)?;
        Ok(Tagger { model: Mutex::new(model), labels })
    }

    /// The `TOP_K` likeliest sounds in a stretch of 16 kHz audio, likeliest
    /// first. The model takes the same 80-band fbank as ZIPA and gives a
    /// probability per class, each on its own.
    fn top(&self, samples: &[f32]) -> Result<Vec<(String, f32)>, String> {
        let err = |e: ort::Error| format!("the sound tagger failed: {e}");
        let (frames, feats) = frames::fbank(samples);
        if frames == 0 {
            return Ok(Vec::new());
        }
        let p = self.model.lock().unwrap().with(|session| {
            let x = Tensor::from_array(([1usize, frames, frames::MEL_BINS], feats.clone())).map_err(err)?;
            let lens = Tensor::from_array(([1usize], vec![frames as i64])).map_err(err)?;
            let out = session.run(ort::inputs![x, lens]).map_err(err)?;
            Ok(out[0].try_extract_tensor::<f32>().map_err(err)?.1.to_vec())
        })?;
        let mut order: Vec<usize> = (0..p.len().min(self.labels.len())).collect();
        order.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
        Ok(order.into_iter().take(TOP_K).map(|i| (self.labels[i].clone(), p[i])).collect())
    }

    /// The sounds of each hop of 16 kHz audio, likeliest first: those the
    /// tagger is sure enough of that are neither speech nor the room, so an
    /// empty list where nothing else was heard. `on_hop` gets the seconds
    /// done; it returns false to stop.
    pub fn label(&self, samples: &[f32], rate: u32, mut on_hop: impl FnMut(f64) -> bool) -> Result<Vec<Hop>, String> {
        let skip: HashSet<&str> = SPEECH.iter().chain(ROOM).copied().collect();
        let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
        let gain = (0.9 / peak.max(1e-6)).min(MAX_GAIN);
        let samples: Vec<f32> = samples.iter().map(|v| v * gain).collect();
        let (win, hop) = ((WIN_S * rate as f64) as usize, (HOP_S * rate as f64) as usize);
        let mut out = Vec::new();
        let mut at = 0usize;
        // One window centred on each hop: hop k covers [k, k+1) * HOP_S and
        // is judged by the window around it, clipped at the ends.
        while at < samples.len() {
            let centre = at + hop / 2;
            let a = centre.saturating_sub(win / 2);
            let b = (a + win).min(samples.len());
            let a = b.saturating_sub(win).min(a);
            let events = self.top(&samples[a..b])?;
            if std::env::var_os("LANA_TAGS_DEBUG").is_some() {
                let top: Vec<String> = events.iter().take(5).map(|(n, p)| format!("{n} {p:.2}")).collect();
                eprintln!("{:6.1}: {}", at as f64 / rate as f64, top.join(" | "));
            }
            let heard = events.into_iter()
                .filter(|(n, p)| *p >= THRESHOLD && !skip.contains(n.as_str()))
                .collect();
            out.push((at as f64 / rate as f64, heard));
            at += hop;
            if !on_hop((at.min(samples.len())) as f64 / rate as f64) {
                return Err("stopped".into());
            }
        }
        Ok(out)
    }
}

/// AudioSet's class list, `index,mid,"display name"` under a header line,
/// the indices in order.
fn read_labels(text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, ',');
        let (index, _mid, name) = (parts.next(), parts.next(), parts.next().unwrap_or("").trim());
        if index.and_then(|i| i.trim().parse::<usize>().ok()) != Some(out.len()) {
            return Err(format!("labels.csv: line {} is out of order", out.len() + 2));
        }
        out.push(name.trim_matches('"').to_owned());
    }
    Ok(out)
}

/// Hops -> events. An event goes on for as long as its sound is still among
/// those heard, even when another has become likelier: AudioSet's classes
/// nest ("Animal", "Cat", "Domestic animals, pets"), and which of them comes
/// first changes from window to window, which would cut one cat into five
/// intervals. Otherwise the hop's likeliest sound starts a new event.
pub fn events(hops: &[Hop], duration: f64) -> Vec<Event> {
    let mut out: Vec<Event> = Vec::new();
    for (t, heard) in hops {
        let end = (t + HOP_S).min(duration);
        if let Some(e) = out.last_mut().filter(|e| (e.2 - t).abs() < 1e-6) {
            if let Some((_, p)) = heard.iter().find(|(n, _)| *n == e.0) {
                e.2 = end;
                e.3 = e.3.max(*p);
                continue;
            }
        }
        if let Some((name, p)) = heard.first() {
            out.push((name.clone(), *t, end, *p));
        }
    }
    out
}

/// Events -> the cue contract: one cue and one word per event, the word's
/// text the sound's name. `event` marks the cue as a sound, so the page sends
/// it to the events tier rather than taking it for a transcription; `conf`
/// is the tagger's highest score over the stretch.
pub fn cues(events: &[Event]) -> Vec<Value> {
    events.iter().map(|(name, a, b, p)| {
        let (a, b) = (r3(*a), r3(*b));
        json!({ "start": a, "end": b, "event": name,
                "words": [{ "text": name, "start": a, "end": b, "conf": r3(*p as f64),
                            "phones": [], "phone_times": [], "aligned": true }] })
    }).collect()
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_keep_commas_inside_quotes() {
        let l = read_labels("index,mid,display_name\n0,/m/09x0r,\"Speech\"\n1,/m/05zppz,\"Male speech, man speaking\"\n").unwrap();
        assert_eq!(l, ["Speech", "Male speech, man speaking"]);
    }

    #[test]
    fn runs_of_one_sound_merge() {
        let s = |v: &[(&str, f32)]| v.iter().map(|(n, p)| (n.to_string(), *p)).collect::<Vec<_>>();
        let hops = vec![(0.0, s(&[("Laughter", 0.6)])), (1.0, s(&[("Laughter", 0.8)])), (2.0, s(&[])),
                        (3.0, s(&[("Cough", 0.4)])), (4.0, s(&[("Animal", 0.9), ("Cat", 0.8)])),
                        (5.0, s(&[("Cat", 0.9), ("Animal", 0.7)])), (6.0, s(&[("Cat", 0.6)]))];
        let e = events(&hops, 6.5);
        assert_eq!(e, vec![("Laughter".into(), 0.0, 2.0, 0.8), ("Cough".into(), 3.0, 4.0, 0.4), ("Animal".into(), 4.0, 6.0, 0.9),
                           ("Cat".into(), 6.0, 6.5, 0.6)]);
        let c = cues(&e);
        assert_eq!(c[0]["event"], json!("Laughter"));
        assert_eq!(c[0]["words"][0]["text"], json!("Laughter"));
    }
}
