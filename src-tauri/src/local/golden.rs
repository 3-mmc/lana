// SPDX-License-Identifier: GPL-3.0-or-later
// What the models sherpa-onnx used to run (VAD, the sound tagger, speakers,
// Parakeet) gave on fixed clips, kept as JSON in tests/golden so that Lana's
// own runs of the same models can be held to them. Recorded once with sherpa,
// at commit 572d0a8; the two cannot run side by side, as each brings its own
// ONNX Runtime.
//
// Except the speakers: those are Lana's port on ONNX Runtime 1.30. The
// runtime inside sherpa-onnx 1.13.8 (1.28) gets CAM++ wrong when the fbank
// has some frame counts (100, 300 and 500 of the first 600; cosine 0.12 to
// the unoptimised graph's embedding, which 1.30 matches exactly), and so
// gave other turns. On 1.28.3 the port reproduced sherpa's turns exactly.
//
//   GOLDEN_RECORD=1 cargo test -- --ignored golden --nocapture   (record)
//   cargo test -- --ignored golden --nocapture                   (compare)
//   GOLDEN_DUMP=<file> ...                                       (and keep this run)
//
// The models are looked for where the earlier tests left them; LANA_GOLDEN_MODELS
// and LANA_GOLDEN_SPEAKERS point elsewhere.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::{audio, parakeet, speakers, tags, vad};

const RATE: u32 = 16_000;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn models() -> PathBuf {
    std::env::var("LANA_GOLDEN_MODELS").map(PathBuf::from).unwrap_or_else(|_| home().join(".cache/lana-agentb-models"))
}

fn speaker_models() -> speakers::Models {
    let dir = std::env::var("LANA_GOLDEN_SPEAKERS").map(PathBuf::from).unwrap_or_else(|_| home().join("lab/diarize/models"));
    speakers::models_at(&dir.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"), &dir.join("wespeaker_en_voxceleb_CAM++.onnx"), 4)
}

fn wav(path: &Path) -> Vec<f32> {
    audio::decode_16k_mono(std::fs::read(path).unwrap(), "a.wav").unwrap()
}

/// The clips: the two demo recordings, the two joined (a man speaking German,
/// then a woman speaking Georgian), and 116 s of mixed sounds.
fn clips() -> Vec<(&'static str, Vec<f32>)> {
    let de = wav(Path::new("../web/demo/de-vinson.wav"));
    let ka = wav(Path::new("../web/demo/ka-vinson.wav"));
    let both: Vec<f32> = de.iter().chain(&ka).copied().collect();
    vec![("de", de), ("ka", ka), ("de+ka", both), ("mix", wav(&models().join("mix.wav")))]
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn turns(t: &[speakers::Turn]) -> Value {
    json!(t.iter().map(|t| json!([t.start, t.end, t.speaker])).collect::<Vec<_>>())
}

/// Everything the four paths give on every clip.
fn run() -> Value {
    // GOLDEN_CACHE=<dir>: with optimised graphs kept there, as the app keeps them.
    if let Some(d) = std::env::var_os("GOLDEN_CACHE") {
        super::runtime::cache_in(d.into());
    }
    vad::use_model(models().join("vad/silero_vad_v5.onnx"));
    let tagger = tags::Tagger::load(&models().join("sounds/model.int8.onnx"), &models().join("sounds/labels.csv"), 4).unwrap();
    let para = parakeet::Parakeet::load(&models().join("parakeet-tdt-0.6b-v3/encoder.int8.onnx"), 4).unwrap();
    let sm = speaker_models();
    let mut out = serde_json::Map::new();
    for (name, x) in clips() {
        let secs = x.len() as f64 / RATE as f64;
        let speech: Vec<Value> = vad::speech(&x, RATE).unwrap().iter()
            .map(|r| json!([r.start, r.end])).collect();
        let hops: Vec<Value> = tagger.label(&x, RATE, |_| true).unwrap().into_iter()
            .map(|(t, heard)| json!([t, heard.into_iter().map(|(n, p)| json!([n, r3(p as f64)])).collect::<Vec<_>>()]))
            .collect();
        let mut clip = json!({ "vad": speech, "tags": hops });
        if name != "mix" {
            let words: Vec<Value> = para.words(&x, RATE, 0.0, secs).into_iter()
                .map(|(w, a, b)| json!([w, r3(a), r3(b)])).collect();
            clip["parakeet"] = json!(words);
            let auto = speakers::diarize(&sm, &x, 0, 0.7).unwrap();
            let mut held = speakers::Held { hash: String::new(), samples: x.clone(), turns: auto.clone(), windows: None };
            let k2 = speakers::recluster(&sm, &mut held, 2).unwrap().0;
            let k3 = speakers::recluster(&sm, &mut held, 3).unwrap().0;
            let (_, vecs) = held.windows.as_ref().unwrap();
            clip["speakers"] = json!({
                "auto": turns(&auto),
                "fixed2": turns(&speakers::diarize(&sm, &x, 2, 0.7).unwrap()),
                "recluster2": turns(&k2),
                "recluster3": turns(&k3),
                "embeddings": vecs.iter().take(3).map(|v| v.iter().map(|f| (*f as f64 * 1e5).round() / 1e5).collect::<Vec<_>>()).collect::<Vec<_>>(),
            });
        }
        out.insert(name.to_owned(), clip);
    }
    Value::Object(out)
}

fn golden_path() -> PathBuf {
    PathBuf::from("tests/golden/sherpa.json")
}

#[test]
#[ignore]
fn golden() {
    let now = run();
    if let Some(p) = std::env::var_os("GOLDEN_DUMP") {
        std::fs::write(p, serde_json::to_string_pretty(&now).unwrap()).unwrap();
    }
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        std::fs::create_dir_all(golden_path().parent().unwrap()).unwrap();
        std::fs::write(golden_path(), serde_json::to_string_pretty(&now).unwrap()).unwrap();
        println!("recorded {}", golden_path().display());
        return;
    }
    let then: Value = serde_json::from_str(&std::fs::read_to_string(golden_path()).unwrap()).unwrap();
    let mut bad = Vec::new();
    for (name, want) in then.as_object().unwrap() {
        let got = &now[name];
        compare(name, want, got, &mut bad);
    }
    for b in &bad {
        println!("{b}");
    }
    assert!(bad.is_empty(), "{} differences from sherpa-onnx", bad.len());
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// Differences that matter: VAD stretches more than one Silero window (32 ms)
/// apart, a hop whose likeliest sound changed, a speaker turn more than a
/// tenth of a second off or given to another voice, an embedding that turned
/// (cosine below 0.99), a Parakeet word read differently or more than one of
/// its 80 ms frames off. Parakeet is held to this on its own languages only:
/// on the Georgian clip its English is noise-level guessing (89% CER) that
/// changes even with the number of threads the joiner runs on.
fn compare(name: &str, want: &Value, got: &Value, bad: &mut Vec<String>) {
    let (w, g) = (want["vad"].as_array().unwrap(), got["vad"].as_array().unwrap());
    if w.len() != g.len() {
        bad.push(format!("{name} vad: {} stretches, now {}", w.len(), g.len()));
    } else {
        for (a, b) in w.iter().zip(g) {
            if (f(&a[0]) - f(&b[0])).abs() > 512.0 || (f(&a[1]) - f(&b[1])).abs() > 512.0 {
                bad.push(format!("{name} vad: {a} now {b}"));
            }
        }
    }
    let (w, g) = (want["tags"].as_array().unwrap(), got["tags"].as_array().unwrap());
    for (a, b) in w.iter().zip(g) {
        if a[1].get(0).map(|x| &x[0]) != b[1].get(0).map(|x| &x[0]) {
            bad.push(format!("{name} tags at {}: {} now {}", a[0], a[1], b[1]));
        }
    }
    if w.len() != g.len() {
        bad.push(format!("{name} tags: {} hops, now {}", w.len(), g.len()));
    }
    if let (Some(w), Some(g), false) = (want["parakeet"].as_array(), got["parakeet"].as_array(), name == "ka") {
        let text = |v: &[Value]| v.iter().map(|x| x[0].as_str().unwrap().to_owned()).collect::<Vec<_>>().join(" ");
        if text(w) != text(g) {
            bad.push(format!("{name} parakeet:\n  {}\n  {}", text(w), text(g)));
        } else {
            for (a, b) in w.iter().zip(g) {
                if (f(&a[1]) - f(&b[1])).abs() > 0.081 || (f(&a[2]) - f(&b[2])).abs() > 0.081 {
                    bad.push(format!("{name} parakeet: {a} now {b}"));
                }
            }
        }
    }
    if want["speakers"].is_object() {
        for k in ["auto", "fixed2", "recluster2", "recluster3"] {
            let (w, g) = (want["speakers"][k].as_array().unwrap(), got["speakers"][k].as_array().unwrap());
            let same = w.len() == g.len() && w.iter().zip(g).all(|(a, b)| {
                (f(&a[0]) - f(&b[0])).abs() <= 0.1 && (f(&a[1]) - f(&b[1])).abs() <= 0.1 && a[2] == b[2]
            });
            if !same {
                bad.push(format!("{name} speakers {k}:\n  {}\n  {}", Value::Array(w.clone()), Value::Array(g.clone())));
            }
        }
        let (w, g) = (want["speakers"]["embeddings"].as_array().unwrap(), got["speakers"]["embeddings"].as_array().unwrap());
        for (i, (a, b)) in w.iter().zip(g).enumerate() {
            let (a, b): (Vec<f64>, Vec<f64>) = (a.as_array().unwrap().iter().map(f).collect(), b.as_array().unwrap().iter().map(f).collect());
            let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let cos = dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|x| x * x).sum::<f64>().sqrt());
            if cos < 0.99 {
                bad.push(format!("{name} embedding {i}: cosine {cos:.4}"));
            }
        }
    }
}

