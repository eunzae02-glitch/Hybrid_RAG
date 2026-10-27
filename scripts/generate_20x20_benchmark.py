"""Create clearly labeled synthetic PDFs for a balanced RAG retrieval benchmark.

This generator does not import documents into the active library or measure search
quality. The normal documents are coherent *synthetic research notes*, not real
papers, and the adversarial documents are keyword repetition controls.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmark_20x20"
NORMAL = OUT / "normal_synthetic"
STUFFED = OUT / "keyword_stuffing_synthetic"
# The first ten topics follow DEBUG_REPORT.md; ten further topics add display,
# packaging, and power-device material. Every claim below describes a synthetic
# scenario and must not be represented as published experimental evidence.
TOPICS = [
    ("gaa", "Gate-All-Around Nanosheet Transistors", "GAA MBCFET nanosheet gate leakage subthreshold",
     "A gate surrounding the channel can strengthen electrostatic control over a narrow nanosheet.",
     "compare wrapped-gate and three-sided-gate cross sections under the same conceptual bias conditions",
     "the wrapped-gate case is described as having a more controlled channel barrier",
     "contact resistance and sheet-width variation can complicate the interpretation",
     "How does GAA gate wrapping affect short-channel leakage?"),
    ("euv", "High-NA EUV Patterning", "EUV High-NA lithography stochastic resist line edge",
     "A higher numerical aperture can resolve smaller features while reducing depth-of-focus margin.",
     "compare two focus settings while holding the mask pattern and resist family fixed",
     "the tighter focus setting is described as improving edge definition in the idealized pattern",
     "stochastic resist failures and mask effects are not captured by the simple comparison",
     "Why can High-NA EUV improve resolution while making process control harder?"),
    ("3ddram", "Vertical-Channel 3D DRAM Cells", "3D DRAM VCT vertical channel cell capacitor bitline",
     "Vertical-channel cells move a switching path out of the planar footprint and change the cell layout trade-off.",
     "compare a planar cell sketch with a stacked vertical-channel cell at equal logical capacity",
     "the vertical arrangement is described as relaxing lateral cell-area pressure",
     "capacitor integration and interlayer alignment remain separate constraints",
     "What problem does a vertical channel address in 3D DRAM?"),
    ("hbm", "HBM Through-Silicon Via Packaging", "HBM TSV microbump interposer bandwidth thermal",
     "Through-silicon vias provide vertical electrical paths between stacked memory dies.",
     "compare a long package trace with a short vertical interconnect at matched signaling assumptions",
     "the vertical path is described as reducing interconnect distance and enabling parallel channels",
     "stack heat removal and die yield are not resolved by shortening the electrical path",
     "How do TSVs support bandwidth in HBM stacks?"),
    ("3dnand", "Charge-Trap 3D NAND Strings", "3D NAND charge trap nitride wordline channel retention",
     "A charge-trap dielectric stores electrons locally along a vertically stacked memory string.",
     "compare trapped-charge and uncharged states under a hypothetical read-bias sequence",
     "a threshold-state difference is described as the basis of the stored-bit readout",
     "retention and disturbance need dedicated measurement rather than inference from this sketch",
     "What role does the charge-trap layer play in 3D NAND?"),
    ("ald", "ALD High-k Dielectric Conformality", "ALD high-k precursor conformality trench dielectric",
     "Self-limiting surface reactions can coat recessed features more uniformly than line-of-sight deposition.",
     "compare complete and incomplete precursor exposure within a model high-aspect-ratio trench",
     "the complete exposure case is described as producing more uniform sidewall coverage",
     "precursor transport and residue can limit coverage in real structures",
     "Why is ALD useful for coating deep high-aspect-ratio features?"),
    ("harc", "High-Aspect-Ratio Plasma Etching", "HARC plasma etch ion radical bowing selectivity",
     "Directional ions and reactive neutrals jointly determine depth progression and sidewall shape.",
     "compare two idealized ion-energy settings at equal target etch depth",
     "the balanced setting is described as limiting lateral sidewall erosion",
     "charging and local transport can still change the profile along the depth",
     "What causes bowing during high-aspect-ratio plasma etching?"),
    ("bspdn", "Backside Power Delivery Networks", "BSPDN backside power buried rail IR drop routing",
     "Moving power delivery behind the device layer can separate power routing from frontside signal lines.",
     "compare frontside-only and backside-assisted routing at matched logical load",
     "the backside-assisted layout is described as freeing some frontside routing space",
     "via resistance and wafer thinning introduce new integration costs",
     "How can backside power delivery reduce frontside routing congestion?"),
    ("cmp", "CMP Dishing and Pattern Density", "CMP slurry pad dishing erosion pattern density",
     "Chemical removal and pad contact together produce different removal rates across patterned areas.",
     "compare sparse and dense metal patterns under the same hypothetical polishing recipe",
     "the sparse wide feature is described as more susceptible to local dishing",
     "pad wear and local slurry supply can alter the trend",
     "Why does pattern density matter for CMP dishing?"),
    ("igzo", "IGZO Oxide TFT Bias Stability", "IGZO oxide TFT bias stress vacancy threshold voltage",
     "Charge trapping and oxygen-related defects can shift the threshold of an oxide thin-film transistor.",
     "compare an unstressed device with a device under prolonged gate bias in a synthetic protocol",
     "the stressed case is described as showing a threshold-state change",
     "ambient conditions and contact effects must be controlled before assigning a mechanism",
     "What can cause threshold-voltage drift in an IGZO TFT?"),
    ("oled", "OLED Pixel Aging Compensation", "OLED pixel aging luminance current compensation",
     "Organic emitter aging changes the relation between drive current and emitted luminance over use.",
     "compare uncompensated and feedback-compensated pixel driving in a simulated usage profile",
     "feedback is described as reducing the visible luminance difference between old and new pixels",
     "sensor noise and color-channel imbalance remain unresolved in the simplified model",
     "Why is compensation needed for OLED pixel aging?"),
    ("microled", "MicroLED Mass-Transfer Alignment", "MicroLED mass transfer alignment yield bonding pixel",
     "Placing many microscopic emitters onto a backplane creates placement and contact-alignment constraints.",
     "compare a loose and a tight placement tolerance in a synthetic assembly workflow",
     "the tighter tolerance is described as reducing open contacts at pixel sites",
     "repair throughput and inspection cost are not inferred from the alignment sketch",
     "How can alignment error affect MicroLED pixel yield?"),
    ("ltpo", "LTPO Display Backplane Refresh Control", "LTPO low temperature polycrystalline oxide refresh leakage power",
     "Combining high-mobility drive TFTs and low-leakage switching TFTs supports variable refresh operation.",
     "compare a fixed-refresh and variable-refresh driving sequence at the same static image",
     "the variable-refresh sequence is described as reducing unnecessary update events",
     "flicker and circuit overhead can offset the benefit in some conditions",
     "Why can LTPO reduce power during static display content?"),
    ("qd_oled", "Quantum-Dot Color Conversion", "QD OLED quantum dot color conversion blue emission optical",
     "A quantum-dot layer can absorb blue light and emit a narrower longer-wavelength spectrum.",
     "compare converted and unconverted pixel optical paths under the same source spectrum",
     "the converted path is described as shifting the output color band",
     "reabsorption and material stability require separate optical testing",
     "How does quantum-dot color conversion change OLED emission?"),
    ("tfe", "Thin-Film Encapsulation Barriers", "TFE thin film encapsulation moisture oxygen barrier pinhole",
     "Alternating inorganic and organic barrier layers can interrupt direct moisture paths to an emitter.",
     "compare a single brittle barrier with a multilayer stack containing an intermediate planarization layer",
     "the multilayer path is described as making a continuous defect channel less likely",
     "edge seals and mechanical bending remain independent failure paths",
     "Why use multilayer thin-film encapsulation for OLED panels?"),
    ("flexible", "Flexible Display Bending Reliability", "flexible display bending neutral plane strain crack",
     "Bending places layers at different distances from the neutral plane under different strain.",
     "compare two stack layouts with different positions of a brittle conductor layer",
     "the layout closer to the neutral plane is described as experiencing lower bending strain",
     "cyclic fatigue and adhesive slip cannot be estimated from a single-bend sketch",
     "Why does neutral-plane placement matter for flexible displays?"),
    ("sic", "SiC MOSFET Gate-Oxide Reliability", "SiC MOSFET gate oxide interface trap bias reliability",
     "Interface traps near the SiC-oxide boundary can change channel transport under electrical stress.",
     "compare an unstressed and bias-stressed gate interface in an illustrative device model",
     "the stressed interface is described as having a changed effective channel response",
     "real lifetime claims require temperature-accelerated measurement and uncertainty analysis",
     "How can interface traps affect SiC MOSFET gate reliability?"),
    ("gan", "GaN HEMT Dynamic On-Resistance", "GaN HEMT trapping dynamic RON buffer surface",
     "Trapped charge in buffer or surface regions can temporarily change the conducting channel after high-voltage stress.",
     "compare a steady-state read with a pulsed read after an idealized off-state stress interval",
     "the pulsed case is described as showing a transient on-resistance change",
     "trap location cannot be uniquely identified from resistance alone",
     "What causes dynamic on-resistance in GaN HEMTs?"),
    ("hybrid_bonding", "Copper Hybrid Bonding Alignment", "hybrid bonding copper oxide wafer alignment interconnect",
     "Direct dielectric and copper bonding can create fine-pitch vertical links between dies.",
     "compare aligned and offset pad layouts at the same nominal bonding pitch",
     "the aligned layout is described as providing a larger effective contact overlap",
     "surface cleanliness and void formation must also be examined",
     "Why is alignment important in copper hybrid bonding?"),
    ("thermal", "Thermal Management in Stacked Chips", "stacked chip thermal hotspot heat spreader TSV",
     "Vertical stacking shortens some electrical routes but creates longer and less uniform heat-removal paths.",
     "compare a stack with and without a hypothetical heat-spreading layer at equal power input",
     "the heat-spreader case is described as redistributing a localized temperature peak",
     "interface resistance and package boundary conditions can reverse the local trend",
     "Why is heat removal difficult in stacked semiconductor packages?"),
]


def normal_sections(topic):
    slug, title, terms, mechanism, method, observation, limitation, question = topic
    return [
        ("Abstract",
         f"This synthetic research note considers {title.lower()} as a retrieval case. {mechanism} "
         f"We {method}. In the deliberately simplified comparison, {observation}. "
         "The note supplies a coherent causal explanation for search evaluation, not measured research results."),
        ("1. Introduction",
         f"The central question is: {question} A useful answer must connect the named structure or process "
         f"to its physical consequence rather than list terms. {mechanism} "
         f"The terms {terms.replace(' ', ', ', 2)} identify the topic, but those words alone do not explain the mechanism."),
        ("2. Comparison Design",
         f"We define an illustrative baseline and variant to isolate one design choice: {method}. "
         "The comparison holds the remaining conceptual assumptions fixed and evaluates the predicted direction "
         "of change. No fabricated measurement table, claimed sample size, or statistical significance is used."),
        ("3. Qualitative Observations",
         f"Within this synthetic scenario, {observation}. The observation is interpreted through the following "
         f"physical link: {mechanism} A reader can test retrieval quality by asking for this link and checking "
         "whether the returned passage contains an explanation rather than repeated labels."),
        ("4. Discussion",
         f"The mechanism explains why the comparison could matter, but {limitation}. "
         "A single qualitative scenario cannot establish a device-performance number or rule out competing "
         "causes. A real study would need calibrated measurements and control samples before making a claim."),
        ("5. Conclusion",
         f"For the benchmark question, the answer should mention that {mechanism.lower()} "
         f"It should also acknowledge that {limitation}. This document is an invented, coherent control "
         "for comparing retrieval behavior and is not a published paper."),
    ]


def stuffing_text(topic, target_chars, section_index):
    terms = topic[2].split()
    # Rotate the keyword sequence by section so blocks differ in surface layout,
    # while remaining intentionally repetitive and devoid of evidence or method.
    offset = section_index % len(terms)
    sequence = terms[offset:] + terms[:offset]
    phrase = " ".join(sequence) + ". "
    text = (phrase * (target_chars // len(phrase) + 2))[:target_chars]
    return text.rsplit(" ", 1)[0].rstrip(" .") + "."


def render_pdf(path, topic, sections):
    title = topic[1]
    doc = pymupdf.open()
    for page_idx, pair in enumerate((sections[:2], sections[2:4], sections[4:6]), start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((43, 39), "SYNTHETIC BENCHMARK - NOT A REAL PAPER",
                         fontname="hebo", fontsize=9.5, color=(.63, 0, .20))
        page.insert_textbox(pymupdf.Rect(43, 53, 550, 110),
                            f"Research note: {title}", fontname="hebo", fontsize=15,
                            color=(.12, .12, .16))
        page.draw_line((43, 118), (552, 118), color=(.65, .65, .65), width=.7)
        for (heading, body), top in zip(pair, (151, 423)):
            page.insert_text((43, top), heading, fontname="hebo", fontsize=12,
                             color=(.25, .10, .20))
            remaining = page.insert_textbox(pymupdf.Rect(43, top + 16, 552, top + 242),
                                             body, fontname="helv", fontsize=10.3,
                                             lineheight=1.25, color=(.18, .18, .18))
            if remaining < 0:
                doc.close()
                raise ValueError(f"Text overflow: {path.name} page={page_idx} section={heading}")
        page.draw_line((43, 787), (552, 787), color=(.78, .78, .78), width=.5)
        page.insert_text((43, 803), "Synthetic retrieval fixture; no experimental measurements or citations.",
                         fontname="helv", fontsize=8, color=(.43, .43, .43))
        page.insert_text((510, 803), f"{page_idx} / 3", fontname="helv", fontsize=8,
                         color=(.43, .43, .43))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pdf(path, title, expected_sections):
    with pymupdf.open(path) as doc:
        if len(doc) != 3:
            raise AssertionError(f"Unexpected page count: {path}")
        text = "\n".join(page.get_text() for page in doc)
        if title not in text or not all(name in text for name in expected_sections):
            raise AssertionError(f"Lost title or section: {path}")
        if text.count("SYNTHETIC BENCHMARK - NOT A REAL PAPER") != 3:
            raise AssertionError(f"Missing synthetic label: {path}")
        return {"pages": len(doc), "extracted_characters": len(text), "sha256": sha256(path)}


def generate():
    if len(TOPICS) != 20 or len({topic[0] for topic in TOPICS}) != 20:
        raise AssertionError("The benchmark must have exactly 20 distinct topics")
    NORMAL.mkdir(parents=True, exist_ok=True)
    STUFFED.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "synthetic_retrieval_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/generate_20x20_benchmark.py",
        "basis": "DEBUG_REPORT.md topic categories; new balanced 20-pair fixture",
        "warning": "All PDFs are synthetic. They are not published papers or evidence of previous measured results.",
        "normal_count": 20, "keyword_stuffing_count": 20,
        "topics": [],
    }
    for idx, topic in enumerate(TOPICS, start=1):
        sections = normal_sections(topic)
        fake = [(name, stuffing_text(topic, len(body), n)) for n, (name, body) in enumerate(sections)]
        base = f"{idx:02d}_{topic[0]}"
        normal_path = NORMAL / f"normal_{base}.pdf"
        fake_path = STUFFED / f"stuffing_{base}.pdf"
        render_pdf(normal_path, topic, sections)
        render_pdf(fake_path, topic, fake)
        expected = [name for name, _ in sections]
        normal_meta = validate_pdf(normal_path, topic[1], expected)
        fake_meta = validate_pdf(fake_path, topic[1], expected)
        ratio = fake_meta["extracted_characters"] / normal_meta["extracted_characters"]
        if not .88 <= ratio <= 1.12:
            raise AssertionError(f"Imbalanced PDF text size: {base} ratio={ratio:.3f}")
        manifest["topics"].append({
            "id": base, "title": topic[1], "question": topic[7],
            "expected_relevant_file": normal_path.relative_to(OUT).as_posix(),
            "adversarial_file": fake_path.relative_to(OUT).as_posix(),
            "normal": normal_meta, "keyword_stuffing": fake_meta,
            "text_size_ratio": round(ratio, 3),
        })
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "questions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "topic", "question", "normal_file", "keyword_stuffing_file"))
        writer.writeheader()
        for item in manifest["topics"]:
            writer.writerow({"id": item["id"], "topic": item["title"],
                             "question": item["question"],
                             "normal_file": item["expected_relevant_file"],
                             "keyword_stuffing_file": item["adversarial_file"]})
    readme = """# 20:20 합성 검색 평가셋

이 폴더에는 **정상형 합성 연구 노트 20편**과 **같은 주제의 키워드 반복형 합성 문서 20편**이 있습니다. 모든 PDF는 실제 발표 논문이 아니며, 측정 결과·저자·학회·인용을 꾸며 넣지 않았습니다. 각 문서의 모든 페이지에도 합성 자료임을 표시했습니다.

- `normal_synthetic/`: 주제별 메커니즘, 가정, 비교 방법, 해석의 한계를 설명하는 문서
- `keyword_stuffing_synthetic/`: 동일 주제·동일 페이지 수·유사한 글자 수로 키워드를 반복하는 대조 문서
- `manifest.json`: 20개 질문, 각 질문의 정상형/반복형 파일 짝, 페이지·글자 수·SHA-256
- `questions.csv`: 엑셀에서 열기 쉬운 20개 질문과 문서 짝

이 폴더는 현재 앱의 `data/uploaded_pdfs/`와 분리돼 있습니다. 파일 생성만으로 검색 DB나 발표 자료의 점수가 바뀌지 않습니다. 평가하려면 별도 색인에 40편을 등록하고, `manifest.json`의 20개 질문을 **같은 데이터·코드·질문 순서**로 가중치별 실행한 뒤 상위 결과를 기록해야 합니다. 이 자료만으로 자기소개서의 과거 18/20 또는 15/20 결과가 입증되지는 않습니다.

재생성: `runtime\\python.exe -X utf8 -B scripts\\generate_20x20_benchmark.py`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    return {"normal": len(list(NORMAL.glob("*.pdf"))),
            "keyword_stuffing": len(list(STUFFED.glob("*.pdf"))),
            "manifest": str(OUT / "manifest.json")}


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False))
