// Editable paper figure: original evaluation frames and native PowerPoint text.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { autoFontSize } = require('./helpers/text');
const { imageSizingContain } = require('./helpers/image');
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require('./helpers/layout');

const ROOT = __dirname;
const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'manifest.json')));
const titles = JSON.parse(fs.readFileSync(path.join(ROOT, 'display_titles.json')));
const pptx = new pptxgen();
const WIDTH = 26.6666;
const HEIGHT = 24.0;
const FONT = 'Liberation Serif';
const COLORS = { back: '0072B2', left: 'A86A00', right: '007D5D', top: '9E4C7E' };
const SUITES = { spatial: 'Spatial', object: 'Object', goal: 'Goal', long: 'Long' };
const VIEWS = { back: 'Back', left: 'Left', right: 'Right', top: 'Top' };
const VIEW_ORDER = ['back', 'left', 'right', 'top'];
const PROGRESS = ['Start', '20%', '40%', '60%', '80%', 'End'];
const OUTER = 0.16;
const QUAD_GAP = 0.26;
const QUAD_WIDTH = (WIDTH - 2 * OUTER - QUAD_GAP) / 2;
const FRAME_GAP = 0.035;
const FRAME_SIZE = (QUAD_WIDTH - 5 * FRAME_GAP) / 6;

pptx.defineLayout({ name: 'PAPER_FIGURE', width: WIDTH, height: HEIGHT });
pptx.layout = 'PAPER_FIGURE';
pptx.author = 'ACPD research';
pptx.subject = 'Randomly sampled ACPD evaluation episodes, four views by four LIBERO suites';
pptx.title = 'ACPD rollouts across four deployment views and four LIBERO suites';
pptx.company = 'ACPD';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'en-US' };

function text(slide, content, x, y, w, h, size, extra = {}) {
  const geometry = autoFontSize(content, FONT, {
    x, y, w, h, fontSize: size, minFontSize: size, maxFontSize: size,
    mode: 'auto', margin: 0, padding: 0,
  });
  slide.addText(content, {
    ...geometry, fontFace: FONT, fontSize: size, margin: 0,
    breakLine: false, valign: 'mid', color: '202020',
    ...extra,
  });
}

function frame(slide, record, item, x, y, size) {
  const file = path.join(ROOT, item.path);
  slide.addImage({
    path: file,
    ...imageSizingContain(file, x, y, size, size),
    altText: `ACPD, ${record.view} view, LIBERO-${record.suite}, task ${record.task_id}, episode ${record.episode}, recorded step ${item.index}, ${record.result}`,
  });
}

function notes(slide, records) {
  slide.addNotes([
    'ACPD only. Original closed-loop evaluation videos, not demonstration trajectories.',
    `Random selection seed: ${manifest.selection_seed}. One independent uniform task/episode draw per view-suite cell, without outcome filtering.`,
    'Six uniformly spaced recorded frames include the first and last available frame. Percentages indicate progress through each recorded episode, not matched physical times across episodes.',
    'Success/failure is the original evaluator outcome. The last image is the last recorded pre-action frame.',
    ...records.map(r => `${r.view}/${r.suite}, task ${r.task_id}, episode ${r.episode}, ${r.result}: ${r.instruction}\n${r.source_video}`),
  ].join('\n'));
}

function check(slide) {
  // Images, labels, and headers have separate bounds; there are no intentional overlaps.
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function quadrant(slide, view, index) {
  const x = OUTER + (index % 2) * (QUAD_WIDTH + QUAD_GAP);
  const y = index < 2 ? 0.12 : 12.04;
  const records = manifest.rollouts.filter(r => r.view === view);
  text(slide, `(${String.fromCharCode(97 + index)})  ${VIEWS[view]} view`, x, y, 8.0, 0.62, 38, {
    bold: true, color: COLORS[view],
  });
  text(slide, 'Episode progress →', x + 8.05, y + 0.08, QUAD_WIDTH - 8.05, 0.45, 28, {
    align: 'right', color: '666666',
  });
  PROGRESS.forEach((label, column) => {
    text(slide, label, x + column * (FRAME_SIZE + FRAME_GAP), y + 0.63,
      FRAME_SIZE, 0.43, 27, { align: 'center', color: '666666' });
  });
  records.forEach((record, row) => {
    const rowY = y + 1.10 + row * 2.66;
    text(slide, SUITES[record.suite], x, rowY, 2.06, 0.46, 28, { bold: true });
    text(slide, titles[`${view}_${record.suite}`], x + 2.16, rowY, 8.72, 0.46, 27);
    text(slide, record.result === 'success' ? 'Success' : 'Failure',
      x + 11.06, rowY, QUAD_WIDTH - 11.06, 0.46, 26,
      { align: 'right', color: '666666' });
    record.frames.forEach((item, column) => {
      frame(slide, record, item, x + column * (FRAME_SIZE + FRAME_GAP), rowY + 0.49, FRAME_SIZE);
    });
  });
}

const main = pptx.addSlide();
main.background = { color: 'FFFFFF' };
VIEW_ORDER.forEach((view, index) => quadrant(main, view, index));
notes(main, manifest.rollouts);
check(main);

// Four enlargement pages keep the same sampled data, with exact frame indices.
for (const view of VIEW_ORDER) {
  const slide = pptx.addSlide();
  slide.background = { color: 'FFFFFF' };
  const records = manifest.rollouts.filter(r => r.view === view);
  text(slide, `ACPD · ${VIEWS[view]} view`, OUTER, 0.12, 18, 0.70, 44, {
    bold: true, color: COLORS[view],
  });
  text(slide, 'Episode progression →', 18.7, 0.22, WIDTH - 18.7 - OUTER, 0.52, 32, {
    align: 'right', color: '666666',
  });
  const size = (WIDTH - 2 * OUTER - 5 * 0.08) / 6;
  records.forEach((record, row) => {
    const y = 0.95 + row * 5.66;
    text(slide, SUITES[record.suite], OUTER, y, 3.2, 0.56, 34, { bold: true });
    text(slide, record.instruction, 3.55, y, WIDTH - 3.55 - OUTER, 0.56, 30);
    record.frames.forEach((item, column) => {
      const x = OUTER + column * (size + 0.08);
      const imageY = y + 0.61;
      frame(slide, record, item, x, imageY, size);
      const suffix = column === 5 ? ` · ${record.result === 'success' ? 'Success' : 'Failure'}` : '';
      text(slide, `Step ${item.index}${suffix}`, x, imageY + size + 0.025, size, 0.46, 27, {
        align: 'center', color: '666666', bold: column === 5,
      });
    });
  });
  notes(slide, records);
  check(slide);
}

pptx.writeFile({ fileName: process.argv[2] || path.join(ROOT, 'Figure4_ACPD_rollouts.pptx') });
