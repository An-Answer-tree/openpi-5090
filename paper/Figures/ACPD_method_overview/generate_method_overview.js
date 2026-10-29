// Compact paper Figure 2. Use --full-canvas for the original three-page deck.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { autoFontSize } = require('./helpers/text');
const { imageSizingContain } = require('./helpers/image');
const { warnIfSlideHasOverlaps, warnIfSlideElementsOutOfBounds } = require('./helpers/layout');

const ROOT = __dirname;
const ARGS = process.argv.slice(2);
const W = 16.6666667;
const CONTENT_H = 5.75;
const H = ARGS.includes('--full-canvas') ? 9.375 : CONTENT_H;
const Y = (H - CONTENT_H) / 2;
const FONT = 'Arial';
const MATH = 'Times New Roman';
const C = {
  ink: '26364B', muted: '66788D', border: 'D9E3EC',
  teacher: '356796', teacherFill: 'F4F8FD',
  student: '187C70', studentFill: 'F1F8F5',
  cue: '7B53AC', cueFill: 'F5EFFB',
  acl: 'A97520', aclFill: 'FFF6E6', wrist: 'C79A45',
};
const pptx = new pptxgen();
pptx.defineLayout({ name: 'ACPD_COMPACT', width: W, height: H });
pptx.layout = 'ACPD_COMPACT';
pptx.author = 'ACPD research';
pptx.title = 'ACPD method overview';
pptx.subject = 'Compact teacher/student diagram with embedded cue and velocity supervision';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'en-US' };
const slide = pptx.addSlide();
slide.background = { color: 'FFFFFF' };

function text(content, x, y, w, h, size = 20, extra = {}) {
  const box = autoFontSize(content, extra.fontFace || FONT, {
    x, y: y + Y, w, h, fontSize: size, minFontSize: size, maxFontSize: size,
    bold: !!extra.bold, mode: 'auto', margin: 0, padding: 0,
  });
  slide.addText(content, {
    ...box, fontFace: FONT, fontSize: size, color: C.ink,
    margin: 0, valign: 'mid', wrap: false, breakLine: false,
    paraSpaceAfterPt: 0, ...extra,
  });
}

function rect(x, y, w, h, fill, lineColor = C.border, width = 1, rounded = true) {
  slide.addShape(rounded ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y: y + Y, w, h, radius: 0.10,
    fill: { color: fill }, line: { color: lineColor, width },
  });
}

function line(x1, y1, x2, y2, color = C.ink, arrow = false, dashed = false, width = 1.5) {
  slide.addShape(pptx.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2) + Y,
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: {
      color, width, beginArrowType: 'none', endArrowType: arrow ? 'triangle' : 'none',
      ...(dashed ? { dashType: 'dash' } : {}),
    },
  });
}

function route(points, color, dashed = false, arrow = true, width = 1.5) {
  points.slice(1).forEach((point, i) => {
    line(...points[i], ...point, color, arrow && i === points.length - 2, dashed, width);
  });
}

function circle(cx, cy, diameter, label, color, size = 22) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x: cx - diameter / 2, y: cy - diameter / 2 + Y, w: diameter, h: diameter,
    fill: { color: 'FFFFFF' }, line: { color, width: 1.4 },
  });
  text(label, cx - diameter / 2 + 0.035, cy - diameter / 2 + 0.035,
    diameter - 0.07, diameter - 0.07, size, { color, align: 'center' });
}

function photograph(filename, x, y, size, label, color) {
  const source = path.join(ROOT, 'assets', filename);
  rect(x - 0.02, y - 0.02, size + 0.04, size + 0.04, 'FFFFFF', color, 0.9, false);
  slide.addImage({ path: source, ...imageSizingContain(source, x, y + Y, size, size) });
  text(label, x - 0.08, y + size + 0.055, size + 0.16, 0.28, 18,
    { align: 'center', color });
}

function backbone(y, color) {
  rect(2.65, y, 1.70, 1.04, 'FFFFFF', color, 1.15);
  text('VLA\nbackbone', 2.74, y + 0.14, 1.52, 0.75, 20,
    { align: 'center', bold: true });
}

function actionExpert(y, color) {
  rect(4.80, y, 2.30, 1.82, 'FFFFFF', color, 1.15);
  text('Flow-matching\naction expert', 4.92, y + 0.12, 2.06, 0.64, 21,
    { align: 'center', bold: true });
  rect(4.96, y + 0.98, 1.98, 0.34, 'FFFFFF', color, 0.6, false);
  text('Layer 10', 5.03, y + 1.025, 1.10, 0.25, 17, { bold: true, color });
  for (let i = 0; i < 3; i++) {
    rect(6.19 + i * 0.23, y + 1.065, 0.16, 0.16, color, color, 0, false);
  }
  text('Final action states', 4.96, y + 1.50, 1.98, 0.24, 16.5,
    { align: 'center', color: C.muted });
}

function cue(x, y, predicted, view, color) {
  rect(x, y, 1.73, 0.45, 'FFFFFF', color, 0.85);
  text([
    { text: predicted ? 'Ĉ' : 'C', options: { fontFace: MATH, italic: true } },
    { text: view, options: { subscript: true, fontSize: 14 } },
  ], x + 0.08, y + 0.035, 0.84, 0.35, 23);
  for (let i = 0; i < 4; i++) {
    rect(x + 0.96 + i * 0.16, y + 0.15, 0.11, 0.15, color, color, 0, false);
  }
}

// One continuous frame: no title banner, objective footer or central loss aisle.
// Row backgrounds intentionally contain their editable child modules.
rect(0.12, 0.10, 16.42, 5.53, 'FFFFFF', 'CAD8E4', 0.9, false);
rect(0.16, 0.14, 16.34, 2.63, C.teacherFill, C.teacherFill, 0, false);
rect(0.16, 2.77, 16.34, 2.82, C.studentFill, C.studentFill, 0, false);
line(0.18, 2.77, 16.48, 2.77, 'D6E3E9', false, false, 0.7);
text('Multi-view teacher (frozen)', 0.44, 0.32, 6.10, 0.36, 23,
  { bold: true, color: C.teacher });
text('Single-view student', 0.44, 2.94, 4.15, 0.34, 23,
  { bold: true, color: C.student });
line(12.67, 0.41, 13.00, 0.41, C.ink, true, false, 1.2);
text('Forward', 13.08, 0.27, 1.07, 0.28, 16);
line(14.37, 0.41, 14.70, 0.41, C.cue, true, true, 1.2);
text('Supervision', 14.78, 0.27, 1.47, 0.28, 16);

// Teacher attention supplies the view-specific targets.
photograph('agent.png', 0.44, 1.18, 0.80, 'Agent', C.teacher);
photograph('wrist.png', 1.44, 1.18, 0.80, 'Wrist', C.wrist);
backbone(1.08, C.teacher);
actionExpert(0.75, C.teacher);
line(2.30, 1.60, 2.58, 1.60, C.teacher, true);
line(4.41, 1.60, 4.73, 1.60, C.teacher, true);

rect(7.66, 0.85, 2.17, 1.76, 'FFFFFF', 'C9D9E8', 0.9);
text('Cue extraction', 7.77, 0.98, 1.95, 0.32, 20,
  { bold: true, color: C.teacher, align: 'center' });
const weights = [
  [0.72, 0.44, 0.28, 0.55, 0.12, 0.17, 0.09, 0.13],
  [0.36, 0.62, 0.75, 0.42, 0.19, 0.12, 0.15, 0.08],
  [0.57, 0.21, 0.38, 0.74, 0.14, 0.10, 0.08, 0.16],
];
function tint(hex, amount) {
  return [0, 2, 4].map(i => Math.round(255 + (parseInt(hex.slice(i, i + 2), 16) - 255) * amount)
    .toString(16).padStart(2, '0')).join('');
}
weights.forEach((row, r) => row.forEach((value, column) => {
  // Schematic attention, not measured experimental data.
  const color = column < 2 ? C.teacher : column < 4 ? C.wrist : '718096';
  rect(7.89 + column * 0.20, 1.44 + r * 0.205, 0.17, 0.17,
    tint(color, value), tint(color, value), 0, false);
}));
text('Attention × values', 7.76, 2.10, 1.97, 0.22, 16.5,
  { align: 'center', color: C.muted });
text('Projection + scaling', 7.74, 2.36, 2.01, 0.20, 15.5,
  { align: 'center', color: C.teacher });
line(7.16, 1.90, 7.59, 1.90, C.teacher, true);
rect(10.23, 0.97, 1.93, 1.33, 'F7F9FC', 'E1E9F2', 0.6);
cue(10.33, 1.07, false, 'agent', C.teacher);
cue(10.33, 1.71, false, 'wrist', C.wrist);
route([[9.88, 1.82], [10.04, 1.82], [10.04, 1.30], [10.27, 1.30]], C.teacher);
route([[10.04, 1.82], [10.04, 1.94], [10.27, 1.94]], C.wrist);

// Teacher velocity bypasses extraction; losses occupy the right-hand branch.
route([[7.16, 1.13], [7.38, 1.13], [7.38, 0.70], [15.37, 0.70], [15.37, 1.03]], C.teacher);
text('vᵗ', 15.05, 1.10, 0.64, 0.43, 30,
  { fontFace: MATH, italic: true, color: C.teacher, align: 'center' });
text('Teacher velocity', 14.34, 1.57, 1.99, 0.25, 16,
  { color: C.teacher, align: 'center' });
rect(14.70, 2.05, 1.35, 0.49, C.aclFill, 'E6D4AF', 0.9);
text('ACL', 14.80, 2.12, 1.15, 0.31, 22,
  { bold: true, color: C.acl, align: 'center' });
line(15.37, 1.88, 15.37, 1.99, C.teacher, true, true);

// PCD reads each view separately, before summation for fusion.
rect(12.79, 2.05, 1.35, 0.49, C.cueFill, 'DCCDEE', 0.9);
text('PCD', 12.89, 2.12, 1.15, 0.31, 22,
  { bold: true, color: C.cue, align: 'center' });
route([[12.21, 1.64], [12.48, 1.64], [12.48, 2.30], [12.73, 2.30]], C.cue, true);
route([[12.20, 3.64], [12.32, 3.64], [12.32, 2.65], [13.47, 2.65], [13.47, 2.59]], C.cue, true);

// Student layer-10 prediction and final-state decoding remain separate routes.
photograph('back.png', 0.92, 3.35, 1.04, 'Weak view', C.student);
backbone(3.35, C.student);
actionExpert(3.10, C.student);
line(2.04, 3.87, 2.58, 3.87, C.student, true);
line(4.41, 3.87, 4.73, 3.87, C.student, true);
rect(7.66, 3.73, 1.66, 1.02, 'FFFFFF', C.cue, 1.1);
text('Cue\npredictors', 7.76, 3.87, 1.46, 0.55, 19,
  { align: 'center', bold: true, color: C.cue });
text('LN + affine', 7.75, 4.48, 1.48, 0.20, 16,
  { align: 'center', color: C.muted });
line(7.16, 4.24, 7.59, 4.24, C.cue, true);
text('Predicted cues', 10.21, 3.13, 1.97, 0.27, 18,
  { align: 'center', color: C.cue });
rect(10.23, 3.50, 1.93, 1.33, C.cueFill, 'E5D9F1', 0.6);
cue(10.33, 3.60, true, 'agent', C.cue);
cue(10.33, 4.24, true, 'wrist', C.cue);
route([[9.38, 4.24], [9.88, 4.24], [9.88, 3.83], [10.27, 3.83]], C.cue);
route([[9.88, 4.24], [9.88, 4.47], [10.27, 4.47]], C.cue);

text('Gated cue fusion', 12.62, 3.13, 2.22, 0.28, 19,
  { bold: true, color: C.student, align: 'center' });
route([[12.12, 3.83], [12.40, 3.83], [12.40, 4.15]], C.cue, false, false);
route([[12.12, 4.47], [12.40, 4.47], [12.40, 4.15]], C.cue, false, false);
circle(12.40, 4.15, 0.27, '+', C.cue, 16);
line(12.56, 4.15, 12.73, 4.15, C.cue, true);
rect(12.80, 3.93, 0.46, 0.44, 'FFFFFF', 'AD94C6', 0.8, false);
text('sg', 12.83, 3.98, 0.40, 0.32, 20,
  { fontFace: MATH, italic: true, color: C.cue, align: 'center' });
line(13.31, 4.15, 13.47, 4.15, C.cue, true);
rect(13.54, 3.84, 0.94, 0.62, 'FFFFFF', C.student, 1.1);
text('Gate', 13.61, 3.99, 0.80, 0.30, 20,
  { bold: true, color: C.student, align: 'center' });
route([[14.54, 4.15], [14.59, 4.15], [14.59, 4.72], [14.31, 4.72], [14.31, 4.83]], C.student);
route([[7.16, 4.71], [7.39, 4.71], [7.39, 5.08], [14.02, 5.08]], C.student, false, true, 1.8);
text('Final action representations', 8.06, 5.27, 5.01, 0.24, 18,
  { align: 'center', color: C.student });
circle(14.31, 5.08, 0.43, '+', C.student, 24);
line(14.57, 5.08, 14.73, 5.08, C.student, true);
rect(14.80, 4.68, 0.99, 0.82, 'FFFFFF', C.student, 1.1);
text('Action\noutput', 14.86, 4.82, 0.87, 0.55, 17.5,
  { align: 'center', bold: true });
line(15.84, 5.08, 15.96, 5.08, C.student, true);
text('vˢ', 16.00, 4.86, 0.38, 0.40, 29,
  { fontFace: MATH, italic: true, color: C.student, align: 'center' });
route([[16.19, 4.80], [16.37, 4.80], [16.37, 2.30], [16.12, 2.30]], C.student, true);

slide.addNotes([
  'Compact third-slide replacement: title, shared-input banner and objective/deployment footer removed.',
  'Teacher and student occupy adjacent rows in one continuous frame. PCD and ACL are embedded on the right; there is no full-width loss aisle.',
  'Teacher cues use native action-token attention at layer 10, with original normalization, output projection and time-conditioned scaling.',
  'PCD compares each view separately. The plus sign in the student path is only the sum for cue fusion.',
  'Gate denotes tanh(gamma), initialized at zero. sg blocks action-loss gradients through predicted cues.',
  'Cue predictors and the gate remain active at deployment. ACL uses the same noisy actions and flow times for both models.',
  'Ground-truth flow supervision remains active; its full objective is omitted from this compact main figure.',
  'Attention colors are schematic. Camera inputs are genuine synchronized Spatial images.',
].join('\n'));

// The continuous frame and row backgrounds intentionally contain their modules.
warnIfSlideHasOverlaps(slide, pptx);
warnIfSlideElementsOutOfBounds(slide, pptx);
const target = ARGS.find(arg => !arg.startsWith('--')) || path.join(ROOT, '../Figure2_ACPD_method_overview.pptx');
fs.mkdirSync(path.dirname(path.resolve(target)), { recursive: true });
pptx.writeFile({ fileName: target });
