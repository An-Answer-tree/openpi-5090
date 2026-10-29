// Figure 2 redesign. All modules, arrows, feature strips and labels are editable.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { autoFontSize } = require('./helpers/text');
const { imageSizingContain } = require('./helpers/image');
const { warnIfSlideHasOverlaps, warnIfSlideElementsOutOfBounds } = require('./helpers/layout');

const ROOT = __dirname;
const W = 16.6666667;
const H = 9.375;
const FONT = 'Liberation Sans';
const MATH = 'Liberation Serif';
const C = {
  ink: '233348', muted: '64748B', border: 'D8E1EA',
  teacher: '346594', teacherFill: 'F3F7FC',
  student: '167B70', studentFill: 'F0F8F5',
  cue: '7952A8', cueFill: 'F6F1FB',
  acl: 'A46C19', aclFill: 'FFF7E7', wrist: 'C79742',
};
const pptx = new pptxgen();
pptx.defineLayout({ name: 'ORIGINAL_CANVAS', width: W, height: H });
pptx.layout = 'ORIGINAL_CANVAS';
pptx.author = 'ACPD research';
pptx.title = 'ACPD: action-conditioned privileged distillation';
pptx.subject = 'Updated method overview with teacher-native cues and gated student fusion';
pptx.lang = 'en-US';
pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'en-US' };
const slide = pptx.addSlide();
slide.background = { color: 'FFFFFF' };

function text(content, x, y, w, h, size = 20, extra = {}) {
  const box = autoFontSize(content, extra.fontFace || FONT, {
    x, y, w, h, fontSize: size, minFontSize: size, maxFontSize: size,
    mode: 'auto', margin: 0, padding: 0,
  });
  slide.addText(content, {
    ...box, fontFace: FONT, fontSize: size, color: C.ink,
    margin: 0, valign: 'mid', breakLine: false, ...extra,
  });
}

function rect(x, y, w, h, fill, line = C.border, width = 1.0, rounded = true) {
  slide.addShape(rounded ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h, radius: 0.12,
    rectRadius: 0.12, fill: { color: fill },
    line: { color: line, width },
  });
}

function line(x1, y1, x2, y2, color = C.ink, arrow = false, dashed = false, width = 1.5) {
  slide.addShape(pptx.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: {
      color, width, beginArrowType: 'none',
      endArrowType: arrow ? 'triangle' : 'none',
      ...(dashed ? { dashType: 'dash' } : {}),
    },
  });
}

function route(points, color, dashed = false, arrow = true, width = 1.5) {
  points.slice(1).forEach((point, i) => {
    line(...points[i], ...point, color, arrow && i === points.length - 2, dashed, width);
  });
}

function circle(x, y, diameter, label, color, fill = 'FFFFFF', size = 22) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: diameter, h: diameter,
    fill: { color: fill }, line: { color, width: 1.5 },
  });
  text(label, x + 0.03, y + 0.03, diameter - 0.06, diameter - 0.06, size,
    { align: 'center', color });
}

function photograph(filename, x, y, size, label, color) {
  const source = path.join(ROOT, 'assets', filename);
  // A slightly larger frame contains the image, avoiding a duplicate overlay.
  slide.addShape(pptx.ShapeType.rect, {
    x: x - 0.02, y: y - 0.02, w: size + 0.04, h: size + 0.04,
    fill: { color: 'FFFFFF', transparency: 100 }, line: { color, width: 1.0 },
  });
  slide.addImage({ path: source, ...imageSizingContain(source, x, y, size, size) });
  text(label, x - 0.08, y + size + 0.065, size + 0.16, 0.27, 18,
    { align: 'center', color });
}

function featureStrip(x, y, color) {
  rect(x, y, 2.14, 0.31, 'FFFFFF', color, 0.7, false);
  text('Layer 10', x + 0.07, y + 0.015, 0.90, 0.27, 16,
    { bold: true, color });
  for (let i = 0; i < 4; i++) {
    rect(x + 1.04 + i * 0.25, y + 0.062, 0.18, 0.18, color, color, 0, false);
  }
}

function actionExpert(x, y, color) {
  rect(x, y, 2.50, 1.64, 'FFFFFF', color, 1.25);
  text('Flow-matching\naction expert', x + 0.12, y + 0.13, 2.26, 0.65, 22,
    { align: 'center', bold: true });
  featureStrip(x + 0.18, y + 0.90, color);
  text('Final action states', x + 0.17, y + 1.30, 2.16, 0.25, 17,
    { align: 'center', color: C.muted });
}

function cue(x, y, predicted, view, color) {
  rect(x, y, 1.63, 0.42, 'FFFFFF', color, 0.9);
  text([
    { text: predicted ? 'Ĉ' : 'C', options: { fontFace: MATH, italic: true } },
    { text: view, options: { fontSize: 13.5, subscript: true } },
  ], x + 0.09, y + 0.02, 0.72, 0.36, 22);
  for (let i = 0; i < 4; i++) {
    rect(x + 0.89 + 0.16 * i, y + 0.135, 0.11, 0.15, color, color, 0, false);
  }
}

// Restrained header and a shared-input strip avoid repeating conditions in each branch.
text('Action-conditioned privileged distillation', 0.49, 0.20, 11.95, 0.42, 26,
  { bold: true });
line(12.80, 0.40, 13.25, 0.40, C.ink, true);
text('Forward', 13.36, 0.25, 1.02, 0.29, 16);
line(14.55, 0.40, 15.00, 0.40, C.cue, true, true);
text('Supervision', 15.10, 0.25, 1.31, 0.29, 16);
rect(0.27, 0.76, 16.13, 0.36, 'F7F9FB', 'F7F9FB', 0);
text('Shared instruction, robot state, noisy action chunk and flow time',
  0.47, 0.79, 15.70, 0.28, 17, { color: C.muted });

// Lane containers intentionally contain their editable modules and connectors.
rect(0.27, 1.31, 16.13, 2.62, C.teacherFill, 'D7E3F0', 0.9);
rect(0.27, 4.91, 16.13, 2.84, C.studentFill, 'CCE3DC', 0.9);
text('FROZEN MULTI-VIEW TEACHER', 0.51, 1.47, 7.8, 0.32, 21,
  { bold: true, color: C.teacher });
text('TRAINABLE SINGLE-VIEW STUDENT', 0.51, 5.08, 8.2, 0.32, 21,
  { bold: true, color: C.student });

// Teacher backbone and action expert.
photograph('agent.png', 0.52, 2.10, 0.80, 'Agent', C.teacher);
photograph('wrist.png', 1.48, 2.10, 0.80, 'Wrist', C.wrist);
rect(2.76, 2.12, 1.62, 1.03, 'FFFFFF', C.teacher, 1.25);
text('VLA\nbackbone', 2.84, 2.24, 1.46, 0.77, 20,
  { align: 'center', bold: true });
actionExpert(4.94, 1.98, C.teacher);
line(2.33, 2.64, 2.69, 2.64, C.teacher, true);
line(4.43, 2.64, 4.86, 2.64, C.teacher, true);
line(7.48, 2.37, 7.99, 2.37, C.teacher, true);
text('vᵗ', 8.08, 2.13, 0.70, 0.42, 30,
  { fontFace: MATH, italic: true, color: C.teacher, align: 'center' });
text('Teacher velocity', 7.59, 2.61, 1.78, 0.28, 16, { color: C.teacher, align: 'center' });

// Native attention extraction replaces the old trainable cross-attention selector.
rect(9.56, 1.97, 4.46, 1.71, 'FFFFFF', 'C9D7E7', 0.9);
text('Teacher-native cue extraction', 9.73, 2.10, 4.12, 0.32, 20,
  { bold: true, color: C.teacher });
const attention = [
  [0.72, 0.44, 0.28, 0.55, 0.12, 0.17, 0.09, 0.13],
  [0.36, 0.62, 0.75, 0.42, 0.19, 0.12, 0.15, 0.08],
  [0.57, 0.21, 0.38, 0.74, 0.14, 0.10, 0.08, 0.16],
];
function tint(hex, amount) {
  return [0, 2, 4].map(i => Math.round(255 + (parseInt(hex.slice(i, i + 2), 16) - 255) * amount)
    .toString(16).padStart(2, '0')).join('');
}
attention.forEach((row, r) => row.forEach((value, column) => {
  const color = column < 2 ? C.teacher : column < 4 ? C.wrist : '718096';
  // Schematic matrix, not measured attention values.
  rect(9.78 + column * 0.179, 2.66 + r * 0.21, 0.154, 0.18,
    tint(color, value), tint(color, value), 0, false);
}));
text('Per-action attention', 9.64, 3.34, 2.15, 0.24, 16, { align: 'center', color: C.muted });
rect(12.02, 2.51, 1.83, 1.14, 'F6F8FC', 'E4EAF3', 0.6);
cue(12.12, 2.59, false, 'agent', C.teacher);
cue(12.12, 3.15, false, 'wrist', C.wrist);
route([[11.30, 2.96], [11.68, 2.96], [11.68, 2.80], [12.04, 2.80]], C.teacher);
route([[11.68, 2.96], [11.68, 3.36], [12.04, 3.36]], C.wrist);
route([[7.45, 3.04], [9.18, 3.04], [9.18, 2.97], [9.50, 2.97]], C.teacher);
text('Attention\n& values', 7.72, 3.20, 1.70, 0.55, 17,
  { align: 'center', color: C.teacher });
text('View-wise values → projection × scale', 9.73, 3.73, 4.12, 0.19, 15.5,
  { align: 'center', color: C.teacher });

// Teacher output feeds ACL along the upper-right perimeter, away from cue extraction.
route([[8.43, 2.11], [8.43, 1.87], [15.02, 1.87], [15.02, 4.08]], C.teacher, true);

// PCD is a training-only comparison of the view-matched targets and predictions.
rect(9.47, 4.19, 3.24, 0.51, C.cueFill, 'D8CAE8', 0.9);
text('PCD · cue regression', 9.60, 4.26, 2.98, 0.34, 20,
  { bold: true, color: C.cue, align: 'center' });
// The loss reads each view separately; its input is the pair, not their sum.
route([[13.89, 3.08], [14.15, 3.08], [14.15, 4.44], [12.77, 4.44]], C.cue, true);
route([[11.51, 5.94], [11.57, 5.94], [11.57, 4.76]], C.cue, true);

// Student trunk and the two routes from selected/final action representations.
photograph('back.png', 0.78, 5.97, 1.08, 'Weak view', C.student);
rect(2.76, 5.97, 1.62, 1.03, 'FFFFFF', C.student, 1.25);
text('VLA\nbackbone', 2.84, 6.09, 1.46, 0.77, 20,
  { align: 'center', bold: true });
actionExpert(4.94, 5.57, C.student);
line(1.94, 6.49, 2.69, 6.49, C.student, true);
line(4.43, 6.49, 4.86, 6.49, C.student, true);
rect(7.93, 5.98, 1.50, 1.00, 'FFFFFF', C.cue, 1.2);
text('Cue\npredictors', 7.99, 6.09, 1.38, 0.56, 18,
  { align: 'center', bold: true, color: C.cue });
text('LN + affine', 7.99, 6.70, 1.38, 0.20, 15.5,
  { align: 'center', color: C.muted });
line(7.48, 6.49, 7.86, 6.49, C.cue, true);
rect(9.68, 5.88, 1.83, 1.16, C.cueFill, 'E6DAF0', 0.6);
cue(9.78, 5.95, true, 'agent', C.cue);
cue(9.78, 6.55, true, 'wrist', C.cue);
route([[9.44, 6.48], [9.57, 6.48], [9.57, 6.16], [9.73, 6.16]], C.cue);
route([[9.57, 6.48], [9.57, 6.76], [9.73, 6.76]], C.cue);
text('Predicted cues', 9.71, 5.57, 1.80, 0.28, 18,
  { align: 'center', color: C.cue });

// GCF: sum cues -> stop gradient -> learned scalar -> add to final hidden state.
text('Gated cue fusion', 11.73, 5.49, 2.65, 0.31, 21,
  { bold: true, color: C.student, align: 'center' });
route([[11.46, 6.16], [11.66, 6.16], [11.66, 6.48]], C.cue, false, false);
route([[11.46, 6.76], [11.66, 6.76], [11.66, 6.48]], C.cue, false, false);
circle(11.53, 6.35, 0.26, '+', C.cue, 'FFFFFF', 16);
line(11.66, 6.48, 11.93, 6.48, C.cue, true);
rect(11.99, 6.26, 0.44, 0.44, 'FFFFFF', '9A7EBA', 0.9, false);
text('sg', 12.00, 6.29, 0.42, 0.34, 20,
  { align: 'center', fontFace: MATH, italic: true, color: C.cue });
line(12.47, 6.48, 12.66, 6.48, C.cue, true);
rect(12.73, 6.14, 1.02, 0.68, 'FFFFFF', C.student, 1.2);
text('tanh γ', 12.78, 6.29, 0.92, 0.34, 23,
  { align: 'center', fontFace: MATH, color: C.student });
text('learned gate', 12.64, 6.90, 1.20, 0.23, 15.5,
  { align: 'center', color: C.student });
route([[13.81, 6.48], [14.22, 6.48], [14.22, 6.91]], C.student);
line(7.48, 7.15, 14.00, 7.15, C.student, true, false, 1.8);
text('Final action representations', 8.13, 7.31, 4.34, 0.26, 18,
  { align: 'center', color: C.student });
circle(14.01, 6.94, 0.42, '+', C.student, 'FFFFFF', 24);
line(14.48, 7.15, 14.64, 7.15, C.student, true);
rect(14.67, 6.73, 1.00, 0.84, 'FFFFFF', C.student, 1.2);
text('Action\noutput', 14.71, 6.87, 0.92, 0.55, 17.5,
  { align: 'center', bold: true });
line(15.72, 7.15, 15.86, 7.15, C.student, true);
text('vˢ', 15.89, 6.91, 0.35, 0.42, 29,
  { fontFace: MATH, italic: true, color: C.student, align: 'center' });

// ACL compares the velocity predictions, not actions from different noise draws.
rect(14.39, 4.15, 1.94, 0.62, C.aclFill, 'E6D4AF', 0.9);
text('ACL · Pearson\ncorrelation', 14.48, 4.22, 1.76, 0.48, 18,
  { bold: true, color: C.acl, align: 'center' });
route([[16.02, 6.86], [16.23, 6.86], [16.23, 4.84]], C.student, true);

// Compact objective/footer. Cue prediction and GCF remain active at deployment.
rect(0.27, 8.02, 16.13, 1.07, 'F7F9FB', 'E2E8EE', 0.9);
text('Training objective', 0.52, 8.20, 3.00, 0.34, 21, { bold: true });
text('Ground-truth flow supervision', 0.52, 8.63, 3.20, 0.24, 16.5,
  { color: C.muted });
text([
  { text: 'L = L', options: { fontFace: MATH, italic: true } },
  { text: 'flow', options: { fontFace: MATH, subscript: true, fontSize: 18 } },
  { text: ' + λ', options: { fontFace: MATH, italic: true, color: C.cue } },
  { text: 'cue', options: { fontFace: MATH, subscript: true, fontSize: 18, color: C.cue } },
  { text: 'L', options: { fontFace: MATH, italic: true, color: C.cue } },
  { text: 'cue', options: { fontFace: MATH, subscript: true, fontSize: 18, color: C.cue } },
  { text: ' + λ', options: { fontFace: MATH, italic: true, color: C.acl } },
  { text: 'ACL', options: { fontFace: MATH, subscript: true, fontSize: 18, color: C.acl } },
  { text: 'L', options: { fontFace: MATH, italic: true, color: C.acl } },
  { text: 'ACL', options: { fontFace: MATH, subscript: true, fontSize: 18, color: C.acl } },
], 4.00, 8.18, 7.35, 0.56, 29);
text('sg blocks action-loss gradients through predicted cues.',
  4.01, 8.77, 7.50, 0.20, 16.2, { color: C.muted });
rect(12.03, 8.18, 4.12, 0.73, 'EAF4F0', 'D1E5DD', 0.7);
text('Deployment: student only', 12.22, 8.27, 3.74, 0.28, 20,
  { bold: true, color: C.student, align: 'center' });
text('Cue prediction and gate retained', 12.20, 8.61, 3.78, 0.22, 16.5,
  { color: C.student, align: 'center' });

slide.addNotes([
  'Figure 2: current ACPD method, replacing the previous learned cross-attention selector diagram.',
  'The teacher is frozen and uses agent and wrist views. The student uses a single weak view.',
  'Teacher cues are extracted from native action-token attention at layer 10, retaining original attention weights, output projection and time-conditioned scaling.',
  'View-specific affine predictors use normalized student layer-10 action states. PCD is normalized per-view regression.',
  'GCF adds tanh(gamma) times the sum of detached predicted cues to final student action states before the existing action output projection.',
  'ACL uses Pearson correlation of teacher and student flow velocities at identical noisy actions and flow times. The ground-truth flow objective remains active.',
  'The teacher is removed at deployment. Cue predictors and the learned gate remain in the student.',
  'Attention cells are a schematic illustration, not measured attention values. Camera inputs are genuine synchronized LIBERO-Spatial images.',
].join('\n'));

// Containers, photo borders and the fusion join intentionally contain their labels.
warnIfSlideHasOverlaps(slide, pptx);
warnIfSlideElementsOutOfBounds(slide, pptx);
const target = process.argv[2] || path.join(ROOT, '../Figure2_ACPD_method_overview.pptx');
fs.mkdirSync(path.dirname(path.resolve(target)), { recursive: true });
pptx.writeFile({ fileName: target });
