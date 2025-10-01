/* Titanic Binary Classifier with TensorFlow.js
   Runs 100% in the browser. No server required.

   Sections implemented:
   1) Load and Inspect CSV (robust parsing with PapaParse)
   2) Preprocessing: impute, standardize, one-hot, optional engineered features
   3) Model: Dense(16,'relu') -> Dense(1,'sigmoid')
   4) Training: stratified 80/20 split, 50 epochs, batch 32, live charts, early stopping (patience=5)
   5) Metrics: ROC curve + AUC, confusion matrix with threshold slider, Precision/Recall/F1
   6) Inference + Export: submission.csv and probabilities.csv; model.save('downloads://...')
   Reuse note: swap SCHEMA.FEATURES and categorical maps to adapt to other datasets.
*/

/* --------------------------- Globals and schema --------------------------- */

const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  featuresCore: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],
  numeric: ['Age','Fare'],
  categorical: {
    Sex: ['male','female'],                  // will be auto-expanded if unseen
    Pclass: ['1','2','3'],                   // treat as string for one-hot stability
    Embarked: ['C','Q','S']
  }
};

let TRAIN_ROWS = [];   // raw objects from train.csv
let TEST_ROWS  = [];   // raw objects from test.csv

// After preprocessing
let FEATURE_NAMES = [];    // final one-hot expanded feature names (train order = test order)
let TRAIN_X, TRAIN_y;      // tensors
let VAL_TRUE = [], VAL_PROB = []; // cached for ROC/metrics
let MODEL = null;

// Standardization stats computed on train split only
const STATS = { mean:{}, std:{} };

/* --------------------------- Utilities --------------------------- */

function alertError(msg, e) {
  console.error(msg, e || '');
  alert(msg);
}

function renderTable(el, rows, limit=10) {
  if (!rows || !rows.length) { el.innerHTML = 'No rows'; return; }
  const cols = Object.keys(rows[0]);
  let html = '<thead><tr>' + cols.map(c=>`<th>${c}</th>`).join('') + '</tr></thead><tbody>';
  for (let i=0; i<Math.min(rows.length,limit); i++) {
    html += '<tr>' + cols.map(c=>`<td>${rows[i][c] ?? ''}</td>`).join('') + '</tr>';
  }
  html += '</tbody>';
  el.innerHTML = html;
}

function percentage(n, d) { return d ? (100*n/d).toFixed(1)+'%' : '0%'; }

function downloadCsv(filename, rows) {
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(',')].concat(
    rows.map(r => cols.map(c => {
      const v = r[c] ?? '';
      const s = String(v);
      return (s.includes(',') || s.includes('"')) ? `"${s.replace(/"/g,'""')}"` : s;
    }).join(','))
  );
  const blob = new Blob([lines.join('\n')], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function setHTML(id, html) { document.getElementById(id).innerHTML = html; }

/* --------------------------- CSV parsing --------------------------- */

function normalizeRow(row) {
  // Turn empty strings / 'NULL' to null, keep others; coerce numeric where clear
  const o = {};
  for (const k of Object.keys(row)) {
    let v = row[k];
    if (v === '' || v === 'NULL' || v === undefined) v = null;
    o[k] = v;
  }
  return o;
}

function parseCsvFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: false, // we will handle typing explicitly
      skipEmptyLines: 'greedy',
      worker: true,
      quoteChar: '"',
      escapeChar: '"',
      complete: r => resolve(r.data.map(normalizeRow)),
      error: reject
    });
  });
}

/* --------------------------- EDA helpers --------------------------- */

function missingPercents(rows) {
  if (!rows.length) return {};
  const cols = Object.keys(rows[0]);
  const out = {};
  for (const c of cols) {
    let miss=0;
    for (const r of rows) if (r[c] === null) miss++;
    out[c] = miss / rows.length;
  }
  return out;
}

// Рисуем столбики прямо в нашем блоке EDA, без правой панели Visor.
// С защитами + текстовый фолбэк, если канвас не отрисовался.
async function renderSimpleBars(containerId, title, labels, values) {
  try { if (window.tfvis?.visor) tfvis.visor().close(); } catch (e) { /* ignore */ }

  const host = document.getElementById(containerId);
  // Заголовок мини-графика
  const cap = document.createElement('div');
  cap.textContent = title;
  cap.style.margin = '6px 0 2px';
  cap.style.fontWeight = '600';
  host.appendChild(cap);

  // Контейнер под график
  const holder = document.createElement('div');
  holder.style.height = '220px';
  holder.style.border = '1px dashed #e5e7eb';
  holder.style.borderRadius = '8px';
  holder.style.padding = '6px';
  holder.style.background = '#fafafa';
  host.appendChild(holder);

  // tfjs-vis для bar ожидает {label, value}; численные значения защищаем
  const clean = labels.map((l, i) => ({
    label: String(l),
    value: Number.isFinite(Number(values[i])) ? Number(values[i]) : 0
  }));

  try {
    await tfvis.render.barchart(
      holder,
      { values: clean },
      { yLabel: 'Rate', height: 200 }
    );
  } catch (e) {
    console.warn('tfjs-vis barchart failed, falling back to text', e);
  }

  // Фолбэк: если график не нарисовался (ни canvas, ни svg), покажем текстом
  if (!holder.querySelector('canvas,svg')) {
    holder.innerHTML = `<pre style="margin:0; font-family:ui-monospace,Menlo,Consolas,monospace">
${clean.map(r => `${r.label}: ${r.value.toFixed ? r.value.toFixed(3) : r.value}`).join('\n')}
</pre>`;
  }
}


function survivalRateBy(rows, key) {
  const counts = {}; // {category: {n:count, s:survivedCount}}
  for (const r of rows) {
    const y = Number(r[SCHEMA.target]);
    if (isNaN(y)) continue;
    let k = r[key];
    if (k === null || k === undefined) k = 'NA';
    k = String(k);
    if (!counts[k]) counts[k] = {n:0, s:0};
    counts[k].n++; counts[k].s += y;
  }
  const labels = Object.keys(counts);
  const values = labels.map(k => counts[k].s / counts[k].n);
  return {labels, values};
}

/* --------------------------- Preprocessing --------------------------- */

function computeMedian(nums) {
  const v = nums.filter(x=>x!==null && !isNaN(Number(x))).map(Number).sort((a,b)=>a-b);
  if (!v.length) return null;
  const m = Math.floor(v.length/2);
  return v.length%2 ? v[m] : (v[m-1]+v[m])/2;
}

function computeMode(arr) {
  const c = new Map();
  for (const x of arr) c.set(x, (c.get(x)||0)+1);
  let best=null, cnt=-1;
  for (const [k,v] of c) if (v>cnt) { best=k; cnt=v; }
  return best;
}

function standardizeInPlace(arr, name) {
  const nums = arr.map(Number);
  const mean = nums.reduce((a,b)=>a+b,0)/nums.length;
  const std = Math.sqrt(nums.reduce((a,b)=>a+(b-mean)**2,0)/nums.length) || 1;
  STATS.mean[name]=mean; STATS.std[name]=std;
}

function standardizeValue(name, val) {
  const mean = STATS.mean[name], std = STATS.std[name] || 1;
  return (Number(val) - mean) / std;
}

function buildFeatureSpace(trainRows, addFamily) {
  // Determine categories from train; keep stable order
  const cats = JSON.parse(JSON.stringify(SCHEMA.categorical));
  for (const key of Object.keys(cats)) {
    const set = new Set(cats[key]);
    for (const r of trainRows) {
      const v = r[key];
      if (v === null) continue;
      set.add(String(v));
    }
    cats[key] = Array.from(set);
  }
  FEATURE_NAMES = [];

  // Optional engineered features
  const engineered = [];
  if (addFamily) { engineered.push('FamilySize','IsAlone'); }

  // Final feature names in fixed order
  // numeric standardized
  for (const n of SCHEMA.numeric) FEATURE_NAMES.push(n+'_z');
  // one-hot categorical
  for (const key of Object.keys(cats)) {
    for (const v of cats[key]) FEATURE_NAMES.push(`${key}=${v}`);
  }
  for (const e of engineered) FEATURE_NAMES.push(e);

  return {cats};
}

function preprocessRows(rows, cats, addFamily, fitStats=false) {
  // If fitStats is true, compute median/mode and standardization over rows
  const out = [];
  const ageCol = rows.map(r => r.Age);
  const ageMedian = computeMedian(ageCol);
  const embarkedMode = computeMode(rows.map(r => r.Embarked ?? 'S')); // default S if all null

  // Build temporary numeric for standardization stats
  const ageFilled = rows.map(r => (r.Age===null?ageMedian:Number(r.Age)));
  const fareFilled = rows.map(r => (r.Fare===null?0:Number(r.Fare)));
  if (fitStats) {
    standardizeInPlace(ageFilled, 'Age');
    standardizeInPlace(fareFilled, 'Fare');
  }

  for (let i=0; i<rows.length; i++) {
    const r = rows[i];
    const feat = {};

    // numeric standardized
    const age = (r.Age===null ? ageMedian : Number(r.Age));
    const fare = (r.Fare===null ? 0 : Number(r.Fare));
    feat['Age_z']  = standardizeValue('Age', age);
    feat['Fare_z'] = standardizeValue('Fare', fare);

    // one-hot categorical
    for (const key of Object.keys(cats)) {
      const catsForKey = cats[key];
      const val = r[key]===null ? 'NA' : String(r[key]);
      for (const v of catsForKey) {
        feat[`${key}=${v}`] = (val===v) ? 1 : 0;
      }
    }

    // engineered
    if (addFamily) {
      const sibsp = Number(r.SibSp ?? 0);
      const parch = Number(r.Parch ?? 0);
      const family = sibsp + parch + 1;
      feat['FamilySize'] = family;
      feat['IsAlone'] = (family === 1) ? 1 : 0;
    }

    out.push(feat);
  }
  return out;
}

function toTensorXY(rowsX, rowsY=null) {
  const x = tf.tensor2d(rowsX.map(feat => FEATURE_NAMES.map(k => feat[k] ?? 0)));
  let y = null;
  if (rowsY) y = tf.tensor2d(rowsY.map(v => [Number(v)]));
  return {x, y};
}

/* --------------------------- Stratified split --------------------------- */

function stratifiedSplit(rowsX, rowsY, valRatio=0.2, seed=42) {
  const pairs = rowsX.map((x,i)=>({x, y:Number(rowsY[i])}));
  const g0 = pairs.filter(p=>p.y===0);
  const g1 = pairs.filter(p=>p.y===1);

  function shuffle(a, seed) {
    let s = seed;
    for (let i=a.length-1; i>0; i--) {
      s = (s*9301 + 49297) % 233280;
      const j = Math.floor((s/233280) * (i+1));
      [a[i], a[j]] = [a[j], a[i]];
    }
  }
  shuffle(g0, seed); shuffle(g1, seed+1);

  const v0 = Math.floor(g0.length*valRatio), v1 = Math.floor(g1.length*valRatio);
  const val = g0.slice(0,v0).concat(g1.slice(0,v1));
  const train = g0.slice(v0).concat(g1.slice(v1));
  shuffle(train, seed+2); shuffle(val, seed+3);

  const xTr = train.map(p=>p.x), yTr = train.map(p=>p.y);
  const xVal = val.map(p=>p.x),   yVal = val.map(p=>p.y);
  return {xTr, yTr, xVal, yVal};
}

/* --------------------------- Model --------------------------- */

function buildModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units:16, activation:'relu', inputShape:[FEATURE_NAMES.length] }));
  m.add(tf.layers.dense({ units:1, activation:'sigmoid' }));
  m.compile({ optimizer:'adam', loss:'binaryCrossentropy', metrics:['accuracy'] });
  return m;
}

function showSummary(model) {
  const lines = [];
  model.summary(80, undefined, x => lines.push(x));
  setHTML('modelSummary', lines.join('\n'));
}

/* --------------------------- Training with early stopping --------------------------- */

async function trainModel(xTr, yTr, xVal, yVal) {
  const surfaceTrain = { name:'Training', tab:'Charts' };
  const surfaceVal = { name:'Validation', tab:'Charts' };

  let bestVal = Infinity, patience=5, wait=0, stopped=false;

  const visCallbacks = tfvis.show.fitCallbacks(surfaceTrain, ['loss','acc'], {callbacks:['onEpochEnd']});

  const history = [];
  await MODEL.fit(xTr, yTr, {
    epochs:50,
    batchSize:32,
    validationData:[xVal, yVal],
    shuffle:true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        history.push({epoch, ...logs});
        await visCallbacks.onEpochEnd(epoch, logs);

        // mirror val curves
        await tfvis.render.linechart(surfaceVal,
          { values: [
              history.map(h=>({x:h.epoch+1, y:h.val_loss, name:'val_loss'})),
              history.map(h=>({x:h.epoch+1, y:h.val_acc,  name:'val_acc'}))
          ], series:['val_loss','val_acc'] },
          { xLabel:'epoch', yLabel:'value', height:260 });

        if (logs.val_loss < bestVal - 1e-6) { bestVal = logs.val_loss; wait=0; }
        else if (++wait >= patience) { stopped=true; MODEL.stopTraining = true; }
      }
    }
  });

  if (stopped) console.log('Early stopping triggered');

  // Cache validation probabilities for metrics/ROC
  const prob = MODEL.predict(xVal);
  VAL_PROB = Array.from(await prob.data());
  VAL_TRUE = Array.from((await yVal.data()));
  prob.dispose();

  setHTML('predInfo', `Validation cached: ${VAL_PROB.length} probabilities.`);
}

/* --------------------------- Metrics: ROC, AUC, confusion --------------------------- */

function rocPoints(yTrue, yProb) {
  // Sort by probability descending
  const pairs = yProb.map((p,i)=>({p, y:yTrue[i]})).sort((a,b)=>b.p-a.p);
  const P = pairs.filter(t=>t.y===1).length;
  const N = pairs.length - P;

  let tp=0, fp=0;
  const pts = [{x:0, y:0}];
  for (const t of pairs) {
    if (t.y===1) tp++; else fp++;
    pts.push({ x: fp/N, y: tp/P });
  }
  pts.push({x:1, y:1});
  return pts;
}

function auc(points) {
  // trapezoidal
  let area=0;
  for (let i=1;i<points.length;i++) {
    const x1=points[i-1].x, y1=points[i-1].y;
    const x2=points[i].x,   y2=points[i].y;
    area += (x2 - x1) * (y1 + y2) / 2;
  }
  return area;
}

async function renderRoc() {
  const pts = rocPoints(VAL_TRUE, VAL_PROB);
  const A = auc(pts);
  await tfvis.render.linechart(
    {name:`ROC (AUC=${A.toFixed(3)})`, tab:'Metrics'},
    {values:[pts], series:['ROC']},
    {xLabel:'FPR', yLabel:'TPR', height:300, xAxisDomain:[0,1], yAxisDomain:[0,1]}
  );
  const surf = document.querySelector('[data-name^="ROC (AUC"]')?.parentElement?.parentElement;
  if (surf) document.getElementById('rocChart').appendChild(surf);
}

function confusionAndPRF(yTrue, yProb, thr) {
  let TP=0, FP=0, TN=0, FN=0;
  for (let i=0;i<yTrue.length;i++) {
    const y = yTrue[i];
    const yhat = (yProb[i] >= thr) ? 1 : 0;
    if (y===1 && yhat===1) TP++;
    else if (y===0 && yhat===1) FP++;
    else if (y===0 && yhat===0) TN++;
    else if (y===1 && yhat===0) FN++;
  }
  const acc = (TP+TN)/(TP+TN+FP+FN);
  const prec = (TP+FP)>0 ? TP/(TP+FP) : 0;
  const rec  = (TP+FN)>0 ? TP/(TP+FN) : 0;
  const f1   = (prec+rec)>0 ? 2*prec*rec/(prec+rec) : 0;
  return {TP,FP,TN,FN, acc, prec, rec, f1};
}

function renderConfusion(thr) {
  const m = confusionAndPRF(VAL_TRUE, VAL_PROB, thr);
  setHTML('confusion',
    `Confusion Matrix\n`+
    `Actual+  Pred+ = ${m.TP} | Pred- = ${m.FN}\n`+
    `Actual-  Pred+ = ${m.FP} | Pred- = ${m.TN}`
  );
  setHTML('prf',
    `Accuracy=${m.acc.toFixed(3)}  Precision=${m.prec.toFixed(3)}  Recall=${m.rec.toFixed(3)}  F1=${m.f1.toFixed(3)}`
  );
}

/* --------------------------- Event handlers --------------------------- */

document.getElementById('trainCsv').addEventListener('change', async e => {
  try {
    TRAIN_ROWS = await parseCsvFile(e.target.files[0]);
    setHTML('previewInfo', `Loaded train.csv: ${TRAIN_ROWS.length} rows`);
    renderTable(document.getElementById('previewTable'), TRAIN_ROWS, 10);
  } catch (err) { alertError('Failed to parse train.csv', err); }
});

document.getElementById('testCsv').addEventListener('change', async e => {
  try {
    TEST_ROWS = await parseCsvFile(e.target.files[0]);
    setHTML('previewInfo', (document.getElementById('previewInfo').innerText || '') + ` | test.csv: ${TEST_ROWS.length} rows`);
  } catch (err) { alertError('Failed to parse test.csv', err); }
});

// Обработчик кнопки "Inspect data"
document.getElementById('btnInspect').addEventListener('click', () => {
  if (!TRAIN_ROWS.length) { alert('Load train.csv first'); return; }

  // Закрыть возможный Visor и очистить левый блок EDA
  try { if (window.tfvis?.visor) tfvis.visor().close(); } catch (_) {}
  const eda = document.getElementById('edaCharts');
  eda.innerHTML = '';

  // Информация о данных (shape + % пропусков)
  const miss = missingPercents(TRAIN_ROWS);
  const shape = `${TRAIN_ROWS.length} rows × ${Object.keys(TRAIN_ROWS[0]).length} cols`;
  setHTML('previewInfo', `Shape: ${shape}\nMissing %: ` + JSON.stringify(miss, null, 2));

  // Доли выживших по Sex и по Pclass
  const bySex   = survivalRateBy(TRAIN_ROWS, 'Sex');     // labels, values
  const byClass = survivalRateBy(TRAIN_ROWS, 'Pclass');  // labels, values

  // На всякий случай лог в консоль (можно удалить)
  console.log('BySex', bySex);
  console.log('ByPclass', byClass);

  // Рисуем два мини-графика в EDA
  renderSimpleBars('edaCharts', 'Survival by Sex',    bySex.labels,   bySex.values);
  renderSimpleBars('edaCharts', 'Survival by Pclass', byClass.labels, byClass.values);
});


document.getElementById('btnPreprocess').addEventListener('click', () => {
  if (!TRAIN_ROWS.length) return alert('Load train.csv first');
  const addFamily = document.getElementById('toggleFamily').checked;

  const {cats} = buildFeatureSpace(TRAIN_ROWS, addFamily);
  // Build X/Y arrays for train
  const rowsY = TRAIN_ROWS.map(r => r[SCHEMA.target]);
  const rowsX = preprocessRows(TRAIN_ROWS, cats, addFamily, true);

  // Tensors
  const {x, y} = toTensorXY(rowsX, rowsY);
  TRAIN_X = x; TRAIN_y = y;

  setHTML('featInfo', `Features (${FEATURE_NAMES.length}):\n` + FEATURE_NAMES.join(', '));
  setHTML('shapeInfo', `X shape: ${TRAIN_X.shape}\nY shape: ${TRAIN_y.shape}\nStandardization: ` + JSON.stringify(STATS, null, 2));

  // Cache encoded test rows for later
  window.__cats = cats; window.__addFamily = addFamily;
});

document.getElementById('btnBuild').addEventListener('click', () => {
  if (!FEATURE_NAMES.length) return alert('Run preprocessing first');
  MODEL?.dispose();
  MODEL = buildModel();
  alert('Model built');
});

document.getElementById('btnSummary').addEventListener('click', () => {
  if (!MODEL) return alert('Build the model first');
  showSummary(MODEL);
});

document.getElementById('btnTrain').addEventListener('click', async () => {
  if (!MODEL || !TRAIN_X || !TRAIN_y) return alert('Build model and preprocess first');

  // Stratified 80/20 split
  const rowsX = TRAIN_X.arraySync().map(row => {
    const obj = {}; FEATURE_NAMES.forEach((k,i)=>obj[k]=row[i]); return obj;
  });
  const rowsY = TRAIN_y.arraySync().map(v => v[0]);
  const {xTr, yTr, xVal, yVal} = stratifiedSplit(rowsX, rowsY, 0.2, 42);
  const tTr = toTensorXY(xTr, yTr), tVal = toTensorXY(xVal, yVal);

  await trainModel(tTr.x, tTr.y, tVal.x, tVal.y);

  await renderRoc();
  renderConfusion(parseFloat(document.getElementById('threshold').value));

  // cleanup temp tensors
  tTr.x.dispose(); tTr.y.dispose(); tVal.x.dispose(); tVal.y.dispose();
});

document.getElementById('threshold').addEventListener('input', e => {
  const thr = parseFloat(e.target.value); setHTML('thrVal', thr.toFixed(2));
  if (VAL_PROB.length) renderConfusion(thr);
});

document.getElementById('btnPredict').addEventListener('click', async () => {
  if (!MODEL || !TEST_ROWS.length) return alert('Need model and test.csv');
  const cats = window.__cats, addFamily = window.__addFamily;
  const preX = preprocessRows(TEST_ROWS, cats, addFamily, false);
  const xTest = tf.tensor2d(preX.map(f => FEATURE_NAMES.map(k=>f[k] ?? 0)));
  const prob = MODEL.predict(xTest);
  const probs = Array.from(await prob.data());
  xTest.dispose(); prob.dispose();

  const thr = parseFloat(document.getElementById('threshold').value);
  const preds = probs.map(p => (p >= thr ? 1 : 0));
  const rows = TEST_ROWS.map((r,i) => ({ PassengerId: r[SCHEMA.id], Survived: preds[i] }));
  const rowsProb = TEST_ROWS.map((r,i) => ({ PassengerId: r[SCHEMA.id], Prob: probs[i] }));

  window.__submission = rows;
  window.__probabilities = rowsProb;
  setHTML('predInfo', `Predicted ${rows.length} rows. Threshold=${thr.toFixed(2)}. Ready to export.`);
});

document.getElementById('btnExport').addEventListener('click', () => {
  if (!window.__submission) return alert('Run Predict first');
  downloadCsv('submission.csv', window.__submission);
  downloadCsv('probabilities.csv', window.__probabilities);
});

document.getElementById('btnSaveModel').addEventListener('click', async () => {
  if (!MODEL) return alert('No model to save');
  await MODEL.save('downloads://titanic-tfjs');
});
