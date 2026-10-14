const $ = id => document.getElementById(id);
let serviceConfig = null;
let task = 'species', selected = null, previewUrl = null, busy = false;
function resetResult() { $('result').hidden = true; $('empty').hidden = false; $('result-badge').textContent = 'Waiting for image'; $('error').hidden = true; }
function buttonText() { $('analyze').textContent = busy ? 'Processing…' : (!serviceConfig ? 'Connecting…' : serviceConfig.tasks[task].mode === 'demo' ? 'Preview demo result ↗' : 'Analyze image ↗'); }
function showError(text) { $('error').textContent = text; $('error').hidden = false; }
function selectFile(file) {
  if (busy) return;
  resetResult();
  selected = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('preview').hidden = true; $('preview').removeAttribute('src'); $('upload-prompt').hidden = false;
  $('filename').textContent = 'No photo selected'; $('clear').hidden = true; $('analyze').disabled = true;
  if (!file) return;
  if (!['image/jpeg','image/png'].includes(file.type) || file.size > 8*1024*1024) { showError('Choose a JPG or PNG image up to 8 MB.'); return; }
  selected = file; previewUrl = URL.createObjectURL(file); $('preview').src = previewUrl;
  $('preview').hidden = false; $('upload-prompt').hidden = true; $('filename').textContent = file.name;
  $('clear').hidden = false; $('analyze').disabled = !serviceConfig;
}
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  task = button.dataset.task;
  document.querySelectorAll('.tab').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  $('guidance').textContent = task === 'species' ? 'Include the entire fish, with its head, fins and tail clearly visible.' : 'Upload a clear close-up of the fish eye. Automatic eye cropping is not included.';
  $('file').value = ''; selectFile(null); updateMode();
}));
$('file').addEventListener('change', e => selectFile(e.target.files[0]));
$('clear').addEventListener('click', () => { $('file').value = ''; selectFile(null); });
['dragover','dragleave','drop'].forEach(event => $('dropzone').addEventListener(event, e => {
  e.preventDefault(); $('dropzone').classList.toggle('drag', event === 'dragover');
  if (event === 'drop' && !busy) { $('file').value = ''; selectFile(e.dataTransfer.files[0]); }
}));
async function post(path, data) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data), signal:AbortSignal.timeout(60000)});
  const output = await response.json(); if (!response.ok) throw new Error(output.error || 'Request failed'); return output;
}
function readBase64(file) { return new Promise((resolve,reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.onerror = () => reject(new Error('Unable to read image')); reader.readAsDataURL(file); }); }
$('analyze').addEventListener('click', async () => {
  if (!selected || busy) return;
  resetResult(); busy = true;
  document.querySelectorAll('button,select,input').forEach(el => el.disabled = true); buttonText();
  try {
    const result = await post('/api/predict/'+task, {image_base64:await readBase64(selected)});
    $('empty').hidden = true; $('result').hidden = false;
    $('result-badge').textContent = result.mode === 'demo' ? 'DEMO · not a prediction' : 'Model output';
    $('result-type').textContent = task === 'species' ? 'SPECIES CLASSIFICATION' : 'VISUAL FRESHNESS';
    $('label').textContent = result.label.replaceAll('_',' '); $('scientific').textContent = result.scientific_name || 'Dataset-defined visual category';
    $('score').textContent = result.score === null ? 'Not available' : result.score.toFixed(3);
    $('version').textContent = 'A model score is not a guarantee of correctness.';
    $('explanation-text').textContent = 'Preparing explanation…'; $('location').textContent = '';
    $('safety').textContent = 'Visual appearance cannot confirm food safety.';
    try {
      const explanation = await post('/api/explain', {result_id:result.id});
      $('explanation-text').textContent = explanation.text; $('location').textContent = explanation.location_note; $('safety').textContent = explanation.safety_note;
    } catch (error) { $('explanation-text').textContent = 'Explanation unavailable. The classification result is still shown above.'; }
  } catch(error) { showError(error.name === 'TimeoutError' ? 'The service timed out. Please try again.' : error.message); }
  finally { busy = false; document.querySelectorAll('button,select,input').forEach(el => el.disabled = false); $('analyze').disabled = !selected; buttonText(); }
});
function updateMode() {
  $('demo-banner').hidden = !serviceConfig || serviceConfig.tasks[task].mode !== 'demo';
  buttonText();
}
fetch('/api/config').then(response => {
  if (!response.ok) throw new Error('Unable to load service configuration.');
  return response.json();
}).then(data => {
  serviceConfig = data; updateMode(); $('analyze').disabled = !selected;
}).catch(() => { showError('Cannot connect to the service. Please refresh to retry.'); $('analyze').disabled = true; });
buttonText();
