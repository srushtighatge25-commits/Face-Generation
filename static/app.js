const form = document.querySelector('#generateForm');
const generateButton = document.querySelector('#generateButton');
const buttonLabel = generateButton.querySelector('.button-label');
const generatedImage = document.querySelector('#generatedImage');
const emptyState = document.querySelector('#emptyState');
const loadingState = document.querySelector('#loadingState');
const generationNumber = document.querySelector('#generationNumber');
const downloadLink = document.querySelector('#downloadLink');
const toast = document.querySelector('#toast');
const seedInput = document.querySelector('#seedInput');
const randomButton = document.querySelector('#randomButton');
const portraitFrame = document.querySelector('#portraitFrame');
const portraitSeed = document.querySelector('#portraitSeed');
const historyGrid = document.querySelector('#historyGrid');
const historySection = document.querySelector('#historySection');
let generationCount = 0;
let toastTimer;
let selected = null;
let busy = false;
const history = [];

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('visible');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('visible'), 6000);
}

function selectPortrait(portrait, reuseSeed = false) {
  selected = portrait;
  generatedImage.src = portrait.url;
  generatedImage.alt = `Generated portrait with seed ${portrait.seed}`;
  generatedImage.hidden = false;
  emptyState.hidden = true;
  downloadLink.href = portrait.url;
  downloadLink.download = `face-${portrait.seed}.png`;
  downloadLink.hidden = false;
  generationNumber.textContent = `NO. ${String(portrait.number).padStart(2, '0')}`;
  portraitSeed.textContent = `Seed ${portrait.seed}`;
  if (reuseSeed) seedInput.value = portrait.seed;
  for (const item of history) item.button.setAttribute('aria-pressed', String(item === selected));
}

async function generateFace(event) {
  event.preventDefault();
  if (busy) return;
  const seed = seedInput.value.trim();
  if (seed && (!/^\d{1,10}$/.test(seed) || Number(seed) > 4294967295)) {
    showToast('Enter a seed from 0 to 4294967295, or leave it blank.');
    seedInput.focus();
    return;
  }
  busy = true;
  generateButton.disabled = seedInput.disabled = randomButton.disabled = true;
  for (const item of history) item.button.disabled = true;
  buttonLabel.textContent = 'Generating…';
  portraitFrame.setAttribute('aria-busy', 'true');
  emptyState.hidden = generatedImage.hidden = true;
  loadingState.hidden = false;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 60000);
  let pendingUrl;
  try {
    const response = await fetch(seed ? `/generate?seed=${encodeURIComponent(seed)}` : '/generate', {
      method: 'POST', cache: 'no-store', headers: { Accept: 'image/png' }, signal: controller.signal,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || 'The server could not generate a portrait. Please try again.');
    }
    if (!response.headers.get('Content-Type')?.startsWith('image/png')) {
      throw new Error('The server returned an unexpected response. Please try again.');
    }
    const blob = await response.blob();
    pendingUrl = URL.createObjectURL(blob);
    const preview = new Image();
    preview.src = pendingUrl;
    await preview.decode();
    const portrait = { url: pendingUrl, seed: response.headers.get('X-Generation-Seed'), number: ++generationCount };
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'history-item';
    button.setAttribute('aria-label', `View portrait ${portrait.number}, seed ${portrait.seed}`);
    preview.alt = '';
    button.append(preview);
    const caption = document.createElement('span');
    caption.textContent = `#${portrait.number}`;
    button.append(caption);
    button.addEventListener('click', () => { if (!busy) selectPortrait(portrait, true); });
    portrait.button = button;
    history.unshift(portrait);
    historyGrid.prepend(button);
    selectPortrait(portrait);
    pendingUrl = null;
    if (history.length > 8) {
      const removed = history.pop();
      removed.button.remove();
      URL.revokeObjectURL(removed.url);
    }
    historySection.hidden = false;
    showToast(`Portrait ready. Seed ${portrait.seed}.`);
  } catch (error) {
    if (selected) selectPortrait(selected);
    else emptyState.hidden = false;
    showToast(error.name === 'AbortError' ? 'Generation took too long. Please try again.' : error.message || 'Connection failed. Please try again.');
  } finally {
    if (pendingUrl) URL.revokeObjectURL(pendingUrl);
    window.clearTimeout(timeout);
    loadingState.hidden = true;
    portraitFrame.setAttribute('aria-busy', 'false');
    generateButton.disabled = seedInput.disabled = randomButton.disabled = false;
    for (const item of history) item.button.disabled = false;
    buttonLabel.textContent = selected ? 'Generate another' : 'Generate a face';
    busy = false;
  }
}
form.addEventListener('submit', generateFace);
randomButton.addEventListener('click', () => { seedInput.value = ''; seedInput.focus(); });
