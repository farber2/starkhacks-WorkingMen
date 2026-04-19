const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json'
    },
    ...options
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }

  return response.json();
}

export function getState() {
  return request('/state');
}

export function playMove(uci) {
  return request('/move', {
    method: 'POST',
    body: JSON.stringify({ uci })
  });
}

export function getHelp(speak = false) {
  return request('/help', {
    method: 'POST',
    body: JSON.stringify({ speak })
  });
}

export function speak(text) {
  return request('/speak', {
    method: 'POST',
    body: JSON.stringify({ text })
  });
}

export function resetGame() {
  return request('/reset', { method: 'POST' });
}

export function postVisionBoard(payload = {}) {
  return request('/vision/board', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function postGlassesAudio(payload = {}) {
  return request('/glasses/audio', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function processVision(payload = {}) {
  return request('/vision/process', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function getVisionState() {
  return request('/vision/state');
}

export function resetVision() {
  return request('/vision/reset', { method: 'POST' });
}
