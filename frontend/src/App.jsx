import React, { useEffect, useMemo, useState } from 'react';
import Board from './components/Board';
import {
  getHelp,
  getState,
  playMove,
  postGlassesAudio,
  postVisionBoard,
  resetGame,
  speak
} from './lib/api';

export default function App() {
  const [state, setState] = useState(null);
  const [moveInput, setMoveInput] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [speakHelp, setSpeakHelp] = useState(true);
  const [metaGlassesMode, setMetaGlassesMode] = useState(false);
  const [visionMessage, setVisionMessage] = useState(
    'Current mode: manual move input. Future mode: glasses camera -> board recognition.'
  );
  const [audioMessage, setAudioMessage] = useState(
    'Current route: local speaker (Piper). Future route: Meta glasses audio output.'
  );

  useEffect(() => {
    (async () => {
      try {
        const snapshot = await getState();
        setState(snapshot);
      } catch (err) {
        setError(err.message || 'Failed to load state.');
      }
    })();
  }, []);

  const history = state?.history ?? [];
  const coachText = state?.latest_coach_text ?? '';
  const status = state?.status ?? { engine: 'unknown', local_ai: 'unknown', tts: 'unknown' };

  const turnLabel = useMemo(() => {
    if (!state?.turn) return '-';
    return state.turn === 'white' ? 'White' : 'Black';
  }, [state?.turn]);

  async function handlePlayMove() {
    const uci = moveInput.trim().toLowerCase();
    if (!uci) return;

    setLoading(true);
    setError('');
    try {
      const next = await playMove(uci);
      setState(next);
      setMoveInput('');
    } catch (err) {
      setError(err.message || 'Move failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleHelp() {
    setLoading(true);
    setError('');
    try {
      const next = await getHelp(speakHelp);
      setState(next);
    } catch (err) {
      setError(err.message || 'Help failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleSpeak() {
    if (!coachText) return;
    setLoading(true);
    setError('');
    try {
      await speak(coachText);
    } catch (err) {
      setError(err.message || 'Speak failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleCaptureBoard() {
    setLoading(true);
    setError('');
    try {
      const result = await postVisionBoard({
        source: metaGlassesMode ? 'meta_glasses_mock_capture' : 'manual_mock_capture'
      });
      setVisionMessage(result.message || 'Vision capture placeholder completed.');
    } catch (err) {
      setError(err.message || 'Capture failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleRouteAudio() {
    setLoading(true);
    setError('');
    try {
      const result = await postGlassesAudio({
        event: 'coach_output',
        route: metaGlassesMode ? 'glasses_future' : 'local_speaker',
        text: coachText || 'No coach text yet.'
      });
      setAudioMessage(result.message || 'Audio route placeholder updated.');
    } catch (err) {
      setError(err.message || 'Audio route update failed.');
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    setLoading(true);
    setError('');
    try {
      const next = await resetGame();
      setState(next);
      setMoveInput('');
    } catch (err) {
      setError(err.message || 'Reset failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Local Chess Tutor</h1>
        <div className="top-meta">
          <span>Turn: {turnLabel}</span>
          <span>Result: {state?.result ?? '*'}</span>
          <span className={`glasses-badge ${metaGlassesMode ? 'on' : 'off'}`}>
            Meta Glasses Mode: {metaGlassesMode ? 'On' : 'Off'}
          </span>
        </div>
      </header>

      <main className="layout">
        <section className="board-column card">
          {state?.board ? (
            <Board boardGrid={state.board} lastMoveUci={state.last_move_uci} />
          ) : (
            <div className="loading-board">Loading board...</div>
          )}

          <div className="controls">
            <input
              className="move-input"
              value={moveInput}
              onChange={(e) => setMoveInput(e.target.value)}
              placeholder="Enter UCI move (e2e4)"
              disabled={loading}
            />
            <button onClick={handlePlayMove} disabled={loading || !moveInput.trim()}>
              Play Move
            </button>
            <button onClick={handleHelp} disabled={loading}>
              Help
            </button>
            <button onClick={handleReset} disabled={loading}>
              Reset
            </button>
          </div>

          <div className="glasses-controls">
            <label className="glasses-toggle">
              <input
                type="checkbox"
                checked={metaGlassesMode}
                onChange={(e) => setMetaGlassesMode(e.target.checked)}
              />
              Enable Meta Glasses Mode
            </label>
            <button onClick={handleCaptureBoard} disabled={loading}>
              Capture Board
            </button>
            <button onClick={handleRouteAudio} disabled={loading}>
              Test Audio Route
            </button>
          </div>

          {metaGlassesMode ? (
            <div className="meta-mode-note">
              In future mode, the app would accept board images from Meta AI glasses and return
              spoken coaching.
            </div>
          ) : null}

          <div className="tts-row">
            <label>
              <input
                type="checkbox"
                checked={speakHelp}
                onChange={(e) => setSpeakHelp(e.target.checked)}
              />
              Auto-speak Help
            </label>
            <button onClick={handleSpeak} disabled={loading || !coachText}>
              Speak Coach Text
            </button>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <aside className="side-column">
          <section className="card coach-card">
            <h2>Coach Recommendation</h2>
            <p>{coachText || 'Press Help to get a recommendation for the current position.'}</p>
          </section>

          <section className="card mode-card">
            <h2>Vision Input</h2>
            <p>{visionMessage}</p>
          </section>

          <section className="card mode-card">
            <h2>Audio Route</h2>
            <p>{audioMessage}</p>
          </section>

          <section className="card status-card">
            <h2>System Status</h2>
            <div className="status-grid">
              <StatusPill label="Engine" value={status.engine} />
              <StatusPill label="Local AI" value={status.local_ai} />
              <StatusPill label="TTS" value={status.tts} />
            </div>
          </section>

          <section className="card history-card">
            <h2>Move History</h2>
            {history.length === 0 ? (
              <p className="muted">No moves yet.</p>
            ) : (
              <ol>
                {history.map((entry) => (
                  <li key={entry.ply}>
                    <span className="history-side">{entry.side}</span>
                    <span>{entry.san}</span>
                    <code>{entry.uci}</code>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ label, value }) {
  const ok = value === 'ready';
  return (
    <div className={`status-pill ${ok ? 'ok' : 'warn'}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
