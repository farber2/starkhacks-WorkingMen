import React, { useMemo } from 'react';
import { pieceFallback, pieceToAsset } from '../lib/pieceMap';

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

function isLastMoveSquare(square, lastMoveUci) {
  if (!lastMoveUci || lastMoveUci.length < 4) {
    return false;
  }
  return square === lastMoveUci.slice(0, 2) || square === lastMoveUci.slice(2, 4);
}

export default function Board({ boardGrid, lastMoveUci }) {
  const flat = useMemo(() => boardGrid?.flat() ?? [], [boardGrid]);

  return (
    <div className="board-shell" role="img" aria-label="Chess board">
      <div className="board-coords files-top">
        {FILES.map((f) => (
          <span key={`top-${f}`}>{f}</span>
        ))}
      </div>

      <div className="board-with-ranks">
        <div className="board-coords ranks-left">
          {[8, 7, 6, 5, 4, 3, 2, 1].map((r) => (
            <span key={`left-${r}`}>{r}</span>
          ))}
        </div>

        <div className="board-grid">
          {flat.map((sq, idx) => {
            const rankIndex = Math.floor(idx / 8);
            const fileIndex = idx % 8;
            const isLight = (rankIndex + fileIndex) % 2 === 0;
            const isRecent = isLastMoveSquare(sq.square, lastMoveUci);
            const img = pieceToAsset(sq.piece);

            return (
              <div
                className={`square ${isLight ? 'light' : 'dark'} ${isRecent ? 'recent' : ''}`}
                key={sq.square}
                title={sq.square}
              >
                {sq.piece ? (
                  img ? (
                    <img
                      src={img}
                      alt={sq.piece}
                      className="piece-img"
                      draggable={false}
                      onError={(e) => {
                        e.currentTarget.style.display = 'none';
                        const fallback = e.currentTarget.nextElementSibling;
                        if (fallback) fallback.classList.remove('hidden');
                      }}
                    />
                  ) : null
                ) : null}
                {sq.piece ? (
                  <span className={`piece-fallback ${img ? 'hidden' : ''}`}>{pieceFallback(sq.piece)}</span>
                ) : null}
              </div>
            );
          })}
        </div>

        <div className="board-coords ranks-right">
          {[8, 7, 6, 5, 4, 3, 2, 1].map((r) => (
            <span key={`right-${r}`}>{r}</span>
          ))}
        </div>
      </div>

      <div className="board-coords files-bottom">
        {FILES.map((f) => (
          <span key={`bottom-${f}`}>{f}</span>
        ))}
      </div>
    </div>
  );
}
